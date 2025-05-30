#!/usr/bin/env python3
"""
Phase 1: データ品質確認スクリプト
小学校出前授業アンケートデータの品質チェックを実施

実施項目:
1. 欠損値パターンの確認（MCAR/MAR/MNAR判定）
2. 回答の一貫性チェック（論理的矛盾の検出）
3. 外れ値・異常値の検出
4. Page_IDの重複確認
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUIバックエンドを使用しない
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from pathlib import Path
import json
from datetime import datetime

# データディレクトリの設定
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "analysis"
REPORT_DIR = BASE_DIR / "reports" / datetime.now().strftime("%Y-%m-%d")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 出力設定
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)


def load_data():
    """データファイルを読み込む"""
    before_df = pd.read_csv(DATA_DIR / "before_excel_compliant.csv")
    after_df = pd.read_csv(DATA_DIR / "after_excel_compliant.csv")
    comment_df = pd.read_csv(DATA_DIR / "comment.csv")
    
    print(f"授業前アンケート: {before_df.shape}")
    print(f"授業後アンケート: {after_df.shape}")
    print(f"感想文: {comment_df.shape}")
    
    return before_df, after_df, comment_df


def check_missing_values(df, df_name):
    """欠損値パターンの確認"""
    print(f"\n=== {df_name} 欠損値分析 ===")
    
    # 欠損値のサマリー
    missing_summary = pd.DataFrame({
        '欠損数': df.isnull().sum(),
        '欠損率(%)': (df.isnull().sum() / len(df) * 100).round(2)
    })
    missing_summary = missing_summary[missing_summary['欠損数'] > 0].sort_values('欠損数', ascending=False)
    
    if len(missing_summary) > 0:
        print("\n欠損値が存在する列:")
        print(missing_summary)
        
        # 欠損値パターンの可視化
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
        plt.title(f'{df_name}: 欠損値パターン')
        plt.tight_layout()
        plt.savefig(REPORT_DIR / f'missing_pattern_{df_name}.png', dpi=300)
        plt.close()
        
        # 欠損値の共起パターン分析
        missing_cols = missing_summary.index.tolist()
        if len(missing_cols) > 1:
            missing_corr = df[missing_cols].isnull().corr()
            plt.figure(figsize=(8, 6))
            sns.heatmap(missing_corr, annot=True, cmap='coolwarm', center=0)
            plt.title(f'{df_name}: 欠損値の相関')
            plt.tight_layout()
            plt.savefig(REPORT_DIR / f'missing_correlation_{df_name}.png', dpi=300)
            plt.close()
    else:
        print("欠損値なし")
    
    return missing_summary


def check_page_id_consistency(before_df, after_df):
    """Page_IDの一貫性チェック"""
    print("\n=== Page_ID 一貫性チェック ===")
    
    # 重複チェック
    before_duplicates = before_df['Page_ID'].duplicated().sum()
    after_duplicates = after_df['Page_ID'].duplicated().sum()
    
    print(f"授業前アンケート重複ID数: {before_duplicates}")
    print(f"授業後アンケート重複ID数: {after_duplicates}")
    
    # マッチングチェック
    before_ids = set(before_df['Page_ID'])
    after_ids = set(after_df['Page_ID'])
    
    only_before = before_ids - after_ids
    only_after = after_ids - before_ids
    matched = before_ids & after_ids
    
    print(f"\nマッチング状況:")
    print(f"両方に存在するID数: {len(matched)}")
    print(f"授業前のみ存在: {len(only_before)}")
    print(f"授業後のみ存在: {len(only_after)}")
    
    if only_before:
        print(f"授業前のみのID例: {list(only_before)[:5]}")
    if only_after:
        print(f"授業後のみのID例: {list(only_after)[:5]}")
    
    return {
        'matched_count': len(matched),
        'only_before': list(only_before),
        'only_after': list(only_after),
        'before_duplicates': before_duplicates,
        'after_duplicates': after_duplicates
    }


def check_logical_consistency(df, df_name):
    """論理的一貫性のチェック"""
    print(f"\n=== {df_name} 論理的一貫性チェック ===")
    
    inconsistencies = []
    
    # Q1の水溶液認識チェック（授業前後共通）
    q1_cols = [col for col in df.columns if col.startswith('Q1_')]
    if q1_cols:
        # 実際にデータがある行のみチェック
        q1_data = df[q1_cols].dropna(how='all')
        if len(q1_data) > 0:
            # すべて同じ回答のケースをチェック
            all_same = q1_data.apply(lambda row: len(set(row.dropna())) == 1 if len(row.dropna()) > 1 else False, axis=1)
            if all_same.sum() > 0:
                inconsistencies.append({
                    'type': 'Q1全て同じ回答',
                    'count': all_same.sum(),
                    'percentage': (all_same.sum() / len(df) * 100).round(2)
                })
    
    # Q3のお茶理解度チェック（授業前後共通）
    q3_cols = [col for col in df.columns if col.startswith('Q3_')]
    if len(q3_cols) == 2:
        q3_data = df[q3_cols].dropna(how='all')
        if len(q3_data) > 0:
            # 矛盾する回答パターンのチェック（両方「そう思う」など）
            # 実際の値を確認してからチェック
            pass
    
    # 授業後特有のチェック
    if df_name == '授業後':
        # Q4とQ6の評価値の矛盾チェック（数値評価）
        if 'Q4_ExperimentInterestRating' in df.columns and 'Q6_DissolvingUnderstandingRating' in df.columns:
            # 実験に興味なし（1,2）だが理解度が高い（4）ケース
            valid_data = df[['Q4_ExperimentInterestRating', 'Q6_DissolvingUnderstandingRating']].dropna()
            if len(valid_data) > 0:
                contradictory_interest = (
                    (valid_data['Q4_ExperimentInterestRating'] <= 2) & 
                    (valid_data['Q6_DissolvingUnderstandingRating'] == 4)
                )
                if contradictory_interest.sum() > 0:
                    inconsistencies.append({
                        'type': '低興味×高理解度',
                        'count': contradictory_interest.sum(),
                        'percentage': (contradictory_interest.sum() / len(valid_data) * 100).round(2)
                    })
    
    if inconsistencies:
        print("\n検出された論理的矛盾:")
        for inc in inconsistencies:
            print(f"- {inc['type']}: {inc['count']}件 ({inc['percentage']}%)")
    else:
        print("論理的矛盾は検出されませんでした")
    
    return inconsistencies


def check_outliers(df, df_name):
    """外れ値・異常値の検出"""
    print(f"\n=== {df_name} 外れ値・異常値チェック ===")
    
    outliers = []
    
    # 数値評価の範囲チェック（授業後のみ）
    if df_name == '授業後':
        rating_cols = ['Q4_ExperimentInterestRating', 'Q5_NewLearningsRating', 'Q6_DissolvingUnderstandingRating']
        for col in rating_cols:
            if col in df.columns:
                valid_data = df[col].dropna()
                if len(valid_data) > 0:
                    out_of_range = valid_data[(valid_data < 1) | (valid_data > 4)]
                    if len(out_of_range) > 0:
                        outliers.append({
                            'column': col,
                            'out_of_range_count': len(out_of_range),
                            'values': out_of_range.value_counts().to_dict()
                        })
                        print(f"\n{col}の範囲外の値: {len(out_of_range)}件")
    
    # classの値チェック
    if 'class' in df.columns:
        valid_classes = df['class'].dropna()
        if len(valid_classes) > 0:
            unexpected_classes = valid_classes[~valid_classes.isin([1.0, 2.0, 3.0, 4.0])]
            if len(unexpected_classes) > 0:
                outliers.append({
                    'column': 'class',
                    'unexpected_values': unexpected_classes.value_counts().to_dict()
                })
                print(f"\nclassの予期しない値: {len(unexpected_classes)}件")
    
    # テキストデータの長さチェック
    text_cols_mapping = {
        'Q2_MisoSalty_Reason': 'みそ汁理由（授業前）',
        'Q2_MisoSaltyReason': 'みそ汁理由（授業後）',
        'Q5_NewLearningsRating': '新しい学び（評価）',
        'GeneralPageComments': '感想'
    }
    
    for col, col_name in text_cols_mapping.items():
        if col in df.columns:
            # テキスト列の場合のみ長さチェック
            if df[col].dtype == 'object':
                text_data = df[col].dropna()
                if len(text_data) > 0:
                    text_lengths = text_data.str.len()
                    if len(text_lengths) > 0:
                        # 極端に短い・長いテキストの検出
                        q1, q3 = text_lengths.quantile([0.25, 0.75])
                        iqr = q3 - q1
                        lower_bound = max(1, q1 - 1.5 * iqr)
                        upper_bound = q3 + 1.5 * iqr
                        
                        outlier_short = (text_lengths < lower_bound).sum()
                        outlier_long = (text_lengths > upper_bound).sum()
                        
                        if outlier_short > 0 or outlier_long > 0:
                            outliers.append({
                                'column': col_name,
                                'short_outliers': outlier_short,
                                'long_outliers': outlier_long,
                                'median_length': text_lengths.median()
                            })
                            print(f"\n{col_name}の文字数統計:")
                            print(f"  中央値: {text_lengths.median():.0f}文字")
                            print(f"  極端に短い回答: {outlier_short}件")
                            print(f"  極端に長い回答: {outlier_long}件")
    
    return outliers


def generate_quality_report(before_df, after_df, comment_df, results):
    """品質チェックレポートの生成"""
    report = {
        'generated_at': datetime.now().isoformat(),
        'data_summary': {
            'before_survey': {
                'total_records': len(before_df),
                'columns': list(before_df.columns)
            },
            'after_survey': {
                'total_records': len(after_df),
                'columns': list(after_df.columns)
            },
            'comments': {
                'total_records': len(comment_df),
                'columns': list(comment_df.columns)
            }
        },
        'quality_checks': results
    }
    
    # JSON形式で保存（DataFrameを辞書に変換、numpy型を通常の型に変換）
    def convert_to_json_serializable(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    json_report = convert_to_json_serializable(report)
    
    with open(REPORT_DIR / 'phase1_quality_report.json', 'w', encoding='utf-8') as f:
        json.dump(json_report, f, ensure_ascii=False, indent=2)
    
    # テキストレポートの生成
    with open(REPORT_DIR / 'phase1_quality_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Phase 1: データ品質確認レポート\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("## データサマリー\n")
        f.write(f"- 授業前アンケート: {len(before_df)}件\n")
        f.write(f"- 授業後アンケート: {len(after_df)}件\n")
        f.write(f"- 感想文: {len(comment_df)}件\n\n")
        
        f.write("## Page_ID一貫性\n")
        page_id_results = results['page_id_consistency']
        f.write(f"- マッチング数: {page_id_results['matched_count']}件\n")
        f.write(f"- 授業前のみ: {len(page_id_results['only_before'])}件\n")
        f.write(f"- 授業後のみ: {len(page_id_results['only_after'])}件\n\n")
        
        f.write("## 欠損値分析\n")
        for key in ['before_missing', 'after_missing', 'comment_missing']:
            if key in results and not results[key].empty:
                f.write(f"\n### {key}:\n")
                f.write(results[key].to_string())
                f.write("\n")
        
        f.write("\n## 品質判定\n")
        # 欠損率20%以上の項目をリストアップ
        high_missing = []
        for df_name, missing_df in [('授業前', results.get('before_missing', pd.DataFrame())),
                                     ('授業後', results.get('after_missing', pd.DataFrame()))]:
            if not missing_df.empty:
                high_missing_cols = missing_df[missing_df['欠損率(%)'] > 20].index.tolist()
                if high_missing_cols:
                    high_missing.extend([(df_name, col) for col in high_missing_cols])
        
        if high_missing:
            f.write("- 警告: 以下の項目は欠損率が20%を超えています:\n")
            for df_name, col in high_missing:
                f.write(f"  - {df_name}: {col}\n")
        else:
            f.write("- 欠損率20%を超える項目はありません\n")
        
        f.write("\n## 推奨事項\n")
        if len(page_id_results['only_before']) > 0 or len(page_id_results['only_after']) > 0:
            f.write("- Page_IDのマッチングに問題があります。前後比較分析では共通IDのみを使用することを推奨します\n")
        
        if results.get('before_logical_inconsistencies') or results.get('after_logical_inconsistencies'):
            f.write("- 論理的矛盾が検出されました。データクリーニングを検討してください\n")
    
    print(f"\n品質レポートを保存しました: {REPORT_DIR}")


def main():
    """メイン処理"""
    print("=" * 80)
    print("Phase 1: データ品質確認")
    print("=" * 80)
    
    # データ読み込み
    before_df, after_df, comment_df = load_data()
    
    # 結果を格納する辞書
    results = {}
    
    # 1. 欠損値パターンの確認
    results['before_missing'] = check_missing_values(before_df, '授業前')
    results['after_missing'] = check_missing_values(after_df, '授業後')
    results['comment_missing'] = check_missing_values(comment_df, '感想文')
    
    # 2. Page_ID一貫性チェック
    results['page_id_consistency'] = check_page_id_consistency(before_df, after_df)
    
    # 3. 論理的一貫性チェック
    results['before_logical_inconsistencies'] = check_logical_consistency(before_df, '授業前')
    results['after_logical_inconsistencies'] = check_logical_consistency(after_df, '授業後')
    
    # 4. 外れ値・異常値チェック
    results['before_outliers'] = check_outliers(before_df, '授業前')
    results['after_outliers'] = check_outliers(after_df, '授業後')
    
    # レポート生成
    generate_quality_report(before_df, after_df, comment_df, results)
    
    print("\n" + "=" * 80)
    print("Phase 1 完了: 品質チェックレポートを確認してください")
    print("=" * 80)


if __name__ == "__main__":
    main()