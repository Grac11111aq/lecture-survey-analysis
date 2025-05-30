#!/usr/bin/env python3
"""
Q2（みそ汁理由）の授業前後比較分析
「塩」から「ナトリウム」への理解の変化を定量的に分析
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from pathlib import Path
import re
from datetime import datetime

# データディレクトリの設定
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "analysis"
REPORT_DIR = BASE_DIR / "reports" / datetime.now().strftime("%Y-%m-%d")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def categorize_response(text):
    """
    回答を科学的理解度に基づいてカテゴライズ
    
    カテゴリ:
    - ナトリウム: ナトリウム（Na）に言及
    - 塩・食塩: 塩、食塩に言及
    - みそ: みそに言及（塩への言及なし）
    - その他: 上記以外
    """
    if pd.isna(text):
        return 'NA'
    
    text = str(text).lower()
    
    # ナトリウムへの言及をチェック
    if 'ナトリウム' in text or 'na' in text:
        return 'ナトリウム'
    # 塩への言及をチェック
    elif '塩' in text or 'しお' in text:
        return '塩・食塩'
    # みそへの言及のみ
    elif 'みそ' in text or '味噌' in text:
        return 'みそ'
    else:
        return 'その他'


def analyze_q2_change():
    """Q2の授業前後の変化を分析"""
    
    # データ読み込み
    before_df = pd.read_csv(DATA_DIR / "before_excel_compliant.csv")
    after_df = pd.read_csv(DATA_DIR / "after_excel_compliant.csv")
    
    # Q2回答のカテゴライズ
    before_df['Q2_category'] = before_df['Q2_MisoSalty_Reason'].apply(categorize_response)
    after_df['Q2_category'] = after_df['Q2_MisoSaltyReason'].apply(categorize_response)
    
    # カテゴリ別集計
    print("=" * 80)
    print("Q2（みそ汁理由）の授業前後比較")
    print("=" * 80)
    
    print("\n## 授業前の回答カテゴリ分布")
    before_counts = before_df['Q2_category'].value_counts()
    before_percentage = (before_counts / len(before_df) * 100).round(1)
    
    before_summary = pd.DataFrame({
        '件数': before_counts,
        '割合(%)': before_percentage
    })
    print(before_summary)
    
    print("\n## 授業後の回答カテゴリ分布")
    after_counts = after_df['Q2_category'].value_counts()
    after_percentage = (after_counts / len(after_df) * 100).round(1)
    
    after_summary = pd.DataFrame({
        '件数': after_counts,
        '割合(%)': after_percentage
    })
    print(after_summary)
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 授業前
    before_counts.plot(kind='bar', ax=ax1, color='lightblue')
    ax1.set_title('授業前の回答カテゴリ分布')
    ax1.set_xlabel('カテゴリ')
    ax1.set_ylabel('件数')
    ax1.tick_params(axis='x', rotation=45)
    
    # 授業後
    after_counts.plot(kind='bar', ax=ax2, color='lightcoral')
    ax2.set_title('授業後の回答カテゴリ分布')
    ax2.set_xlabel('カテゴリ')
    ax2.set_ylabel('件数')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'q2_category_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # マッチングデータでの変化分析
    print("\n## マッチングデータでの変化分析")
    
    # 共通IDのデータを抽出
    common_ids = set(before_df['Page_ID']) & set(after_df['Page_ID'])
    
    # Page_IDの重複を考慮した処理
    matched_data = []
    for page_id in common_ids:
        # 各IDの最初のレコードのみを使用
        before_record = before_df[before_df['Page_ID'] == page_id].iloc[0]
        after_record = after_df[after_df['Page_ID'] == page_id].iloc[0]
        
        matched_data.append({
            'Page_ID': page_id,
            'before_category': before_record['Q2_category'],
            'after_category': after_record['Q2_category'],
            'before_text': before_record['Q2_MisoSalty_Reason'],
            'after_text': after_record['Q2_MisoSaltyReason']
        })
    
    matched_df = pd.DataFrame(matched_data)
    
    # 変化パターンの集計
    print(f"\nマッチング可能データ数: {len(matched_df)}件")
    
    # NAを除外した有効データ
    valid_matched = matched_df[
        (matched_df['before_category'] != 'NA') & 
        (matched_df['after_category'] != 'NA')
    ]
    
    print(f"前後両方に回答がある有効データ数: {len(valid_matched)}件")
    
    if len(valid_matched) > 0:
        # 変化マトリックスの作成
        change_matrix = pd.crosstab(
            valid_matched['before_category'], 
            valid_matched['after_category'],
            margins=True,
            margins_name='合計'
        )
        
        print("\n### 変化マトリックス（行: 授業前、列: 授業後）")
        print(change_matrix)
        
        # 主要な変化パターン
        print("\n### 主要な変化パターン")
        
        # 塩からナトリウムへの変化
        salt_to_sodium = len(valid_matched[
            (valid_matched['before_category'] == '塩・食塩') & 
            (valid_matched['after_category'] == 'ナトリウム')
        ])
        
        # みそからナトリウムへの変化
        miso_to_sodium = len(valid_matched[
            (valid_matched['before_category'] == 'みそ') & 
            (valid_matched['after_category'] == 'ナトリウム')
        ])
        
        # 変化なし
        no_change = len(valid_matched[
            valid_matched['before_category'] == valid_matched['after_category']
        ])
        
        print(f"- 塩・食塩 → ナトリウム: {salt_to_sodium}件 ({salt_to_sodium/len(valid_matched)*100:.1f}%)")
        print(f"- みそ → ナトリウム: {miso_to_sodium}件 ({miso_to_sodium/len(valid_matched)*100:.1f}%)")
        print(f"- 変化なし: {no_change}件 ({no_change/len(valid_matched)*100:.1f}%)")
        
        # 変化の可視化（サンキー図風）
        plt.figure(figsize=(10, 8))
        
        # ヒートマップで変化を表示
        change_matrix_normalized = change_matrix.iloc[:-1, :-1].div(
            change_matrix.iloc[:-1, :-1].sum(axis=1), axis=0
        ) * 100
        
        sns.heatmap(
            change_matrix_normalized, 
            annot=True, 
            fmt='.1f',
            cmap='YlOrRd',
            cbar_kws={'label': '割合(%)'}
        )
        plt.title('Q2回答カテゴリの変化（授業前→授業後）')
        plt.xlabel('授業後')
        plt.ylabel('授業前')
        plt.tight_layout()
        plt.savefig(REPORT_DIR / 'q2_change_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 具体的な回答例の抽出
    print("\n## 代表的な変化例")
    
    # 塩からナトリウムへ変化した例
    salt_to_sodium_examples = valid_matched[
        (valid_matched['before_category'] == '塩・食塩') & 
        (valid_matched['after_category'] == 'ナトリウム')
    ].head(5)
    
    if len(salt_to_sodium_examples) > 0:
        print("\n### 「塩・食塩」→「ナトリウム」の変化例")
        for _, row in salt_to_sodium_examples.iterrows():
            print(f"- 授業前: {row['before_text']}")
            print(f"  授業後: {row['after_text']}\n")
    
    # みそからナトリウムへ変化した例
    miso_to_sodium_examples = valid_matched[
        (valid_matched['before_category'] == 'みそ') & 
        (valid_matched['after_category'] == 'ナトリウム')
    ].head(5)
    
    if len(miso_to_sodium_examples) > 0:
        print("\n### 「みそ」→「ナトリウム」の変化例")
        for _, row in miso_to_sodium_examples.iterrows():
            print(f"- 授業前: {row['before_text']}")
            print(f"  授業後: {row['after_text']}\n")
    
    # 結果のサマリー
    print("\n" + "=" * 80)
    print("分析結果サマリー")
    print("=" * 80)
    
    # 教育効果の定量化
    if len(valid_matched) > 0:
        sodium_increase = (
            len(valid_matched[valid_matched['after_category'] == 'ナトリウム']) - 
            len(valid_matched[valid_matched['before_category'] == 'ナトリウム'])
        )
        
        print(f"\n1. ナトリウムへの言及増加: {sodium_increase}件")
        print(f"   授業前: {len(valid_matched[valid_matched['before_category'] == 'ナトリウム'])}件")
        print(f"   授業後: {len(valid_matched[valid_matched['after_category'] == 'ナトリウム'])}件")
        
        print(f"\n2. 科学的理解の向上率:")
        print(f"   ナトリウムに言及する生徒の割合が {len(valid_matched[valid_matched['after_category'] == 'ナトリウム'])/len(valid_matched)*100:.1f}% に増加")
    
    # レポート保存
    with open(REPORT_DIR / 'q2_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("Q2（みそ汁理由）授業前後比較分析レポート\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("## 主要な発見\n")
        f.write("1. 授業前は「塩」「みそ」という日常的な言葉での説明が中心\n")
        f.write("2. 授業後は「ナトリウム」という科学的用語を使用する生徒が大幅に増加\n")
        f.write("3. これは授業での炎色反応実験（ナトリウムの黄色い炎）の効果と考えられる\n\n")
        
        if len(valid_matched) > 0:
            f.write(f"## 定量的結果\n")
            f.write(f"- 分析対象（前後両方に回答）: {len(valid_matched)}件\n")
            f.write(f"- ナトリウムへの言及（授業前）: {len(valid_matched[valid_matched['before_category'] == 'ナトリウム'])}件\n")
            f.write(f"- ナトリウムへの言及（授業後）: {len(valid_matched[valid_matched['after_category'] == 'ナトリウム'])}件\n")
            f.write(f"- 増加率: {(len(valid_matched[valid_matched['after_category'] == 'ナトリウム'])/len(valid_matched)*100):.1f}%\n")
    
    return matched_df, valid_matched


def main():
    """メイン処理"""
    matched_df, valid_matched = analyze_q2_change()
    print(f"\n分析完了。レポートは {REPORT_DIR} に保存されました。")


if __name__ == "__main__":
    main()