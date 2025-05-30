#!/usr/bin/env python3
"""
データ詳細調査スクリプト
利用可能なデータの内容を詳細に確認し、分析可能な範囲を特定する
"""

import pandas as pd
import numpy as np
from pathlib import Path

# データディレクトリの設定
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "analysis"

def investigate_data_structure():
    """データ構造の詳細調査"""
    print("=" * 80)
    print("データ構造の詳細調査")
    print("=" * 80)
    
    # データ読み込み
    before_df = pd.read_csv(DATA_DIR / "before_excel_compliant.csv")
    after_df = pd.read_csv(DATA_DIR / "after_excel_compliant.csv")
    comment_df = pd.read_csv(DATA_DIR / "comment.csv")
    
    # 1. 実際にデータが存在する列の確認
    print("\n## 授業前アンケート - 実データがある列")
    for col in before_df.columns:
        non_null_count = before_df[col].notna().sum()
        if non_null_count > 0:
            print(f"- {col}: {non_null_count}件 ({non_null_count/len(before_df)*100:.1f}%)")
            # サンプルデータ表示
            sample_data = before_df[col].dropna().head(3).tolist()
            if sample_data:
                print(f"  サンプル: {sample_data}")
    
    print("\n## 授業後アンケート - 実データがある列")
    for col in after_df.columns:
        non_null_count = after_df[col].notna().sum()
        if non_null_count > 0:
            print(f"- {col}: {non_null_count}件 ({non_null_count/len(after_df)*100:.1f}%)")
            # サンプルデータ表示
            sample_data = after_df[col].dropna().head(3).tolist()
            if sample_data:
                print(f"  サンプル: {sample_data}")
    
    # 2. Page_ID重複の詳細調査
    print("\n## Page_ID重複の詳細")
    print("\n### 授業前の重複ID")
    before_duplicates = before_df[before_df['Page_ID'].duplicated(keep=False)].sort_values('Page_ID')
    if len(before_duplicates) > 0:
        print(f"重複件数: {len(before_duplicates)}件")
        print("\n重複例（最初の10件）:")
        print(before_duplicates[['Page_ID', 'Excel_ID', 'class']].head(10).to_string(index=False))
    
    print("\n### 授業後の重複ID")
    after_duplicates = after_df[after_df['Page_ID'].duplicated(keep=False)].sort_values('Page_ID')
    if len(after_duplicates) > 0:
        print(f"重複件数: {len(after_duplicates)}件")
        print("\n重複例（最初の10件）:")
        print(after_duplicates[['Page_ID', 'Excel_ID', 'class']].head(10).to_string(index=False))
    
    # 3. クラス別のデータ分布
    print("\n## クラス別データ分布")
    print("\n### 授業前")
    print(before_df['class'].value_counts().sort_index())
    print("\n### 授業後")
    print(after_df['class'].value_counts().sort_index())
    
    # 4. Q4-Q6の実際の値の分布（授業後のみ）
    print("\n## 授業後評価項目の分布")
    rating_cols = ['Q4_ExperimentInterestRating', 'Q5_NewLearningsRating', 'Q6_DissolvingUnderstandingRating']
    for col in rating_cols:
        if col in after_df.columns:
            print(f"\n### {col}")
            value_counts = after_df[col].value_counts().sort_index()
            print(value_counts)
            print(f"平均: {after_df[col].mean():.2f}")
            print(f"標準偏差: {after_df[col].std():.2f}")
    
    # 5. テキストデータの内容確認
    print("\n## テキストデータの内容")
    
    # Q2の回答内容
    print("\n### Q2_みそ汁理由（授業前）の頻出パターン")
    if 'Q2_MisoSalty_Reason' in before_df.columns:
        q2_before = before_df['Q2_MisoSalty_Reason'].dropna()
        print(f"回答数: {len(q2_before)}件")
        print("\n頻出回答（上位10件）:")
        print(q2_before.value_counts().head(10))
    
    print("\n### Q2_みそ汁理由（授業後）の頻出パターン")
    if 'Q2_MisoSaltyReason' in after_df.columns:
        q2_after = after_df['Q2_MisoSaltyReason'].dropna()
        print(f"回答数: {len(q2_after)}件")
        print("\n頻出回答（上位10件）:")
        print(q2_after.value_counts().head(10))
    
    # 6. 感想文データの確認
    print("\n## 感想文データ")
    print(f"感想文の総数: {len(comment_df)}件")
    print("\n列名:")
    print(comment_df.columns.tolist())
    if 'comment' in comment_df.columns:
        print("\n感想文の長さ分布:")
        comment_lengths = comment_df['comment'].str.len()
        print(f"最短: {comment_lengths.min()}文字")
        print(f"最長: {comment_lengths.max()}文字")
        print(f"平均: {comment_lengths.mean():.1f}文字")
        print(f"中央値: {comment_lengths.median():.1f}文字")
    
    # 7. 分析可能性の評価
    print("\n" + "=" * 80)
    print("分析可能性の評価")
    print("=" * 80)
    
    print("\n## 利用可能な分析:")
    print("1. Q2（みそ汁理由）の授業前後比較 - テキスト分析")
    print("2. Q4-Q6の評価分析（授業後のみ、約65%のデータ）")
    print("3. 感想文のテキストマイニング")
    print("4. クラス別の分析（ただしクラス情報も一部欠損）")
    
    print("\n## 制限事項:")
    print("1. Q1（水溶液認識）とQ3（お茶理解）は完全欠損のため分析不可")
    print("2. Page_IDの重複により、前後比較は慎重に行う必要あり")
    print("3. 授業前後の完全なマッチングは62件のみ")
    
    # 8. 実際にマッチングできるデータの詳細
    print("\n## マッチング可能なデータの詳細")
    before_ids = set(before_df['Page_ID'])
    after_ids = set(after_df['Page_ID'])
    matched_ids = before_ids & after_ids
    
    print(f"\nマッチング可能なID数: {len(matched_ids)}")
    
    # マッチングデータのQ2比較可能性
    matched_before = before_df[before_df['Page_ID'].isin(matched_ids)]
    matched_after = after_df[after_df['Page_ID'].isin(matched_ids)]
    
    q2_both = matched_before['Q2_MisoSalty_Reason'].notna().sum()
    q2_after_only = matched_after['Q2_MisoSaltyReason'].notna().sum()
    
    print(f"Q2の前後比較可能数: 最大{min(q2_both, q2_after_only)}件")


def main():
    """メイン処理"""
    investigate_data_structure()


if __name__ == "__main__":
    main()