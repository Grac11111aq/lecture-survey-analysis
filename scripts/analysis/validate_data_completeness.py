#!/usr/bin/env python3
"""
データ完全性検証スクリプト
rawデータとexcel準拠データの比較、最適なデータソースの特定
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

def compare_data_versions():
    """異なるバージョンのデータを比較"""
    
    print("=" * 80)
    print("データ完全性検証")
    print("=" * 80)
    
    # 各データソースを読み込み
    print("\n## データファイルの読み込みと基本情報")
    
    # 1. Raw データ
    raw_before = pd.read_csv(DATA_DIR / "raw" / "before.csv")
    raw_after = pd.read_csv(DATA_DIR / "raw" / "after.csv")
    raw_comment = pd.read_csv(DATA_DIR / "raw" / "comment.csv")
    
    print(f"\n### Raw データ:")
    print(f"- before.csv: {raw_before.shape}")
    print(f"- after.csv: {raw_after.shape}")
    print(f"- comment.csv: {raw_comment.shape}")
    
    # 2. Excel準拠データ
    excel_before = pd.read_csv(DATA_DIR / "analysis" / "before_excel_compliant.csv")
    excel_after = pd.read_csv(DATA_DIR / "analysis" / "after_excel_compliant.csv")
    excel_comment = pd.read_csv(DATA_DIR / "analysis" / "comment.csv")
    
    print(f"\n### Excel準拠データ:")
    print(f"- before_excel_compliant.csv: {excel_before.shape}")
    print(f"- after_excel_compliant.csv: {excel_after.shape}")
    print(f"- comment.csv: {excel_comment.shape}")
    
    # Q1とQ3のデータ存在状況
    print("\n## Q1（水溶液認識）データの存在状況")
    
    # Raw before
    q1_cols_before = [col for col in raw_before.columns if col.startswith('Q1_')]
    print(f"\n### Raw before.csv のQ1列:")
    for col in q1_cols_before:
        non_null_count = raw_before[col].notna().sum()
        print(f"- {col}: {non_null_count}件 ({non_null_count/len(raw_before)*100:.1f}%)")
    
    # Excel before
    q1_cols_excel_before = [col for col in excel_before.columns if col.startswith('Q1_')]
    print(f"\n### Excel before_excel_compliant.csv のQ1列:")
    for col in q1_cols_excel_before:
        non_null_count = excel_before[col].notna().sum()
        print(f"- {col}: {non_null_count}件 ({non_null_count/len(excel_before)*100:.1f}%)")
    
    # Raw after
    q1_cols_after = [col for col in raw_after.columns if col.startswith('Q1_')]
    print(f"\n### Raw after.csv のQ1列:")
    for col in q1_cols_after:
        non_null_count = raw_after[col].notna().sum()
        print(f"- {col}: {non_null_count}件 ({non_null_count/len(raw_after)*100:.1f}%)")
    
    # Excel after
    q1_cols_excel_after = [col for col in excel_after.columns if col.startswith('Q1_')]
    print(f"\n### Excel after_excel_compliant.csv のQ1列:")
    for col in q1_cols_excel_after:
        non_null_count = excel_after[col].notna().sum()
        print(f"- {col}: {non_null_count}件 ({non_null_count/len(excel_after)*100:.1f}%)")
    
    print("\n## Q3（お茶理解）データの存在状況")
    
    # Raw before Q3
    q3_cols_before = [col for col in raw_before.columns if col.startswith('Q3_')]
    print(f"\n### Raw before.csv のQ3列:")
    for col in q3_cols_before:
        non_null_count = raw_before[col].notna().sum()
        print(f"- {col}: {non_null_count}件 ({non_null_count/len(raw_before)*100:.1f}%)")
    
    # Excel before Q3
    q3_cols_excel_before = [col for col in excel_before.columns if col.startswith('Q3_')]
    print(f"\n### Excel before_excel_compliant.csv のQ3列:")
    for col in q3_cols_excel_before:
        non_null_count = excel_before[col].notna().sum()
        print(f"- {col}: {non_null_count}件 ({non_null_count/len(excel_before)*100:.1f}%)")
    
    # Raw after Q3
    q3_cols_after = [col for col in raw_after.columns if col.startswith('Q3_')]
    print(f"\n### Raw after.csv のQ3列:")
    for col in q3_cols_after:
        non_null_count = raw_after[col].notna().sum()
        print(f"- {col}: {non_null_count}件 ({non_null_count/len(raw_after)*100:.1f}%)")
    
    # Excel after Q3
    q3_cols_excel_after = [col for col in excel_after.columns if col.startswith('Q3_')]
    print(f"\n### Excel after_excel_compliant.csv のQ3列:")
    for col in q3_cols_excel_after:
        non_null_count = excel_after[col].notna().sum()
        print(f"- {col}: {non_null_count}件 ({non_null_count/len(excel_after)*100:.1f}%)")
    
    # Q2とQ4-Q6の比較
    print("\n## その他の項目の比較")
    
    # Q2
    print(f"\n### Q2（みそ汁理由）:")
    raw_q2_before = raw_before['Q2_MisoSalty_Reason'].notna().sum() if 'Q2_MisoSalty_Reason' in raw_before.columns else 0
    excel_q2_before = excel_before['Q2_MisoSalty_Reason'].notna().sum() if 'Q2_MisoSalty_Reason' in excel_before.columns else 0
    raw_q2_after = raw_after['Q2_MisoSaltyReason'].notna().sum() if 'Q2_MisoSaltyReason' in raw_after.columns else 0
    excel_q2_after = excel_after['Q2_MisoSaltyReason'].notna().sum() if 'Q2_MisoSaltyReason' in excel_after.columns else 0
    
    print(f"- Raw before: {raw_q2_before}件")
    print(f"- Excel before: {excel_q2_before}件")
    print(f"- Raw after: {raw_q2_after}件")
    print(f"- Excel after: {excel_q2_after}件")
    
    # Q4-Q6（授業後のみ）
    rating_cols = ['Q4_ExperimentInterestRating', 'Q5_NewLearningsRating', 'Q6_DissolvingUnderstandingRating']
    print(f"\n### Q4-Q6評価（授業後のみ）:")
    for col in rating_cols:
        raw_count = raw_after[col].notna().sum() if col in raw_after.columns else 0
        excel_count = excel_after[col].notna().sum() if col in excel_after.columns else 0
        print(f"- {col}:")
        print(f"  Raw: {raw_count}件")
        print(f"  Excel: {excel_count}件")
    
    # 推奨データソース
    print("\n" + "=" * 80)
    print("推奨データソースの判定")
    print("=" * 80)
    
    print("\n## 結論:")
    print("1. **Q1・Q3分析**: Rawデータを使用（Excel準拠版にはデータなし）")
    print("2. **Q2分析**: どちらも使用可能（Excelの方が若干データ数多い可能性）")
    print("3. **Q4-Q6分析**: Rawデータを使用（より完全）")
    print("4. **感想文分析**: どちらも使用可能")
    
    print("\n## 推奨アクション:")
    print("- 主分析はRawデータ（data/raw/）を使用")
    print("- 必要に応じてExcel準拠データで補完")
    print("- Q1・Q3の授業前後比較が可能")
    
    # サンプルデータの確認
    print("\n## サンプルデータ確認")
    print("\n### Raw before.csv のQ1サンプル:")
    sample_raw_before = raw_before[q1_cols_before + ['Q2_MisoSalty_Reason']].head(3)
    print(sample_raw_before.to_string(index=False))
    
    print("\n### Raw after.csv のQ1サンプル:")
    sample_raw_after = raw_after[q1_cols_after + ['Q2_MisoSaltyReason']].head(3)
    print(sample_raw_after.to_string(index=False))


def main():
    """メイン処理"""
    compare_data_versions()


if __name__ == "__main__":
    main()