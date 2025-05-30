#!/usr/bin/env python3
"""
OCRデータ修正スクリプト
手動入力されたExcelデータに基づいてOCRデータの誤りを修正
"""

import pandas as pd
import numpy as np
from datetime import datetime
import shutil
from pathlib import Path

def apply_corrections():
    """検出された差分に基づいてCSVファイルを修正"""
    print("=== OCRデータ修正の実行 ===")
    
    # バックアップの作成
    create_backup_files()
    
    # 差分レポートの読み込み
    df_differences = pd.read_csv("detailed_ocr_validation_report.csv")
    print(f"修正対象の差分数: {len(df_differences)}")
    
    # エラータイプ別集計表示
    error_summary = df_differences['Error_Type'].value_counts()
    print("\nエラータイプ別集計:")
    for error_type, count in error_summary.items():
        print(f"  {error_type}: {count}件")
    
    corrections_applied = 0
    
    # Before データの修正
    before_diffs = df_differences[df_differences['Dataset'] == 'Before']
    if len(before_diffs) > 0:
        print(f"\n修正中: before.csv ({len(before_diffs)}件)")
        df_before = pd.read_csv("before.csv")
        
        for _, row in before_diffs.iterrows():
            page_id = row['Page_ID']
            ocr_col = row['OCR_Column'] 
            excel_converted = row['Excel_Converted']
            
            # DataFrameの該当行を修正
            mask = df_before['Page_ID'] == page_id
            if mask.any():
                df_before.loc[mask, ocr_col] = excel_converted
                corrections_applied += 1
                
                if corrections_applied <= 5:  # 最初の5件の詳細を表示
                    print(f"  修正: Page_ID={page_id}, {ocr_col}: {row['OCR_Value']} → {excel_converted}")
        
        df_before.to_csv("before.csv", index=False, encoding='utf-8')
        print(f"  before.csv保存完了")
    
    # After データの修正
    after_diffs = df_differences[df_differences['Dataset'] == 'After']
    if len(after_diffs) > 0:
        print(f"\n修正中: after.csv ({len(after_diffs)}件)")
        df_after = pd.read_csv("after.csv")
        
        for _, row in after_diffs.iterrows():
            page_id = row['Page_ID']
            ocr_col = row['OCR_Column']
            excel_converted = row['Excel_Converted']
            
            # DataFrameの該当行を修正
            mask = df_after['Page_ID'] == page_id
            if mask.any():
                df_after.loc[mask, ocr_col] = excel_converted
                corrections_applied += 1
                
                if corrections_applied <= 10:  # 最初の10件の詳細を表示
                    print(f"  修正: Page_ID={page_id}, {ocr_col}: {row['OCR_Value']} → {excel_converted}")
        
        df_after.to_csv("after.csv", index=False, encoding='utf-8')
        print(f"  after.csv保存完了")
    
    print(f"\n=== 修正完了 ===")
    print(f"修正適用完了: {corrections_applied}件")
    print("修正されたファイルは元のファイルを上書きしました")
    print("バックアップファイルは backup/ ディレクトリに保存されています")
    
    return corrections_applied

def create_backup_files():
    """バックアップファイルの作成"""
    backup_dir = Path("backup")
    backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    files_to_backup = ["before.csv", "after.csv", "comment.csv"]
    for file in files_to_backup:
        if Path(file).exists():
            backup_file = backup_dir / f"{file}_{timestamp}"
            shutil.copy2(file, backup_file)
            print(f"バックアップ作成: {backup_file}")

def verify_corrections():
    """修正の検証"""
    print("\n=== 修正の検証 ===")
    
    # 修正後のファイルを読み込み
    df_before = pd.read_csv("before.csv")
    df_after = pd.read_csv("after.csv")
    
    print(f"before.csv: {len(df_before)} rows")
    print(f"after.csv: {len(df_after)} rows")
    
    # ブール値カラムの分布確認
    print("\n修正後のブール値分布（before.csv）:")
    bool_columns_before = [
        'Q1_Saltwater_Response', 'Q1_Sugarwater_Response', 'Q1_Muddywater_Response',
        'Q1_Ink_Response', 'Q1_MisoSoup_Response', 'Q1_SoySauce_Response',
        'Q3_TeaLeavesDissolve', 'Q3_TeaComponentsDissolve'
    ]
    
    for col in bool_columns_before:
        if col in df_before.columns:
            dist = df_before[col].value_counts()
            print(f"  {col}: {dist.to_dict()}")
    
    print("\n修正後のブール値分布（after.csv）:")
    bool_columns_after = [
        'Q1_Saltwater', 'Q1_Sugarwater', 'Q1_Muddywater',
        'Q1_Ink', 'Q1_MisoSoup', 'Q1_SoySauce',
        'Q3_TeaLeaves_DissolveInWater', 'Q3_TeaComponents_DissolveInWater'
    ]
    
    for col in bool_columns_after:
        if col in df_after.columns:
            dist = df_after[col].value_counts()
            print(f"  {col}: {dist.to_dict()}")

def create_summary_report():
    """修正サマリーレポートの作成"""
    print("\n=== 修正サマリーレポートの作成 ===")
    
    df_differences = pd.read_csv("detailed_ocr_validation_report.csv")
    
    # データセット別の集計
    dataset_summary = df_differences.groupby(['Dataset', 'Error_Type']).size().reset_index(name='Count')
    print("\nデータセット別・エラータイプ別集計:")
    print(dataset_summary)
    
    # カラム別の集計
    column_summary = df_differences.groupby(['Dataset', 'OCR_Column']).size().reset_index(name='Count')
    column_summary = column_summary.sort_values('Count', ascending=False)
    print(f"\nカラム別エラー数（上位10件）:")
    print(column_summary.head(10))
    
    # Page_ID別の集計
    page_summary = df_differences.groupby('Page_ID').size().reset_index(name='Error_Count')
    page_summary = page_summary.sort_values('Error_Count', ascending=False)
    print(f"\nPage_ID別エラー数（上位10件）:")
    print(page_summary.head(10))
    
    # サマリーレポートをファイルに保存
    with open("correction_summary_report.txt", "w", encoding="utf-8") as f:
        f.write("OCRデータ修正サマリーレポート\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"修正日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
        f.write(f"総修正件数: {len(df_differences)}\n\n")
        
        f.write("エラータイプ別集計:\n")
        error_summary = df_differences['Error_Type'].value_counts()
        for error_type, count in error_summary.items():
            f.write(f"  {error_type}: {count}件\n")
        
        f.write("\nデータセット別集計:\n")
        dataset_summary_simple = df_differences['Dataset'].value_counts()
        for dataset, count in dataset_summary_simple.items():
            f.write(f"  {dataset}: {count}件\n")
        
        f.write("\n主なOCRエラーパターン:\n")
        f.write("  - Boolean値の誤認識（True/False、true/false、○×の混在）\n")
        f.write("  - 評価値の誤認識（1-4の数値）\n")
        f.write("  - 特殊文字列の誤認識（\"いる/いない\"等）\n")
    
    print("修正サマリーレポートを保存しました: correction_summary_report.txt")

if __name__ == "__main__":
    try:
        # 修正の実行
        corrections_count = apply_corrections()
        
        # 修正の検証
        verify_corrections()
        
        # サマリーレポートの作成
        create_summary_report()
        
        print(f"\n=== 全作業完了 ===")
        print(f"修正されたOCRエラー: {corrections_count}件")
        print("次のステップ:")
        print("1. backup/ディレクトリ内でバックアップファイルを確認")
        print("2. 修正されたbefore.csv, after.csvを検証")
        print("3. correction_summary_report.txtで詳細レポートを確認")
        print("4. 必要に応じて分析用ノートブックで結果を検証")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("バックアップファイルから復元してください")