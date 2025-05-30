#!/usr/bin/env python3
"""
最終データ修正スクリプト
正確なIDマッピングと包括的な修正を実行
"""

import pandas as pd
import numpy as np
from datetime import datetime
import shutil
from pathlib import Path

def comprehensive_data_correction():
    """包括的なデータ修正"""
    print("=== 包括的データ修正の実行 ===\n")
    
    # バックアップの作成
    create_final_backup()
    
    # 元のExcelファイルを読み込み
    excel_file = "refference/250226アンケートデータ/250226アンケートデータ.xlsx"
    df_before_excel = pd.read_excel(excel_file, sheet_name="授業前")
    df_after_excel = pd.read_excel(excel_file, sheet_name="授業後")
    
    # 現在のCSVファイルを読み込み
    df_before_csv = pd.read_csv("before.csv")
    df_after_csv = pd.read_csv("after.csv")
    
    print(f"Excel授業前: {len(df_before_excel)}行, CSV授業前: {len(df_before_csv)}行")
    print(f"Excel授業後: {len(df_after_excel)}行, CSV授業後: {len(df_after_csv)}行")
    
    # 正確なIDマッピングの構築
    id_mapping = build_accurate_id_mapping(df_before_excel, df_before_csv)
    print(f"構築されたIDマッピング: {len(id_mapping)}組")
    
    # 授業前データの完全修正
    print("\n1. 授業前データの完全修正")
    df_before_corrected = correct_before_data_completely(df_before_excel, df_before_csv, id_mapping)
    
    # 授業後データの完全修正
    print("\n2. 授業後データの完全修正")
    df_after_corrected = correct_after_data_completely(df_after_excel, df_after_csv, id_mapping)
    
    # 修正されたデータを保存
    df_before_corrected.to_csv("before.csv", index=False, encoding='utf-8')
    df_after_corrected.to_csv("after.csv", index=False, encoding='utf-8')
    
    print("\n=== 修正完了 ===")
    print("before.csv と after.csv を完全修正しました")
    
    # 最終検証
    perform_final_verification(df_before_excel, df_after_excel, id_mapping)
    
    return df_before_corrected, df_after_corrected

def build_accurate_id_mapping(df_excel, df_csv):
    """より正確なIDマッピングの構築"""
    print("正確なIDマッピングを構築中...")
    
    mapping = {}
    
    # クラス情報を基にした詳細マッピング
    for cls in [1, 2, 3, 4]:
        # Excel側のクラス内データ
        excel_class = df_excel[df_excel['クラス'] == cls].copy()
        excel_class = excel_class.sort_values('整理番号').reset_index(drop=True)
        
        # CSV側のクラス内データ
        csv_class = df_csv[df_csv['class'] == cls].copy()
        csv_class = csv_class.sort_values('Page_ID').reset_index(drop=True)
        
        print(f"クラス{cls}: Excel {len(excel_class)}人, CSV {len(csv_class)}人")
        
        # 1対1マッピングを作成
        min_len = min(len(excel_class), len(csv_class))
        for i in range(min_len):
            excel_id = excel_class.iloc[i]['整理番号']
            page_id = csv_class.iloc[i]['Page_ID']
            mapping[excel_id] = page_id
    
    return mapping

def correct_before_data_completely(df_excel, df_csv, id_mapping):
    """授業前データの完全修正"""
    print("授業前データを修正中...")
    
    # 列マッピング
    column_mapping = {
        'クイズ1（〇が1，×が0）': 'Q1_Saltwater_Response',
        'クイズ2': 'Q1_Sugarwater_Response',
        'クイズ3': 'Q1_Muddywater_Response',
        'クイズ4': 'Q1_Ink_Response',
        'クイズ5': 'Q1_MisoSoup_Response',
        'クイズ6': 'Q1_SoySauce_Response',
        'お茶クイズ1（いる1，いない0）': 'Q3_TeaLeavesDissolve',
        'お茶クイズ2（いる1，いない0）': 'Q3_TeaComponentsDissolve'
    }
    
    df_corrected = df_csv.copy()
    corrections_count = 0
    
    for excel_id, page_id in id_mapping.items():
        # Excel行を取得
        excel_row = df_excel[df_excel['整理番号'] == excel_id]
        if len(excel_row) == 0:
            continue
        excel_row = excel_row.iloc[0]
        
        # CSV行のインデックスを取得
        csv_index = df_corrected[df_corrected['Page_ID'] == page_id].index
        if len(csv_index) == 0:
            continue
        csv_index = csv_index[0]
        
        # 各列を修正
        for excel_col, csv_col in column_mapping.items():
            excel_val = excel_row[excel_col]
            
            if pd.notna(excel_val):
                # 1/0を True/False に変換
                corrected_val = bool(excel_val == 1 or excel_val == 1.0)
                df_corrected.loc[csv_index, csv_col] = corrected_val
                corrections_count += 1
    
    print(f"授業前データ修正数: {corrections_count}")
    return df_corrected

def correct_after_data_completely(df_excel, df_csv, id_mapping):
    """授業後データの完全修正"""
    print("授業後データを修正中...")
    
    # 列マッピング
    column_mapping = {
        'クイズ1（〇が1，×が0）': 'Q1_Saltwater',
        'クイズ2': 'Q1_Sugarwater',
        'クイズ3': 'Q1_Muddywater',
        'クイズ4': 'Q1_Ink',
        'クイズ5': 'Q1_MisoSoup',
        'クイズ6': 'Q1_SoySauce',
        'お茶クイズ1（いる1，いない0）': 'Q3_TeaLeaves_DissolveInWater',
        'お茶クイズ2（いる1，いない0）': 'Q3_TeaComponents_DissolveInWater',
        'おもしろさ': 'Q4_ExperimentInterestRating',
        '新発見': 'Q5_NewLearningsRating',
        '理解': 'Q6_DissolvingUnderstandingRating'
    }
    
    df_corrected = df_csv.copy()
    corrections_count = 0
    
    for excel_id, page_id in id_mapping.items():
        # Excel行を取得
        excel_row = df_excel[df_excel['整理番号'] == excel_id]
        if len(excel_row) == 0:
            continue
        excel_row = excel_row.iloc[0]
        
        # CSV行のインデックスを取得
        csv_index = df_corrected[df_corrected['Page_ID'] == page_id].index
        if len(csv_index) == 0:
            continue
        csv_index = csv_index[0]
        
        # 各列を修正
        for excel_col, csv_col in column_mapping.items():
            excel_val = excel_row[excel_col]
            
            if pd.notna(excel_val):
                if excel_col.startswith('クイズ') or excel_col.startswith('お茶'):
                    # ブール値列: 1/0 を True/False に変換
                    corrected_val = bool(excel_val == 1 or excel_val == 1.0)
                else:
                    # 評価列: 数値をそのまま使用
                    corrected_val = int(excel_val)
                
                df_corrected.loc[csv_index, csv_col] = corrected_val
                corrections_count += 1
    
    print(f"授業後データ修正数: {corrections_count}")
    return df_corrected

def perform_final_verification(df_excel_before, df_excel_after, id_mapping):
    """最終検証"""
    print("\n=== 最終検証 ===")
    
    # 修正後のCSVを読み込み
    df_csv_before = pd.read_csv("before.csv")
    df_csv_after = pd.read_csv("after.csv")
    
    # 授業前データの検証
    before_mismatches = verify_final_before(df_excel_before, df_csv_before, id_mapping)
    
    # 授業後データの検証  
    after_mismatches = verify_final_after(df_excel_after, df_csv_after, id_mapping)
    
    total_mismatches = before_mismatches + after_mismatches
    
    if total_mismatches == 0:
        print("✅ 完全一致確認！")
        print("修正後のCSVデータは手動入力Excelデータと100%一致しています。")
    else:
        print(f"⚠️ まだ{total_mismatches}件の不一致があります。")

def verify_final_before(df_excel, df_csv, id_mapping):
    """授業前データの最終検証"""
    column_mapping = {
        'クイズ1（〇が1，×が0）': 'Q1_Saltwater_Response',
        'クイズ2': 'Q1_Sugarwater_Response',
        'クイズ3': 'Q1_Muddywater_Response',
        'クイズ4': 'Q1_Ink_Response',
        'クイズ5': 'Q1_MisoSoup_Response',
        'クイズ6': 'Q1_SoySauce_Response',
        'お茶クイズ1（いる1，いない0）': 'Q3_TeaLeavesDissolve',
        'お茶クイズ2（いる1，いない0）': 'Q3_TeaComponentsDissolve'
    }
    
    mismatches = 0
    
    for excel_id, page_id in id_mapping.items():
        excel_row = df_excel[df_excel['整理番号'] == excel_id]
        csv_row = df_csv[df_csv['Page_ID'] == page_id]
        
        if len(excel_row) == 0 or len(csv_row) == 0:
            continue
            
        excel_row = excel_row.iloc[0]
        csv_row = csv_row.iloc[0]
        
        for excel_col, csv_col in column_mapping.items():
            excel_val = excel_row[excel_col]
            csv_val = csv_row[csv_col]
            
            if pd.notna(excel_val):
                expected = bool(excel_val == 1 or excel_val == 1.0)
                if expected != csv_val:
                    mismatches += 1
                    if mismatches <= 3:  # 最初の3件のみ表示
                        print(f"  不一致: ID{excel_id}→{page_id}, {excel_col}, Excel:{excel_val}→{expected}, CSV:{csv_val}")
    
    print(f"授業前データ不一致: {mismatches}件")
    return mismatches

def verify_final_after(df_excel, df_csv, id_mapping):
    """授業後データの最終検証"""
    column_mapping = {
        'クイズ1（〇が1，×が0）': 'Q1_Saltwater',
        'クイズ2': 'Q1_Sugarwater',
        'クイズ3': 'Q1_Muddywater',
        'クイズ4': 'Q1_Ink',
        'クイズ5': 'Q1_MisoSoup',
        'クイズ6': 'Q1_SoySauce',
        'お茶クイズ1（いる1，いない0）': 'Q3_TeaLeaves_DissolveInWater',
        'お茶クイズ2（いる1，いない0）': 'Q3_TeaComponents_DissolveInWater',
        'おもしろさ': 'Q4_ExperimentInterestRating',
        '新発見': 'Q5_NewLearningsRating',
        '理解': 'Q6_DissolvingUnderstandingRating'
    }
    
    mismatches = 0
    
    for excel_id, page_id in id_mapping.items():
        excel_row = df_excel[df_excel['整理番号'] == excel_id]
        csv_row = df_csv[df_csv['Page_ID'] == page_id]
        
        if len(excel_row) == 0 or len(csv_row) == 0:
            continue
            
        excel_row = excel_row.iloc[0]
        csv_row = csv_row.iloc[0]
        
        for excel_col, csv_col in column_mapping.items():
            excel_val = excel_row[excel_col]
            csv_val = csv_row[csv_col]
            
            if pd.notna(excel_val):
                if excel_col.startswith('クイズ') or excel_col.startswith('お茶'):
                    expected = bool(excel_val == 1 or excel_val == 1.0)
                else:
                    expected = int(excel_val)
                
                if expected != csv_val:
                    mismatches += 1
                    if mismatches <= 3:  # 最初の3件のみ表示
                        print(f"  不一致: ID{excel_id}→{page_id}, {excel_col}, Excel:{excel_val}→{expected}, CSV:{csv_val}")
    
    print(f"授業後データ不一致: {mismatches}件")
    return mismatches

def create_final_backup():
    """最終バックアップの作成"""
    backup_dir = Path("backup")
    backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_final")
    
    files_to_backup = ["before.csv", "after.csv"]
    for file in files_to_backup:
        if Path(file).exists():
            backup_file = backup_dir / f"{file}_{timestamp}"
            shutil.copy2(file, backup_file)
            print(f"最終バックアップ作成: {backup_file}")

if __name__ == "__main__":
    comprehensive_data_correction()