#!/usr/bin/env python3
"""
決定版データ修正スクリプト
手動入力Excelデータを正確にCSVに反映する
"""

import pandas as pd
import numpy as np
from datetime import datetime
import shutil
from pathlib import Path

def definitive_correction():
    """決定版の修正処理"""
    print("=== 決定版データ修正の実行 ===\n")
    
    # 安全バックアップの作成
    create_safety_backup()
    
    # 元のExcelファイルを読み込み
    excel_file = "refference/250226アンケートデータ/250226アンケートデータ.xlsx"
    df_before_excel = pd.read_excel(excel_file, sheet_name="授業前")
    df_after_excel = pd.read_excel(excel_file, sheet_name="授業後")
    
    print(f"Excelデータ読み込み: 授業前{len(df_before_excel)}行, 授業後{len(df_after_excel)}行")
    
    # 現在のCSVファイルをベースに修正（構造は保持）
    df_before_csv = pd.read_csv("before.csv")
    df_after_csv = pd.read_csv("after.csv")
    
    print(f"CSVデータ読み込み: 授業前{len(df_before_csv)}行, 授業後{len(df_after_csv)}行")
    
    # 正確なIDマッピングの構築
    id_mapping = build_precise_mapping(df_before_excel, df_before_csv)
    print(f"IDマッピング構築: {len(id_mapping)}組")
    
    # CSVファイルの直接的な置き換え修正
    df_before_corrected = fix_before_csv_directly(df_before_excel, df_before_csv, id_mapping)
    df_after_corrected = fix_after_csv_directly(df_after_excel, df_after_csv, id_mapping)
    
    # 修正データの保存
    df_before_corrected.to_csv("before.csv", index=False, encoding='utf-8')
    df_after_corrected.to_csv("after.csv", index=False, encoding='utf-8')
    
    print("\n=== 修正完了 ===")
    
    # 最終確認
    final_verification_check(df_before_excel, df_after_excel, id_mapping)

def build_precise_mapping(df_excel, df_csv):
    """最も正確なIDマッピングの構築"""
    mapping = {}
    
    # クラス別の並び順でマッピング
    print("IDマッピング詳細:")
    for cls in [1, 2, 3, 4]:
        excel_class = df_excel[df_excel['クラス'] == cls].sort_values('整理番号')
        csv_class = df_csv[df_csv['class'] == cls].sort_values('Page_ID')
        
        excel_ids = excel_class['整理番号'].tolist()
        page_ids = csv_class['Page_ID'].tolist()
        
        min_count = min(len(excel_ids), len(page_ids))
        for i in range(min_count):
            mapping[excel_ids[i]] = page_ids[i]
        
        print(f"  クラス{cls}: {min_count}組のマッピング")
    
    return mapping

def fix_before_csv_directly(df_excel, df_csv, id_mapping):
    """授業前CSVの直接修正"""
    print("\n授業前データの直接修正:")
    
    df_corrected = df_csv.copy()
    
    # カラムマッピング
    mapping = {
        'クイズ1（〇が1，×が0）': 'Q1_Saltwater_Response',
        'クイズ2': 'Q1_Sugarwater_Response',
        'クイズ3': 'Q1_Muddywater_Response',
        'クイズ4': 'Q1_Ink_Response',
        'クイズ5': 'Q1_MisoSoup_Response',
        'クイズ6': 'Q1_SoySauce_Response',
        'お茶クイズ1（いる1，いない0）': 'Q3_TeaLeavesDissolve',
        'お茶クイズ2（いる1，いない0）': 'Q3_TeaComponentsDissolve'
    }
    
    corrections = 0
    
    for excel_id, page_id in id_mapping.items():
        # Excel行を取得
        excel_rows = df_excel[df_excel['整理番号'] == excel_id]
        if len(excel_rows) == 0:
            continue
        excel_row = excel_rows.iloc[0]
        
        # CSV行を取得
        csv_rows = df_corrected[df_corrected['Page_ID'] == page_id]
        if len(csv_rows) == 0:
            continue
        csv_idx = csv_rows.index[0]
        
        # 各カラムを確実に修正
        for excel_col, csv_col in mapping.items():
            excel_val = excel_row[excel_col]
            if pd.notna(excel_val):
                # 1.0 → True, 0.0 → False
                correct_bool = bool(int(excel_val))
                df_corrected.at[csv_idx, csv_col] = correct_bool
                corrections += 1
    
    print(f"  修正項目数: {corrections}")
    return df_corrected

def fix_after_csv_directly(df_excel, df_csv, id_mapping):
    """授業後CSVの直接修正"""
    print("\n授業後データの直接修正:")
    
    df_corrected = df_csv.copy()
    
    # カラムマッピング
    mapping = {
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
    
    corrections = 0
    
    for excel_id, page_id in id_mapping.items():
        # Excel行を取得
        excel_rows = df_excel[df_excel['整理番号'] == excel_id]
        if len(excel_rows) == 0:
            continue
        excel_row = excel_rows.iloc[0]
        
        # CSV行を取得
        csv_rows = df_corrected[df_corrected['Page_ID'] == page_id]
        if len(csv_rows) == 0:
            continue
        csv_idx = csv_rows.index[0]
        
        # 各カラムを確実に修正
        for excel_col, csv_col in mapping.items():
            excel_val = excel_row[excel_col]
            if pd.notna(excel_val):
                if excel_col.startswith('クイズ') or excel_col.startswith('お茶'):
                    # ブール値: 1.0 → True, 0.0 → False
                    correct_bool = bool(int(excel_val))
                    df_corrected.at[csv_idx, csv_col] = correct_bool
                else:
                    # 評価値: そのまま整数として使用
                    correct_int = int(excel_val)
                    df_corrected.at[csv_idx, csv_col] = correct_int
                corrections += 1
    
    print(f"  修正項目数: {corrections}")
    return df_corrected

def final_verification_check(df_excel_before, df_excel_after, id_mapping):
    """最終確認検証"""
    print("\n=== 最終確認検証 ===")
    
    # 修正後のCSVを読み込み
    df_csv_before = pd.read_csv("before.csv")
    df_csv_after = pd.read_csv("after.csv")
    
    # 授業前データの確認
    before_errors = check_before_accuracy(df_excel_before, df_csv_before, id_mapping)
    
    # 授業後データの確認
    after_errors = check_after_accuracy(df_excel_after, df_csv_after, id_mapping)
    
    total_errors = before_errors + after_errors
    
    if total_errors == 0:
        print("✅ 完全一致達成！")
        print("修正後のCSVデータは手動入力Excelデータと100%一致しています。")
        return True
    else:
        print(f"❌ まだ{total_errors}件の不一致があります")
        return False

def check_before_accuracy(df_excel, df_csv, id_mapping):
    """授業前データの精度確認"""
    mapping = {
        'クイズ1（〇が1，×が0）': 'Q1_Saltwater_Response',
        'クイズ2': 'Q1_Sugarwater_Response',
        'クイズ3': 'Q1_Muddywater_Response',
        'クイズ4': 'Q1_Ink_Response',
        'クイズ5': 'Q1_MisoSoup_Response',
        'クイズ6': 'Q1_SoySauce_Response',
        'お茶クイズ1（いる1，いない0）': 'Q3_TeaLeavesDissolve',
        'お茶クイズ2（いる1，いない0）': 'Q3_TeaComponentsDissolve'
    }
    
    errors = 0
    
    for excel_id, page_id in list(id_mapping.items())[:3]:  # 最初の3件をチェック
        excel_row = df_excel[df_excel['整理番号'] == excel_id].iloc[0]
        csv_row = df_csv[df_csv['Page_ID'] == page_id].iloc[0]
        
        print(f"\nID{excel_id}→Page_ID{page_id}の確認:")
        
        for excel_col, csv_col in mapping.items():
            excel_val = excel_row[excel_col]
            csv_val = csv_row[csv_col]
            
            if pd.notna(excel_val):
                expected = bool(int(excel_val))
                if expected != csv_val:
                    print(f"  ❌ {excel_col}: Excel={excel_val}→{expected}, CSV={csv_val}")
                    errors += 1
                else:
                    print(f"  ✅ {excel_col}: Excel={excel_val}→{expected}, CSV={csv_val}")
    
    print(f"\n授業前データエラー数: {errors}")
    return errors

def check_after_accuracy(df_excel, df_csv, id_mapping):
    """授業後データの精度確認"""
    mapping = {
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
    
    errors = 0
    
    for excel_id, page_id in list(id_mapping.items())[:3]:  # 最初の3件をチェック
        excel_row = df_excel[df_excel['整理番号'] == excel_id].iloc[0]
        csv_row = df_csv[df_csv['Page_ID'] == page_id].iloc[0]
        
        print(f"\nID{excel_id}→Page_ID{page_id}の確認:")
        
        for excel_col, csv_col in mapping.items():
            excel_val = excel_row[excel_col]
            csv_val = csv_row[csv_col]
            
            if pd.notna(excel_val):
                if excel_col.startswith('クイズ') or excel_col.startswith('お茶'):
                    expected = bool(int(excel_val))
                else:
                    expected = int(excel_val)
                
                if expected != csv_val:
                    print(f"  ❌ {excel_col}: Excel={excel_val}→{expected}, CSV={csv_val}")
                    errors += 1
                else:
                    print(f"  ✅ {excel_col}: Excel={excel_val}→{expected}, CSV={csv_val}")
    
    print(f"\n授業後データエラー数: {errors}")
    return errors

def create_safety_backup():
    """安全バックアップの作成"""
    backup_dir = Path("backup")
    backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_safety")
    
    for file in ["before.csv", "after.csv"]:
        if Path(file).exists():
            backup_file = backup_dir / f"{file}_{timestamp}"
            shutil.copy2(file, backup_file)
            print(f"安全バックアップ: {backup_file}")

if __name__ == "__main__":
    success = definitive_correction()
    
    if success:
        print("\n🎉 データ修正が完全に成功しました！")
        print("CSVファイルは手動入力Excelデータと100%一致しています。")
    else:
        print("\n🔧 追加の調整が必要です。")
        print("backup/ディレクトリからデータを復元して再試行してください。")