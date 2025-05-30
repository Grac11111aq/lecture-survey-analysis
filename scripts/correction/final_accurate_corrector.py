#!/usr/bin/env python3
"""
最終正確データ修正スクリプト
正確なIDマッピングに基づいて完全修正を実行
"""

import pandas as pd
import numpy as np
from datetime import datetime
import shutil
from pathlib import Path

def final_accurate_correction():
    """正確なIDマッピングに基づく最終修正"""
    print("=== 最終正確データ修正の実行 ===\n")
    
    # 最終安全バックアップ
    create_final_backup()
    
    # 正確なIDマッピングを読み込み
    mapping_df = pd.read_csv("correct_id_mapping.csv")
    id_mapping = dict(zip(mapping_df['Excel_ID'], mapping_df['Page_ID']))
    print(f"正確なIDマッピング読み込み: {len(id_mapping)}組")
    
    # 元のExcelファイルを読み込み
    excel_file = "refference/250226アンケートデータ/250226アンケートデータ.xlsx"
    df_before_excel = pd.read_excel(excel_file, sheet_name="授業前")
    df_after_excel = pd.read_excel(excel_file, sheet_name="授業後")
    
    # オリジナルのCSVファイルを復元（バックアップから）
    df_before_csv = pd.read_csv("backup/before.csv_20250530_154506")
    df_after_csv = pd.read_csv("backup/after.csv_20250530_154506")
    
    print(f"データ読み込み完了:")
    print(f"  Excel授業前: {len(df_before_excel)}行")
    print(f"  Excel授業後: {len(df_after_excel)}行")
    print(f"  CSV授業前: {len(df_before_csv)}行")
    print(f"  CSV授業後: {len(df_after_csv)}行")
    
    # 正確な修正を実行
    df_before_corrected = correct_before_accurately(df_before_excel, df_before_csv, id_mapping)
    df_after_corrected = correct_after_accurately(df_after_excel, df_after_csv, id_mapping)
    
    # 修正データを保存
    df_before_corrected.to_csv("before.csv", index=False, encoding='utf-8')
    df_after_corrected.to_csv("after.csv", index=False, encoding='utf-8')
    
    print("\n=== 修正完了 ===")
    
    # 最終検証
    final_verification(df_before_excel, df_after_excel, id_mapping)

def correct_before_accurately(df_excel, df_csv, id_mapping):
    """授業前データの正確な修正"""
    print("\n授業前データの正確な修正:")
    
    df_corrected = df_csv.copy()
    
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
    
    corrections = 0
    successful_mappings = 0
    
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
        
        successful_mappings += 1
        
        # 各カラムを正確に修正
        for excel_col, csv_col in column_mapping.items():
            excel_val = excel_row[excel_col]
            
            if pd.notna(excel_val):
                # 1.0 → True, 0.0 → False
                correct_bool = bool(int(excel_val))
                
                # 現在の値と比較
                current_val = df_corrected.at[csv_idx, csv_col]
                if current_val != correct_bool:
                    df_corrected.at[csv_idx, csv_col] = correct_bool
                    corrections += 1
    
    print(f"  処理対象マッピング: {successful_mappings}組")
    print(f"  修正項目数: {corrections}")
    return df_corrected

def correct_after_accurately(df_excel, df_csv, id_mapping):
    """授業後データの正確な修正"""
    print("\n授業後データの正確な修正:")
    
    df_corrected = df_csv.copy()
    
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
    
    corrections = 0
    successful_mappings = 0
    
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
        
        successful_mappings += 1
        
        # 各カラムを正確に修正
        for excel_col, csv_col in column_mapping.items():
            excel_val = excel_row[excel_col]
            
            if pd.notna(excel_val):
                if excel_col.startswith('クイズ') or excel_col.startswith('お茶'):
                    # ブール値: 1.0 → True, 0.0 → False
                    correct_val = bool(int(excel_val))
                else:
                    # 評価値: そのまま整数として使用
                    correct_val = int(excel_val)
                
                # 現在の値と比較
                current_val = df_corrected.at[csv_idx, csv_col]
                if current_val != correct_val:
                    df_corrected.at[csv_idx, csv_col] = correct_val
                    corrections += 1
    
    print(f"  処理対象マッピング: {successful_mappings}組")
    print(f"  修正項目数: {corrections}")
    return df_corrected

def final_verification(df_excel_before, df_excel_after, id_mapping):
    """最終検証"""
    print("\n=== 最終検証 ===")
    
    # 修正後のCSVを読み込み
    df_csv_before = pd.read_csv("before.csv")
    df_csv_after = pd.read_csv("after.csv")
    
    # 授業前データの検証
    before_errors = verify_before_final(df_excel_before, df_csv_before, id_mapping)
    
    # 授業後データの検証
    after_errors = verify_after_final(df_excel_after, df_csv_after, id_mapping)
    
    total_errors = before_errors + after_errors
    
    print(f"\n=== 検証結果 ===")
    print(f"授業前データエラー: {before_errors}件")
    print(f"授業後データエラー: {after_errors}件")
    print(f"総エラー数: {total_errors}件")
    
    if total_errors == 0:
        print("\n🎉 完全一致達成！")
        print("修正後のCSVデータは手動入力Excelデータと100%一致しています。")
        
        # 成功レポートの作成
        create_success_report(id_mapping)
        return True
    else:
        print(f"\n⚠️ まだ{total_errors}件の不一致があります")
        return False

def verify_before_final(df_excel, df_csv, id_mapping):
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
    
    errors = 0
    comparisons = 0
    
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
                expected = bool(int(excel_val))
                comparisons += 1
                
                if expected != csv_val:
                    errors += 1
                    if errors <= 5:  # 最初の5件のみ表示
                        print(f"  ❌ ID{excel_id}→Page_ID{page_id}: {excel_col} Excel:{excel_val}→{expected} CSV:{csv_val}")
    
    print(f"授業前検証: {comparisons}項目中{errors}件のエラー")
    return errors

def verify_after_final(df_excel, df_csv, id_mapping):
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
    
    errors = 0
    comparisons = 0
    
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
                    expected = bool(int(excel_val))
                else:
                    expected = int(excel_val)
                
                comparisons += 1
                
                if expected != csv_val:
                    errors += 1
                    if errors <= 5:  # 最初の5件のみ表示
                        print(f"  ❌ ID{excel_id}→Page_ID{page_id}: {excel_col} Excel:{excel_val}→{expected} CSV:{csv_val}")
    
    print(f"授業後検証: {comparisons}項目中{errors}件のエラー")
    return errors

def create_success_report(id_mapping):
    """成功レポートの作成"""
    with open("final_correction_success_report.txt", "w", encoding="utf-8") as f:
        f.write("OCRデータ修正完了レポート\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"修正完了日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")
        
        f.write("✅ 修正結果: 完全成功\n")
        f.write("手動入力Excelデータと100%一致しました\n\n")
        
        f.write(f"処理対象データ:\n")
        f.write(f"  - IDマッピング: {len(id_mapping)}組\n")
        f.write(f"  - 授業前データ: 8項目/人\n")
        f.write(f"  - 授業後データ: 11項目/人\n\n")
        
        f.write("修正項目:\n")
        f.write("  - ブール値（○×回答）の誤認識修正\n")
        f.write("  - 評価値（1-4段階）の誤認識修正\n")
        f.write("  - NaN値の適切な処理\n\n")
        
        f.write("データ品質:\n")
        f.write("  - 授業前CSV: 手動入力データと完全一致\n")
        f.write("  - 授業後CSV: 手動入力データと完全一致\n")
        f.write("  - データ整合性: 確認済み\n\n")
        
        f.write("次のステップ:\n")
        f.write("  1. 修正済みCSVファイルを使用した分析の実行\n")
        f.write("  2. アンケート結果の統計分析\n")
        f.write("  3. 出前授業の効果測定\n")
    
    print("成功レポートを作成しました: final_correction_success_report.txt")

def create_final_backup():
    """最終バックアップの作成"""
    backup_dir = Path("backup")
    backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_final_accurate")
    
    for file in ["before.csv", "after.csv"]:
        if Path(file).exists():
            backup_file = backup_dir / f"{file}_{timestamp}"
            shutil.copy2(file, backup_file)
            print(f"最終バックアップ: {backup_file}")

if __name__ == "__main__":
    success = final_accurate_correction()
    
    if success:
        print("\n🌟 データ修正が完全に成功しました！")
        print("CSVファイルは手動入力Excelデータと100%一致しています。")
        print("これで信頼性の高いアンケート分析を実施できます。")
    else:
        print("\n🔧 追加の調整が必要です。")
        print("ログを確認して問題を特定してください。")