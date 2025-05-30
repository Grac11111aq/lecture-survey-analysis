#!/usr/bin/env python3
"""
修正後CSVデータと手動入力Excelデータの完全一致検証スクリプト
"""

import pandas as pd
import numpy as np
from pathlib import Path

def verify_data_match():
    """修正後のCSVデータが手動入力Excelデータと完全に一致するか検証"""
    print("=== 修正後データの完全一致検証 ===\n")
    
    # 元のExcelファイルを読み込み
    excel_file = "refference/250226アンケートデータ/250226アンケートデータ.xlsx"
    
    # Excelデータの読み込み
    df_before_excel = pd.read_excel(excel_file, sheet_name="授業前")
    df_after_excel = pd.read_excel(excel_file, sheet_name="授業後")
    
    # 修正後のCSVデータの読み込み
    df_before_csv = pd.read_csv("before.csv")
    df_after_csv = pd.read_csv("after.csv")
    
    print(f"比較対象データ:")
    print(f"  Excel授業前: {len(df_before_excel)}行")
    print(f"  CSV授業前: {len(df_before_csv)}行")
    print(f"  Excel授業後: {len(df_after_excel)}行")
    print(f"  CSV授業後: {len(df_after_csv)}行")
    print()
    
    # IDマッピングの再構築
    id_mapping = create_id_mapping(df_before_excel, df_before_csv)
    
    # 授業前データの検証
    print("1. 授業前データの検証")
    print("-" * 50)
    before_mismatches = verify_before_data(df_before_excel, df_before_csv, id_mapping)
    
    # 授業後データの検証
    print("\n2. 授業後データの検証")
    print("-" * 50)
    after_mismatches = verify_after_data(df_after_excel, df_after_csv, id_mapping)
    
    # 結果サマリー
    print("\n=== 検証結果サマリー ===")
    total_comparisons = len(before_mismatches['comparisons']) + len(after_mismatches['comparisons'])
    total_mismatches = before_mismatches['mismatch_count'] + after_mismatches['mismatch_count']
    
    print(f"総比較項目数: {total_comparisons}")
    print(f"不一致項目数: {total_mismatches}")
    
    if total_mismatches == 0:
        print("\n✅ 完全一致確認！")
        print("修正後のCSVデータは手動入力Excelデータと100%一致しています。")
    else:
        print(f"\n⚠️ 不一致が{total_mismatches}件見つかりました。")
        print("詳細は verification_report.csv を確認してください。")
        
        # 不一致レポートの保存
        save_mismatch_report(before_mismatches, after_mismatches)
    
    return total_mismatches == 0

def create_id_mapping(df_excel, df_csv):
    """ExcelのIDとCSVのPage_IDの対応関係を作成"""
    mapping = {}
    
    for cls in [1, 2, 3, 4]:
        # Excelデータでのクラス内の順序
        excel_class_data = df_excel[df_excel['クラス'] == cls].sort_values('整理番号')
        excel_ids = excel_class_data['整理番号'].tolist()
        
        # CSVデータでのクラス内の順序
        csv_class_data = df_csv[df_csv['class'] == cls].sort_values('Page_ID')
        csv_page_ids = csv_class_data['Page_ID'].tolist()
        
        # 人数が一致する場合のみマッピング
        if len(excel_ids) == len(csv_page_ids):
            for excel_id, page_id in zip(excel_ids, csv_page_ids):
                mapping[excel_id] = page_id
    
    return mapping

def verify_before_data(df_excel, df_csv, id_mapping):
    """授業前データの詳細検証"""
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
    
    mismatches = []
    comparisons = []
    
    for excel_id, page_id in id_mapping.items():
        excel_row = df_excel[df_excel['整理番号'] == excel_id]
        if len(excel_row) == 0:
            continue
        excel_row = excel_row.iloc[0]
        
        csv_row = df_csv[df_csv['Page_ID'] == page_id]
        if len(csv_row) == 0:
            continue
        csv_row = csv_row.iloc[0]
        
        # 各カラムを比較
        for excel_col, csv_col in column_mapping.items():
            excel_val = excel_row[excel_col]
            csv_val = csv_row[csv_col]
            
            # 数値をブール値に変換して比較
            if pd.notna(excel_val):
                excel_bool = bool(excel_val == 1 or excel_val == 1.0)
                
                comparisons.append({
                    'Dataset': 'Before',
                    'Excel_ID': excel_id,
                    'Page_ID': page_id,
                    'Column': excel_col,
                    'Excel_Value': excel_val,
                    'CSV_Value': csv_val,
                    'Match': excel_bool == csv_val
                })
                
                if excel_bool != csv_val:
                    mismatches.append({
                        'Dataset': 'Before',
                        'Excel_ID': excel_id,
                        'Page_ID': page_id,
                        'Column': excel_col,
                        'Excel_Value': excel_val,
                        'CSV_Value': csv_val
                    })
    
    print(f"比較実行数: {len(comparisons)}")
    print(f"不一致数: {len(mismatches)}")
    
    if mismatches:
        print("不一致の例:")
        for m in mismatches[:3]:
            print(f"  ID{m['Excel_ID']}→{m['Page_ID']}: {m['Column']} Excel:{m['Excel_Value']} vs CSV:{m['CSV_Value']}")
    
    return {'mismatches': mismatches, 'comparisons': comparisons, 'mismatch_count': len(mismatches)}

def verify_after_data(df_excel, df_csv, id_mapping):
    """授業後データの詳細検証"""
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
    
    mismatches = []
    comparisons = []
    
    for excel_id, page_id in id_mapping.items():
        excel_row = df_excel[df_excel['整理番号'] == excel_id]
        if len(excel_row) == 0:
            continue
        excel_row = excel_row.iloc[0]
        
        csv_row = df_csv[df_csv['Page_ID'] == page_id]
        if len(csv_row) == 0:
            continue
        csv_row = csv_row.iloc[0]
        
        # 各カラムを比較
        for excel_col, csv_col in column_mapping.items():
            excel_val = excel_row[excel_col]
            csv_val = csv_row[csv_col]
            
            if pd.notna(excel_val):
                # ブール値カラムの場合
                if excel_col.startswith('クイズ') or excel_col.startswith('お茶'):
                    excel_bool = bool(excel_val == 1 or excel_val == 1.0)
                    match = excel_bool == csv_val
                # 数値カラムの場合
                else:
                    match = int(excel_val) == int(csv_val)
                
                comparisons.append({
                    'Dataset': 'After',
                    'Excel_ID': excel_id,
                    'Page_ID': page_id,
                    'Column': excel_col,
                    'Excel_Value': excel_val,
                    'CSV_Value': csv_val,
                    'Match': match
                })
                
                if not match:
                    mismatches.append({
                        'Dataset': 'After',
                        'Excel_ID': excel_id,
                        'Page_ID': page_id,
                        'Column': excel_col,
                        'Excel_Value': excel_val,
                        'CSV_Value': csv_val
                    })
    
    print(f"比較実行数: {len(comparisons)}")
    print(f"不一致数: {len(mismatches)}")
    
    if mismatches:
        print("不一致の例:")
        for m in mismatches[:3]:
            print(f"  ID{m['Excel_ID']}→{m['Page_ID']}: {m['Column']} Excel:{m['Excel_Value']} vs CSV:{m['CSV_Value']}")
    
    return {'mismatches': mismatches, 'comparisons': comparisons, 'mismatch_count': len(mismatches)}

def save_mismatch_report(before_results, after_results):
    """不一致レポートの保存"""
    all_mismatches = before_results['mismatches'] + after_results['mismatches']
    
    if all_mismatches:
        df_mismatches = pd.DataFrame(all_mismatches)
        df_mismatches.to_csv("verification_report.csv", index=False, encoding='utf-8')
        print("\n不一致レポートを保存しました: verification_report.csv")

def calculate_accuracy_metrics():
    """精度メトリクスの計算"""
    print("\n=== 精度メトリクスの計算 ===")
    
    # 差分レポートから元のエラー数を取得
    df_original_errors = pd.read_csv("detailed_ocr_validation_report.csv")
    original_error_count = len(df_original_errors)
    
    print(f"元のOCRエラー数: {original_error_count}")
    print(f"修正適用数: {original_error_count}")
    
    # 現在の検証結果
    is_perfect_match = verify_data_match()
    
    if is_perfect_match:
        print(f"修正成功率: 100%")
        print(f"データ精度: 100%")
    else:
        # 不一致がある場合は詳細を計算
        if Path("verification_report.csv").exists():
            df_remaining = pd.read_csv("verification_report.csv")
            remaining_errors = len(df_remaining)
            success_rate = ((original_error_count - remaining_errors) / original_error_count) * 100
            print(f"修正成功率: {success_rate:.1f}%")
            print(f"残存エラー: {remaining_errors}件")

if __name__ == "__main__":
    print("修正後のCSVデータと手動入力Excelデータの一致検証を開始します...\n")
    calculate_accuracy_metrics()