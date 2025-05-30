#!/usr/bin/env python3
"""
特定カラムの一致率検証スクリプト
Q1_Sugarwater_ResponseとQ1_Ink_Responseの
ExcelファイルとCSVファイルの一致率を詳細に分析
"""

import pandas as pd
import numpy as np

def validate_specific_columns():
    """Q1_Sugarwater_ResponseとQ1_Ink_Responseの一致率検証"""
    print("=== 特定カラムの一致率検証 ===\n")
    
    # データ読み込み
    excel_file = "refference/250226アンケートデータ/250226アンケートデータ.xlsx"
    df_before_excel = pd.read_excel(excel_file, sheet_name="授業前")
    df_before_csv = pd.read_csv("before.csv")
    id_mapping = pd.read_csv("correct_id_mapping.csv")
    
    print(f"データ読み込み:")
    print(f"  Excel授業前: {len(df_before_excel)}行")
    print(f"  CSV授業前: {len(df_before_csv)}行")
    print(f"  IDマッピング: {len(id_mapping)}組")
    print()
    
    # 対象カラムの定義
    target_columns = {
        'Q1_Sugarwater_Response': 'クイズ2',
        'Q1_Ink_Response': 'クイズ4'
    }
    
    results = {}
    
    for csv_col, excel_col in target_columns.items():
        print(f"=== {csv_col} ({excel_col}) の検証 ===")
        result = validate_single_column(
            df_before_excel, df_before_csv, id_mapping,
            excel_col, csv_col
        )
        results[csv_col] = result
        print()
    
    # 総合結果
    print("=== 総合結果 ===")
    for col, result in results.items():
        print(f"{col}:")
        print(f"  一致率: {result['match_rate']:.1f}%")
        print(f"  一致数: {result['matches']}/{result['total']}")
        print(f"  不一致数: {result['mismatches']}")
    
    return results

def validate_single_column(df_excel, df_csv, id_mapping, excel_col, csv_col):
    """単一カラムの詳細検証"""
    
    matches = 0
    mismatches = 0
    total_comparisons = 0
    mismatch_details = []
    
    # IDマッピングでマッチング
    id_map_dict = dict(zip(id_mapping['Excel_ID'], id_mapping['Page_ID']))
    
    for excel_id, page_id in id_map_dict.items():
        # Excel行を取得
        excel_rows = df_excel[df_excel['整理番号'] == excel_id]
        if len(excel_rows) == 0:
            continue
        excel_row = excel_rows.iloc[0]
        
        # CSV行を取得
        csv_rows = df_csv[df_csv['Page_ID'] == page_id]
        if len(csv_rows) == 0:
            continue
        csv_row = csv_rows.iloc[0]
        
        # 値を取得
        excel_val = excel_row[excel_col]
        csv_val = csv_row[csv_col]
        
        # 値が存在する場合のみ比較
        if pd.notna(excel_val):
            total_comparisons += 1
            
            # Excelの1/0をTrueɿFalseに変換
            excel_bool = bool(int(excel_val))
            
            if excel_bool == csv_val:
                matches += 1
            else:
                mismatches += 1
                mismatch_details.append({
                    'excel_id': excel_id,
                    'page_id': page_id,
                    'excel_val': excel_val,
                    'excel_bool': excel_bool,
                    'csv_val': csv_val,
                    'class': excel_row['クラス']
                })
    
    # 結果の表示
    match_rate = (matches / total_comparisons * 100) if total_comparisons > 0 else 0
    
    print(f"総比較数: {total_comparisons}")
    print(f"一致数: {matches}")
    print(f"不一致数: {mismatches}")
    print(f"一致率: {match_rate:.1f}%")
    
    # 不一致の詳細表示（最初の10件）
    if mismatch_details:
        print(f"\n不一致の詳細（最初の10件）:")
        for i, detail in enumerate(mismatch_details[:10]):
            print(f"  {i+1}. Excel_ID{detail['excel_id']} → Page_ID{detail['page_id']} (クラス{detail['class']})")
            print(f"     Excel: {detail['excel_val']} → {detail['excel_bool']}, CSV: {detail['csv_val']}")
        
        # クラス別の不一致分布
        class_mismatches = {}
        for detail in mismatch_details:
            cls = detail['class']
            if cls not in class_mismatches:
                class_mismatches[cls] = 0
            class_mismatches[cls] += 1
        
        print(f"\nクラス別不一致数:")
        for cls in sorted(class_mismatches.keys()):
            print(f"  クラス{cls}: {class_mismatches[cls]}件")
    
    return {
        'total': total_comparisons,
        'matches': matches,
        'mismatches': mismatches,
        'match_rate': match_rate,
        'mismatch_details': mismatch_details
    }

def analyze_mismatch_patterns(results):
    """不一致パターンの分析"""
    print("\n=== 不一致パターンの分析 ===")
    
    for col, result in results.items():
        if result['mismatch_details']:
            print(f"\n{col} の不一致パターン:")
            
            # True→Falseパターン
            true_to_false = [d for d in result['mismatch_details'] 
                           if d['excel_bool'] == True and d['csv_val'] == False]
            
            # False→Trueパターン  
            false_to_true = [d for d in result['mismatch_details'] 
                           if d['excel_bool'] == False and d['csv_val'] == True]
            
            print(f"  True→False: {len(true_to_false)}件")
            print(f"  False→True: {len(false_to_true)}件")
            
            if true_to_false:
                print(f"  True→False の例:")
                for detail in true_to_false[:3]:
                    print(f"    Excel_ID{detail['excel_id']} → Page_ID{detail['page_id']}")
                    
            if false_to_true:
                print(f"  False→True の例:")
                for detail in false_to_true[:3]:
                    print(f"    Excel_ID{detail['excel_id']} → Page_ID{detail['page_id']}")

def generate_detailed_validation_report(results):
    """詳細な検証レポートの生成"""
    
    report_lines = [
        "Q1_Sugarwater_Response と Q1_Ink_Response 検証レポート",
        "=" * 60,
        "",
        f"検証日時: {pd.Timestamp.now().strftime('%Y年%m月%d日 %H:%M:%S')}",
        "",
        "## 検証概要",
        "手動入力Excelファイルと現在のCSVファイルの",
        "Q1_Sugarwater_Response と Q1_Ink_Response の一致率を検証",
        "",
        "## 検証結果",
    ]
    
    for col, result in results.items():
        report_lines.extend([
            f"",
            f"### {col}",
            f"- 総比較数: {result['total']}",
            f"- 一致数: {result['matches']}",
            f"- 不一致数: {result['mismatches']}",
            f"- **一致率: {result['match_rate']:.1f}%**",
        ])
        
        if result['mismatch_details']:
            report_lines.append("")
            report_lines.append("#### 不一致の詳細")
            for detail in result['mismatch_details'][:5]:
                report_lines.append(
                    f"- Excel_ID{detail['excel_id']} → Page_ID{detail['page_id']}: "
                    f"Excel={detail['excel_val']}({detail['excel_bool']}) ≠ CSV={detail['csv_val']}"
                )
    
    report_lines.extend([
        "",
        "## 結論",
        "",
        f"Q1_Sugarwater_Response: {results['Q1_Sugarwater_Response']['match_rate']:.1f}% 一致",
        f"Q1_Ink_Response: {results['Q1_Ink_Response']['match_rate']:.1f}% 一致",
        "",
        "不一致がある場合は、OCR修正処理の追加調整が必要です。"
    ])
    
    with open("specific_columns_validation_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print("\n詳細レポートを保存しました: specific_columns_validation_report.txt")

if __name__ == "__main__":
    results = validate_specific_columns()
    analyze_mismatch_patterns(results)
    generate_detailed_validation_report(results)