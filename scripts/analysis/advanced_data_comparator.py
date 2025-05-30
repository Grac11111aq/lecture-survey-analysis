#!/usr/bin/env python3
"""
高度なデータ比較スクリプト
OCRデータとExcelデータの詳細比較と差分検出
"""

import pandas as pd
import numpy as np
from pathlib import Path

def detailed_data_investigation():
    """詳細なデータ調査"""
    print("=== 詳細なデータ調査 ===\n")
    
    # 元のExcelファイルを直接読み込み
    excel_file = "refference/250226アンケートデータ/250226アンケートデータ.xlsx"
    
    print("1. 元のExcelファイルの直接読み込み")
    print("-" * 50)
    
    # 授業前データ
    df_before_raw = pd.read_excel(excel_file, sheet_name="授業前")
    print(f"授業前データ（生データ）: {len(df_before_raw)} rows, {len(df_before_raw.columns)} columns")
    print("カラム:", list(df_before_raw.columns))
    print("\n最初の5行:")
    print(df_before_raw.head())
    print("\nクイズ1の値分布:")
    print(df_before_raw['クイズ1（〇が1，×が0）'].value_counts())
    print()
    
    # 授業後データ
    df_after_raw = pd.read_excel(excel_file, sheet_name="授業後")
    print(f"授業後データ（生データ）: {len(df_after_raw)} rows, {len(df_after_raw.columns)} columns")
    print("カラム:", list(df_after_raw.columns))
    print("\n最初の5行:")
    print(df_after_raw.head())
    print()
    
    # 2. OCRデータとの直接比較
    print("2. OCRデータとの直接比較")
    print("-" * 50)
    
    # OCRデータロード
    df_before_ocr = pd.read_csv("before.csv")
    df_after_ocr = pd.read_csv("after.csv")
    
    # クラス別の実際の比較
    compare_by_class(df_before_raw, df_before_ocr, df_after_raw, df_after_ocr)
    
    return df_before_raw, df_after_raw, df_before_ocr, df_after_ocr

def compare_by_class(df_before_raw, df_before_ocr, df_after_raw, df_after_ocr):
    """クラス別の詳細比較"""
    print("クラス別の詳細比較:")
    
    # 整理番号とPage_IDの対応を作成
    id_mapping = create_proper_id_mapping(df_before_raw, df_before_ocr)
    
    print(f"作成されたIDマッピング数: {len(id_mapping)}")
    print("マッピングサンプル:", dict(list(id_mapping.items())[:5]))
    print()
    
    # 実際の値比較
    differences = []
    
    # 授業前データの比較
    print("授業前データの詳細比較:")
    before_diffs = compare_before_detailed(df_before_raw, df_before_ocr, id_mapping)
    differences.extend(before_diffs)
    
    # 授業後データの比較
    print("\n授業後データの詳細比較:")
    after_diffs = compare_after_detailed(df_after_raw, df_after_ocr, id_mapping)
    differences.extend(after_diffs)
    
    # 差分レポートの保存
    if differences:
        df_differences = pd.DataFrame(differences)
        df_differences.to_csv("detailed_ocr_validation_report.csv", index=False, encoding='utf-8')
        print(f"\n差分レポート保存: detailed_ocr_validation_report.csv")
        print(f"総差分数: {len(differences)}")
        
        # エラータイプ別集計
        error_summary = df_differences['Error_Type'].value_counts()
        print("\nエラータイプ別集計:")
        for error_type, count in error_summary.items():
            print(f"  {error_type}: {count}件")
    else:
        print("\n差分は検出されませんでした。")
    
    return differences

def create_proper_id_mapping(df_excel, df_ocr):
    """適切なIDマッピングの作成"""
    mapping = {}
    
    # クラス別にマッピングを作成
    for cls in [1, 2, 3, 4]:
        # Excelデータでのクラス内の順序
        excel_class_data = df_excel[df_excel['クラス'] == cls].sort_values('整理番号')
        excel_ids = excel_class_data['整理番号'].tolist()
        
        # OCRデータでのクラス内の順序
        ocr_class_data = df_ocr[df_ocr['class'] == cls].sort_values('Page_ID')
        ocr_page_ids = ocr_class_data['Page_ID'].tolist()
        
        print(f"クラス {cls}: Excel {len(excel_ids)}人, OCR {len(ocr_page_ids)}人")
        
        # 人数が一致する場合のみマッピング
        if len(excel_ids) == len(ocr_page_ids):
            for excel_id, page_id in zip(excel_ids, ocr_page_ids):
                mapping[excel_id] = page_id
        else:
            print(f"  警告: クラス {cls} の人数が一致しません")
    
    return mapping

def compare_before_detailed(df_excel, df_ocr, id_mapping):
    """授業前データの詳細比較"""
    differences = []
    
    # カラムマッピング
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
    
    comparison_count = 0
    
    for excel_id, page_id in id_mapping.items():
        # Excelデータの行を取得
        excel_row = df_excel[df_excel['整理番号'] == excel_id]
        if len(excel_row) == 0:
            continue
        excel_row = excel_row.iloc[0]
        
        # OCRデータの行を取得
        ocr_row = df_ocr[df_ocr['Page_ID'] == page_id]
        if len(ocr_row) == 0:
            continue
        ocr_row = ocr_row.iloc[0]
        
        comparison_count += 1
        
        # 各カラムを比較
        for excel_col, ocr_col in column_mapping.items():
            excel_val = excel_row[excel_col]
            ocr_val = ocr_row[ocr_col]
            
            # 数値をTrueɿFalseに変換（1=True, 0=False）
            if excel_val == 1 or excel_val == 1.0:
                excel_val = True
            elif excel_val == 0 or excel_val == 0.0:
                excel_val = False
            
            # 値の比較
            if pd.notna(excel_val) and pd.notna(ocr_val) and excel_val != ocr_val:
                differences.append({
                    'Dataset': 'Before',
                    'Excel_ID': excel_id,
                    'Page_ID': page_id,
                    'Column': excel_col,
                    'OCR_Column': ocr_col,
                    'Excel_Value': excel_row[excel_col],  # 元の値
                    'Excel_Converted': excel_val,  # 変換後の値
                    'OCR_Value': ocr_val,
                    'Error_Type': 'Boolean_Mismatch'
                })
    
    print(f"比較実行数: {comparison_count}")
    print(f"検出された差分: {len(differences)}")
    
    # 差分の詳細表示
    if differences:
        print("差分の詳細:")
        for diff in differences[:5]:  # 最初の5件のみ表示
            print(f"  ID{diff['Excel_ID']}→{diff['Page_ID']}: {diff['Column']} Excel:{diff['Excel_Value']} vs OCR:{diff['OCR_Value']}")
    
    return differences

def compare_after_detailed(df_excel, df_ocr, id_mapping):
    """授業後データの詳細比較"""
    differences = []
    
    # カラムマッピング（授業後）
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
    
    comparison_count = 0
    
    for excel_id, page_id in id_mapping.items():
        # Excelデータの行を取得
        excel_row = df_excel[df_excel['整理番号'] == excel_id]
        if len(excel_row) == 0:
            continue
        excel_row = excel_row.iloc[0]
        
        # OCRデータの行を取得
        ocr_row = df_ocr[df_ocr['Page_ID'] == page_id]
        if len(ocr_row) == 0:
            continue
        ocr_row = ocr_row.iloc[0]
        
        comparison_count += 1
        
        # 各カラムを比較
        for excel_col, ocr_col in column_mapping.items():
            excel_val = excel_row[excel_col]
            ocr_val = ocr_row[ocr_col]
            
            # ブール値の変換（数値から）
            if excel_col.startswith('クイズ') or excel_col.startswith('お茶'):
                if excel_val == 1 or excel_val == 1.0:
                    excel_val = True
                elif excel_val == 0 or excel_val == 0.0:
                    excel_val = False
            
            # 値の比較
            if pd.notna(excel_val) and pd.notna(ocr_val) and excel_val != ocr_val:
                error_type = 'Boolean_Mismatch' if isinstance(excel_val, bool) else 'Rating_Mismatch'
                differences.append({
                    'Dataset': 'After',
                    'Excel_ID': excel_id,
                    'Page_ID': page_id,
                    'Column': excel_col,
                    'OCR_Column': ocr_col,
                    'Excel_Value': excel_row[excel_col],  # 元の値
                    'Excel_Converted': excel_val,  # 変換後の値
                    'OCR_Value': ocr_val,
                    'Error_Type': error_type
                })
    
    print(f"比較実行数: {comparison_count}")
    print(f"検出された差分: {len(differences)}")
    
    # 差分の詳細表示
    if differences:
        print("差分の詳細:")
        for diff in differences[:5]:  # 最初の5件のみ表示
            print(f"  ID{diff['Excel_ID']}→{diff['Page_ID']}: {diff['Column']} Excel:{diff['Excel_Value']} vs OCR:{diff['OCR_Value']}")
    
    return differences

def create_correction_script(differences):
    """修正スクリプトの生成"""
    if not differences:
        print("修正が必要な差分が見つかりませんでした。")
        return
    
    print("\n=== 修正スクリプトの生成 ===")
    
    # DataFrameに変換
    df_diff = pd.DataFrame(differences)
    
    # 修正スクリプト用のコードを生成
    correction_script = generate_correction_code(df_diff)
    
    # スクリプトをファイルに保存
    with open("csv_correction_script.py", "w", encoding="utf-8") as f:
        f.write(correction_script)
    
    print("修正スクリプトを生成しました: csv_correction_script.py")
    print(f"修正対象: {len(differences)}件")

def generate_correction_code(df_diff):
    """修正コードの生成"""
    script_template = '''#!/usr/bin/env python3
"""
OCRデータ修正スクリプト
手動入力されたExcelデータに基づいてOCRデータの誤りを修正
"""

import pandas as pd
import numpy as np
from datetime import datetime

def apply_corrections():
    """検出された差分に基づいてCSVファイルを修正"""
    print("=== OCRデータ修正の実行 ===")
    
    # バックアップの作成
    create_backup_files()
    
    # 修正の適用
    corrections_applied = 0
    
{correction_code}
    
    print(f"修正適用完了: {corrections_applied}件")
    print("修正されたファイルは元のファイルを上書きしました")
    print("バックアップファイルは backup/ ディレクトリに保存されています")

def create_backup_files():
    """バックアップファイルの作成"""
    import shutil
    from pathlib import Path
    
    backup_dir = Path("backup")
    backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    files_to_backup = ["before.csv", "after.csv", "comment.csv"]
    for file in files_to_backup:
        if Path(file).exists():
            backup_file = backup_dir / f"{file}_{timestamp}"
            shutil.copy2(file, backup_file)
            print(f"バックアップ作成: {backup_file}")

if __name__ == "__main__":
    apply_corrections()
'''
    
    # 修正コードの生成
    correction_code = ""
    for dataset in ['Before', 'After']:
        dataset_diffs = df_diff[df_diff['Dataset'] == dataset]
        if len(dataset_diffs) > 0:
            file_name = "before.csv" if dataset == 'Before' else "after.csv"
            correction_code += f'''
    # {dataset}データの修正
    print("修正中: {file_name}")
    df = pd.read_csv("{file_name}")
'''
            
            for _, row in dataset_diffs.iterrows():
                page_id = row['Page_ID']
                ocr_col = row['OCR_Column']
                excel_converted = row['Excel_Converted']
                
                correction_code += f'''    # ID{row['Excel_ID']}→Page_ID{page_id}: {row['Column']}の修正
    df.loc[df['Page_ID'] == {page_id}, '{ocr_col}'] = {repr(excel_converted)}
    corrections_applied += 1
'''
            
            correction_code += f'''    df.to_csv("{file_name}", index=False, encoding='utf-8')
'''
    
    return script_template.format(correction_code=correction_code)

if __name__ == "__main__":
    # 詳細調査の実行
    differences = detailed_data_investigation()
    
    # 修正スクリプトの生成
    if differences:
        create_correction_script(differences)