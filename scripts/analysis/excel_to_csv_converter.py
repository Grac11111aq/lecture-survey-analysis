#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ExcelファイルをCSVに変換するスクリプト
OCRデータの検証用に手動入力されたExcelデータをCSVに変換
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def convert_excel_to_csv():
    """
    Excelファイル（250226アンケートデータ.xlsx）を
    既存のCSVファイルと比較可能な形式に変換
    """
    
    # ファイルパス
    excel_file = "refference/250226アンケートデータ/250226アンケートデータ.xlsx"
    output_dir = "validation_data"
    
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading Excel file: {excel_file}")
    
    # Excelファイルの読み込み
    try:
        # 授業前データ
        df_before_excel = pd.read_excel(excel_file, sheet_name="授業前")
        # 授業後データ  
        df_after_excel = pd.read_excel(excel_file, sheet_name="授業後")
        # 感想文データ
        df_comments_excel = pd.read_excel(excel_file, sheet_name="お礼の手紙の記述")
        
        print(f"授業前データ: {len(df_before_excel)} rows, {len(df_before_excel.columns)} columns")
        print(f"授業後データ: {len(df_after_excel)} rows, {len(df_after_excel.columns)} columns")
        print(f"感想文データ: {len(df_comments_excel)} rows, {len(df_comments_excel.columns)} columns")
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return
    
    # データの前処理と変換
    
    # 1. 授業前データの変換
    print("\n=== 授業前データの変換 ===")
    df_before_converted = convert_before_data(df_before_excel)
    
    # 2. 授業後データの変換 
    print("\n=== 授業後データの変換 ===")
    df_after_converted = convert_after_data(df_after_excel)
    
    # 3. 感想文データの変換
    print("\n=== 感想文データの変換 ===")
    df_comments_converted = convert_comments_data(df_comments_excel)
    
    # CSVファイルとして保存
    before_output = f"{output_dir}/before_excel.csv"
    after_output = f"{output_dir}/after_excel.csv" 
    comments_output = f"{output_dir}/comments_excel.csv"
    
    df_before_converted.to_csv(before_output, index=False, encoding='utf-8')
    df_after_converted.to_csv(after_output, index=False, encoding='utf-8')
    df_comments_converted.to_csv(comments_output, index=False, encoding='utf-8')
    
    print(f"\n=== 変換完了 ===")
    print(f"授業前データ: {before_output}")
    print(f"授業後データ: {after_output}")
    print(f"感想文データ: {comments_output}")
    
    return df_before_converted, df_after_converted, df_comments_converted

def convert_before_data(df):
    """授業前データをCSVフォーマットに変換"""
    
    # カラム名の対応表（Excelのカラム → CSVのカラム）
    column_mapping = {
        '整理番号': 'Excel_ID',
        'クラス': 'class',
        'クイズ1': 'Q1_Saltwater_Response',
        'クイズ2': 'Q1_Sugarwater_Response', 
        'クイズ3': 'Q1_Muddywater_Response',
        'クイズ4': 'Q1_Ink_Response',
        'クイズ5': 'Q1_MisoSoup_Response',
        'クイズ6': 'Q1_SoySauce_Response',
        '味噌汁記述': 'Q2_MisoSalty_Reason',
        'お茶クイズ1': 'Q3_TeaLeavesDissolve',
        'お茶クイズ2': 'Q3_TeaComponentsDissolve'
    }
    
    # カラム名変更
    df_converted = df.rename(columns=column_mapping)
    
    # ○×をTrue/Falseに変換
    bool_columns = [
        'Q1_Saltwater_Response', 'Q1_Sugarwater_Response', 'Q1_Muddywater_Response',
        'Q1_Ink_Response', 'Q1_MisoSoup_Response', 'Q1_SoySauce_Response',
        'Q3_TeaLeavesDissolve', 'Q3_TeaComponentsDissolve'
    ]
    
    for col in bool_columns:
        if col in df_converted.columns:
            df_converted[col] = df_converted[col].map({'○': True, '×': False})
    
    # Page_IDを生成（後で対応関係を作成）
    df_converted['Page_ID'] = df_converted['Excel_ID'].apply(lambda x: f"PAGE_{x:03d}")
    
    # カラムの順序を調整
    columns_order = ['Page_ID', 'Excel_ID', 'class'] + bool_columns + ['Q2_MisoSalty_Reason']
    df_converted = df_converted.reindex(columns=columns_order)
    
    print(f"変換後: {len(df_converted)} rows, {len(df_converted.columns)} columns")
    print("カラム名:", list(df_converted.columns))
    
    return df_converted

def convert_after_data(df):
    """授業後データをCSVフォーマットに変換"""
    
    # 授業前のカラムマッピング + 授業後固有項目
    column_mapping = {
        '整理番号': 'Excel_ID',
        'クラス': 'class',
        'クイズ1': 'Q1_Saltwater',
        'クイズ2': 'Q1_Sugarwater',
        'クイズ3': 'Q1_Muddywater', 
        'クイズ4': 'Q1_Ink',
        'クイズ5': 'Q1_MisoSoup',
        'クイズ6': 'Q1_SoySauce',
        '味噌汁記述': 'Q2_MisoSaltyReason',
        'お茶クイズ1': 'Q3_TeaLeaves_DissolveInWater',
        'お茶クイズ2': 'Q3_TeaComponents_DissolveInWater',
        'おもしろさ': 'Q4_ExperimentInterestRating',
        '新発見': 'Q5_NewLearningsRating',
        '理解': 'Q6_DissolvingUnderstandingRating'
    }
    
    # カラム名変更
    df_converted = df.rename(columns=column_mapping)
    
    # ○×をTrue/Falseに変換
    bool_columns = [
        'Q1_Saltwater', 'Q1_Sugarwater', 'Q1_Muddywater',
        'Q1_Ink', 'Q1_MisoSoup', 'Q1_SoySauce',
        'Q3_TeaLeaves_DissolveInWater', 'Q3_TeaComponents_DissolveInWater'
    ]
    
    for col in bool_columns:
        if col in df_converted.columns:
            df_converted[col] = df_converted[col].map({'○': True, '×': False})
    
    # Page_IDを生成
    df_converted['Page_ID'] = df_converted['Excel_ID'].apply(lambda x: f"PAGE_{x:03d}")
    
    # 評価項目の処理（1-4の数値）
    rating_columns = ['Q4_ExperimentInterestRating', 'Q5_NewLearningsRating', 'Q6_DissolvingUnderstandingRating']
    for col in rating_columns:
        if col in df_converted.columns:
            # 数値データをintに変換
            df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
    
    # CSVに存在しない項目を追加（プレースホルダー）
    df_converted['Q4_ExperimentInterestComment'] = ""
    df_converted['Q6_DissolvingUnderstandingComment'] = ""
    df_converted['GeneralPageComments'] = ""
    
    # カラムの順序を調整
    columns_order = [
        'Page_ID', 'Excel_ID', 'class'
    ] + bool_columns + [
        'Q2_MisoSaltyReason',
        'Q4_ExperimentInterestRating', 'Q4_ExperimentInterestComment',
        'Q5_NewLearningsRating',
        'Q6_DissolvingUnderstandingRating', 'Q6_DissolvingUnderstandingComment',
        'GeneralPageComments'
    ]
    
    df_converted = df_converted.reindex(columns=columns_order)
    
    print(f"変換後: {len(df_converted)} rows, {len(df_converted.columns)} columns")
    print("カラム名:", list(df_converted.columns))
    
    return df_converted

def convert_comments_data(df):
    """感想文データをCSVフォーマットに変換"""
    
    # NaNでない行のみを抽出
    df_filtered = df.dropna(subset=['記述'])
    
    # カラム名変更
    column_mapping = {
        '整理番号': 'Excel_ID',
        '記述': 'comment'
    }
    
    df_converted = df_filtered.rename(columns=column_mapping)
    
    # Page_IDを生成
    df_converted['Page_ID'] = df_converted['Excel_ID'].apply(lambda x: f"PAGE_{x:03d}")
    
    # クラス情報を追加（仮）
    df_converted['class'] = df_converted['Excel_ID'].apply(
        lambda x: 1 if x <= 24 else 
                 2 if x <= 50 else
                 3 if x <= 74 else 4
    )
    
    # page-IDとLRカラムを追加（CSVとの互換性のため）
    df_converted['page-ID'] = df_converted['Page_ID']
    df_converted['LR'] = ""  # プレースホルダー
    
    # カラムの順序を調整
    columns_order = ['Page_ID', 'Excel_ID', 'class', 'page-ID', 'LR', 'comment']
    df_converted = df_converted.reindex(columns=columns_order)
    
    print(f"変換後: {len(df_converted)} rows, {len(df_converted.columns)} columns")
    print("カラム名:", list(df_converted.columns))
    
    return df_converted

def compare_with_ocr_data():
    """
    ExcelデータとOCRデータ（既存CSV）を比較して差分を検出
    """
    print("\n=== OCRデータとの比較 ===")
    
    # 既存のCSVファイルを読み込み
    try:
        df_before_ocr = pd.read_csv("before.csv")
        df_after_ocr = pd.read_csv("after.csv")
        df_comments_ocr = pd.read_csv("comment.csv")
        
        print(f"OCR授業前データ: {len(df_before_ocr)} rows")
        print(f"OCR授業後データ: {len(df_after_ocr)} rows")
        print(f"OCR感想文データ: {len(df_comments_ocr)} rows")
        
    except Exception as e:
        print(f"Error loading OCR CSV files: {e}")
        return
    
    # Excelデータを読み込み
    try:
        df_before_excel = pd.read_csv("validation_data/before_excel.csv")
        df_after_excel = pd.read_csv("validation_data/after_excel.csv")
        df_comments_excel = pd.read_csv("validation_data/comments_excel.csv")
        
        print(f"Excel授業前データ: {len(df_before_excel)} rows")
        print(f"Excel授業後データ: {len(df_after_excel)} rows")
        print(f"Excel感想文データ: {len(df_comments_excel)} rows")
        
    except Exception as e:
        print(f"Error loading Excel CSV files: {e}")
        print("Please run convert_excel_to_csv() first")
        return
    
    # データ比較の実行
    print("\n=== データ比較の実行 ===")
    
    # 1. Page_IDの対応関係を構築
    id_mapping = create_id_mapping(df_before_excel, df_before_ocr)
    
    # 2. 授業前データの比較
    before_differences = compare_before_data(df_before_excel, df_before_ocr, id_mapping)
    
    # 3. 授業後データの比較
    after_differences = compare_after_data(df_after_excel, df_after_ocr, id_mapping)
    
    # 4. 感想文データの比較
    comments_differences = compare_comments_data(df_comments_excel, df_comments_ocr, id_mapping)
    
    # 5. 差分レポートの生成
    generate_difference_report(before_differences, after_differences, comments_differences)
    
    return before_differences, after_differences, comments_differences

def create_id_mapping(df_excel, df_ocr):
    """
    ExcelのIDとOCRのPage_IDの対応関係を作成
    """
    print("Creating ID mapping...")
    
    # 簡単なマッピング: クラス情報を使用してマッチング
    id_mapping = {}
    
    for excel_id in df_excel['Excel_ID'].unique():
        excel_class = df_excel[df_excel['Excel_ID'] == excel_id]['class'].iloc[0]
        page_id = f"PAGE_{excel_id:03d}"  # 仮のマッピング
        id_mapping[excel_id] = page_id
    
    print(f"Created mapping for {len(id_mapping)} IDs")
    return id_mapping

def compare_before_data(df_excel, df_ocr, id_mapping):
    """授業前データの比較"""
    print("\n--- 授業前データの比較 ---")
    differences = []
    
    # ブール値カラムの比較
    bool_columns = [
        'Q1_Saltwater_Response', 'Q1_Sugarwater_Response', 'Q1_Muddywater_Response',
        'Q1_Ink_Response', 'Q1_MisoSoup_Response', 'Q1_SoySauce_Response',
        'Q3_TeaLeavesDissolve', 'Q3_TeaComponentsDissolve'
    ]
    
    for excel_id in df_excel['Excel_ID']:
        if excel_id in id_mapping:
            page_id = id_mapping[excel_id]
            
            # OCRデータから対応する行を検索
            ocr_row = df_ocr[df_ocr['Page_ID'] == page_id]
            if len(ocr_row) == 0:
                continue
                
            excel_row = df_excel[df_excel['Excel_ID'] == excel_id].iloc[0]
            ocr_row = ocr_row.iloc[0]
            
            # 各カラムを比較
            for col in bool_columns:
                if col in df_excel.columns and col in df_ocr.columns:
                    excel_val = excel_row[col]
                    ocr_val = ocr_row[col]
                    
                    if pd.notna(excel_val) and pd.notna(ocr_val) and excel_val != ocr_val:
                        differences.append({
                            'Dataset': 'Before',
                            'Page_ID': page_id,
                            'Excel_ID': excel_id,
                            'Column': col,
                            'Excel_Value': excel_val,
                            'OCR_Value': ocr_val,
                            'Error_Type': 'Boolean_Mismatch'
                        })
    
    print(f"Found {len(differences)} differences in before data")
    return differences

def compare_after_data(df_excel, df_ocr, id_mapping):
    """授業後データの比較"""
    print("\n--- 授業後データの比較 ---")
    differences = []
    
    # ブール値カラムの比較
    bool_columns = [
        'Q1_Saltwater', 'Q1_Sugarwater', 'Q1_Muddywater',
        'Q1_Ink', 'Q1_MisoSoup', 'Q1_SoySauce',
        'Q3_TeaLeaves_DissolveInWater', 'Q3_TeaComponents_DissolveInWater'
    ]
    
    # 評価カラムの比較
    rating_columns = [
        'Q4_ExperimentInterestRating', 'Q5_NewLearningsRating', 'Q6_DissolvingUnderstandingRating'
    ]
    
    for excel_id in df_excel['Excel_ID']:
        if excel_id in id_mapping:
            page_id = id_mapping[excel_id]
            
            # OCRデータから対応する行を検索
            ocr_row = df_ocr[df_ocr['Page_ID'] == page_id]
            if len(ocr_row) == 0:
                continue
                
            excel_row = df_excel[df_excel['Excel_ID'] == excel_id].iloc[0]
            ocr_row = ocr_row.iloc[0]
            
            # ブール値カラムの比較
            for col in bool_columns:
                if col in df_excel.columns and col in df_ocr.columns:
                    excel_val = excel_row[col]
                    ocr_val = ocr_row[col]
                    
                    if pd.notna(excel_val) and pd.notna(ocr_val) and excel_val != ocr_val:
                        differences.append({
                            'Dataset': 'After',
                            'Page_ID': page_id,
                            'Excel_ID': excel_id,
                            'Column': col,
                            'Excel_Value': excel_val,
                            'OCR_Value': ocr_val,
                            'Error_Type': 'Boolean_Mismatch'
                        })
            
            # 評価カラムの比較
            for col in rating_columns:
                if col in df_excel.columns and col in df_ocr.columns:
                    excel_val = excel_row[col]
                    ocr_val = ocr_row[col]
                    
                    if pd.notna(excel_val) and pd.notna(ocr_val) and excel_val != ocr_val:
                        differences.append({
                            'Dataset': 'After',
                            'Page_ID': page_id,
                            'Excel_ID': excel_id,
                            'Column': col,
                            'Excel_Value': excel_val,
                            'OCR_Value': ocr_val,
                            'Error_Type': 'Rating_Mismatch'
                        })
    
    print(f"Found {len(differences)} differences in after data")
    return differences

def compare_comments_data(df_excel, df_ocr, id_mapping):
    """感想文データの比較"""
    print("\n--- 感想文データの比較 ---")
    differences = []
    
    # テキストの類似性比較（簡易版）
    for excel_id in df_excel['Excel_ID']:
        if excel_id in id_mapping:
            page_id = id_mapping[excel_id]
            
            # OCRデータから対応する行を検索
            ocr_row = df_ocr[df_ocr['page-ID'] == page_id]
            if len(ocr_row) == 0:
                continue
                
            excel_row = df_excel[df_excel['Excel_ID'] == excel_id].iloc[0]
            ocr_row = ocr_row.iloc[0]
            
            excel_comment = str(excel_row['comment']).strip()
            ocr_comment = str(ocr_row['comment']).strip()
            
            # 文字数の大きな差がある場合に記録
            if abs(len(excel_comment) - len(ocr_comment)) > 5:
                differences.append({
                    'Dataset': 'Comments',
                    'Page_ID': page_id,
                    'Excel_ID': excel_id,
                    'Column': 'comment',
                    'Excel_Value': excel_comment,
                    'OCR_Value': ocr_comment,
                    'Error_Type': 'Text_Length_Difference'
                })
    
    print(f"Found {len(differences)} differences in comments data")
    return differences

def generate_difference_report(before_diffs, after_diffs, comments_diffs):
    """差分レポートを生成"""
    print("\n=== 差分レポート生成 ===")
    
    all_differences = before_diffs + after_diffs + comments_diffs
    
    if len(all_differences) == 0:
        print("No differences found!")
        return
    
    # DataFrameに変換
    df_differences = pd.DataFrame(all_differences)
    
    # レポートファイルとして保存
    report_file = "ocr_validation_report.csv"
    df_differences.to_csv(report_file, index=False, encoding='utf-8')
    
    print(f"差分レポート保存: {report_file}")
    print(f"総差分数: {len(all_differences)}")
    
    # エラータイプ別の集計
    error_summary = df_differences['Error_Type'].value_counts()
    print("\nエラータイプ別集計:")
    for error_type, count in error_summary.items():
        print(f"  {error_type}: {count}件")
    
    # データセット別の集計
    dataset_summary = df_differences['Dataset'].value_counts()
    print("\nデータセット別集計:")
    for dataset, count in dataset_summary.items():
        print(f"  {dataset}: {count}件")
    
    return df_differences

if __name__ == "__main__":
    # 1. ExcelファイルをCSVに変換
    print("=== Step 1: Converting Excel to CSV ===")
    convert_excel_to_csv()
    
    # 2. OCRデータとの比較
    print("\n=== Step 2: Comparing with OCR data ===")
    compare_with_ocr_data()