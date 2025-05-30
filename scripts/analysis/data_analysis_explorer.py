#!/usr/bin/env python3
"""
データ分析探索スクリプト
ExcelデータとOCRデータの詳細な構造を調査し、適切な比較方法を見つける
"""

import pandas as pd
import numpy as np
from pathlib import Path

def explore_data_structures():
    """データ構造の詳細調査"""
    print("=== データ構造の詳細調査 ===\n")
    
    # 1. OCRデータの調査
    print("1. OCR データの詳細調査")
    print("-" * 40)
    
    # before.csv
    df_before_ocr = pd.read_csv("before.csv")
    print(f"before.csv: {len(df_before_ocr)} rows, {len(df_before_ocr.columns)} columns")
    print("カラム:", list(df_before_ocr.columns))
    print("Page_IDサンプル:", df_before_ocr['Page_ID'].head().tolist())
    print("クラス分布:", df_before_ocr['class'].value_counts().to_dict())
    print()
    
    # after.csv
    df_after_ocr = pd.read_csv("after.csv")
    print(f"after.csv: {len(df_after_ocr)} rows, {len(df_after_ocr.columns)} columns")
    print("カラム:", list(df_after_ocr.columns))
    print("Page_IDサンプル:", df_after_ocr['Page_ID'].head().tolist())
    print("クラス分布:", df_after_ocr['class'].value_counts().to_dict())
    print()
    
    # comment.csv
    df_comments_ocr = pd.read_csv("comment.csv")
    print(f"comment.csv: {len(df_comments_ocr)} rows, {len(df_comments_ocr.columns)} columns")
    print("カラム:", list(df_comments_ocr.columns))
    print("page-IDサンプル:", df_comments_ocr['page-ID'].head().tolist())
    print("クラス分布:", df_comments_ocr['class'].value_counts().to_dict())
    print()
    
    # 2. Excelデータの調査
    print("2. Excel データの詳細調査")
    print("-" * 40)
    
    # Excel before
    df_before_excel = pd.read_csv("validation_data/before_excel.csv")
    print(f"before_excel.csv: {len(df_before_excel)} rows, {len(df_before_excel.columns)} columns")
    print("カラム:", list(df_before_excel.columns))
    print("Excel_IDサンプル:", df_before_excel['Excel_ID'].head().tolist())
    print("クラス分布:", df_before_excel['class'].value_counts().to_dict())
    print()
    
    # Excel after
    df_after_excel = pd.read_csv("validation_data/after_excel.csv")
    print(f"after_excel.csv: {len(df_after_excel)} rows, {len(df_after_excel.columns)} columns")
    print("カラム:", list(df_after_excel.columns))
    print("Excel_IDサンプル:", df_after_excel['Excel_ID'].head().tolist())
    print("クラス分布:", df_after_excel['class'].value_counts().to_dict())
    print()
    
    # Excel comments
    df_comments_excel = pd.read_csv("validation_data/comments_excel.csv")
    print(f"comments_excel.csv: {len(df_comments_excel)} rows, {len(df_comments_excel.columns)} columns")
    print("カラム:", list(df_comments_excel.columns))
    print("Excel_IDサンプル:", df_comments_excel['Excel_ID'].head().tolist())
    print("クラス分布:", df_comments_excel['class'].value_counts().to_dict())
    print()
    
    # 3. データ対応関係の推定
    print("3. データ対応関係の推定")
    print("-" * 40)
    
    # クラス別の詳細分析
    analyze_class_distributions(df_before_ocr, df_before_excel, df_after_ocr, df_after_excel)
    
    # 4. 実際の比較可能性の検証
    print("4. 実際の比較可能性の検証")
    print("-" * 40)
    
    # Page_IDの形式分析
    analyze_page_id_patterns(df_before_ocr, df_after_ocr, df_comments_ocr)
    
    return {
        'before_ocr': df_before_ocr,
        'after_ocr': df_after_ocr,
        'comments_ocr': df_comments_ocr,
        'before_excel': df_before_excel,
        'after_excel': df_after_excel,
        'comments_excel': df_comments_excel
    }

def analyze_class_distributions(df_before_ocr, df_before_excel, df_after_ocr, df_after_excel):
    """クラス別分布の詳細分析"""
    print("クラス別分布の比較:")
    
    # OCRデータのクラス別分布
    ocr_before_dist = df_before_ocr['class'].value_counts().sort_index()
    ocr_after_dist = df_after_ocr['class'].value_counts().sort_index()
    
    # Excelデータのクラス別分布
    excel_before_dist = df_before_excel['class'].value_counts().sort_index()
    excel_after_dist = df_after_excel['class'].value_counts().sort_index()
    
    print("OCR前:", ocr_before_dist.to_dict())
    print("Excel前:", excel_before_dist.to_dict())
    print("OCR後:", ocr_after_dist.to_dict())
    print("Excel後:", excel_after_dist.to_dict())
    print()
    
    # 各クラスのサンプル表示
    for cls in [1, 2, 3, 4]:
        print(f"クラス {cls} のサンプル:")
        
        # OCRデータ
        ocr_sample = df_before_ocr[df_before_ocr['class'] == cls]['Page_ID'].head(3).tolist()
        print(f"  OCR Page_ID: {ocr_sample}")
        
        # Excelデータ
        excel_sample = df_before_excel[df_before_excel['class'] == cls]['Excel_ID'].head(3).tolist()
        print(f"  Excel ID: {excel_sample}")
        print()

def analyze_page_id_patterns(df_before_ocr, df_after_ocr, df_comments_ocr):
    """Page_IDのパターン分析"""
    print("Page_IDパターンの分析:")
    
    # before.csv
    before_ids = df_before_ocr['Page_ID'].tolist()
    print(f"before.csv Page_ID範囲: {min(before_ids)} ~ {max(before_ids)}")
    print(f"before.csv Page_IDサンプル: {before_ids[:10]}")
    
    # after.csv
    after_ids = df_after_ocr['Page_ID'].tolist()
    print(f"after.csv Page_ID範囲: {min(after_ids)} ~ {max(after_ids)}")
    print(f"after.csv Page_IDサンプル: {after_ids[:10]}")
    
    # comment.csv
    comment_ids = df_comments_ocr['page-ID'].tolist()
    print(f"comment.csv page-ID範囲: {min(comment_ids)} ~ {max(comment_ids)}")
    print(f"comment.csv page-IDサンプル: {comment_ids[:10]}")
    
    # ID形式の分析
    print("\nID形式の分析:")
    print("一意ID数 - before:", len(set(before_ids)))
    print("一意ID数 - after:", len(set(after_ids)))
    print("一意ID数 - comments:", len(set(comment_ids)))
    
    # 重複チェック
    common_before_after = set(before_ids) & set(after_ids)
    print(f"before と after の共通ID数: {len(common_before_after)}")
    print()

def advanced_data_comparison():
    """より高度なデータ比較"""
    print("=== 高度なデータ比較 ===\n")
    
    # データロード
    data = explore_data_structures()
    
    # 実際の値比較のためのサンプル抽出
    print("5. 実際の値比較（サンプル）")
    print("-" * 40)
    
    # 最初の数行を比較
    compare_sample_data(data)
    
    # ブール値の分布比較
    print("6. ブール値分布の比較")
    print("-" * 40)
    compare_boolean_distributions(data)

def compare_sample_data(data):
    """サンプルデータの詳細比較"""
    df_before_ocr = data['before_ocr']
    df_before_excel = data['before_excel']
    
    print("最初の5行の比較（授業前データ）:")
    print("\nOCRデータ（before.csv）:")
    ocr_sample = df_before_ocr.head()
    for idx, row in ocr_sample.iterrows():
        print(f"  {row['Page_ID']}: クラス{row['class']}, 食塩水={row.get('Q1_Saltwater_Response', 'N/A')}")
    
    print("\nExcelデータ（before_excel.csv）:")
    excel_sample = df_before_excel.head()
    for idx, row in excel_sample.iterrows():
        print(f"  ID{row['Excel_ID']}: クラス{row['class']}, 食塩水={row.get('Q1_Saltwater_Response', 'N/A')}")
    print()

def compare_boolean_distributions(data):
    """ブール値の分布比較"""
    df_before_ocr = data['before_ocr']
    df_before_excel = data['before_excel']
    
    # 共通カラムを見つける
    bool_columns = [
        'Q1_Saltwater_Response', 'Q1_Sugarwater_Response', 'Q1_Muddywater_Response',
        'Q1_Ink_Response', 'Q1_MisoSoup_Response', 'Q1_SoySauce_Response'
    ]
    
    print("ブール値カラムの分布比較:")
    for col in bool_columns:
        if col in df_before_ocr.columns and col in df_before_excel.columns:
            ocr_dist = df_before_ocr[col].value_counts()
            excel_dist = df_before_excel[col].value_counts()
            
            print(f"\n{col}:")
            print(f"  OCR: {ocr_dist.to_dict()}")
            print(f"  Excel: {excel_dist.to_dict()}")

if __name__ == "__main__":
    advanced_data_comparison()