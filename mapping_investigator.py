#!/usr/bin/env python3
"""
IDマッピング詳細調査スクリプト
ExcelとCSVの正確な対応関係を特定
"""

import pandas as pd
import numpy as np

def investigate_id_mapping():
    """IDマッピングの詳細調査"""
    print("=== IDマッピング詳細調査 ===\n")
    
    # 元のExcelファイルを読み込み
    excel_file = "refference/250226アンケートデータ/250226アンケートデータ.xlsx"
    df_before_excel = pd.read_excel(excel_file, sheet_name="授業前")
    df_after_excel = pd.read_excel(excel_file, sheet_name="授業後")
    
    # 修正前のオリジナルCSVファイルを読み込み（バックアップから）
    df_before_csv = pd.read_csv("backup/before.csv_20250530_154506")
    df_after_csv = pd.read_csv("backup/after.csv_20250530_154506")
    
    print(f"データサイズ:")
    print(f"  Excel授業前: {len(df_before_excel)}行 (ID: {df_before_excel['整理番号'].min()}-{df_before_excel['整理番号'].max()})")
    print(f"  CSV授業前: {len(df_before_csv)}行 (Page_ID: {df_before_csv['Page_ID'].min()}-{df_before_csv['Page_ID'].max()})")
    print(f"  Excel授業後: {len(df_after_excel)}行 (ID: {df_after_excel['整理番号'].min()}-{df_after_excel['整理番号'].max()})")
    print(f"  CSV授業後: {len(df_after_csv)}行 (Page_ID: {df_after_csv['Page_ID'].min()}-{df_after_csv['Page_ID'].max()})")
    print()
    
    # クラス別の詳細分析
    analyze_class_patterns(df_before_excel, df_before_csv, df_after_excel, df_after_csv)
    
    # データ内容による対応関係の推定
    find_content_based_mapping(df_before_excel, df_before_csv, df_after_excel, df_after_csv)

def analyze_class_patterns(df_before_excel, df_before_csv, df_after_excel, df_after_csv):
    """クラス別パターンの分析"""
    print("クラス別パターン分析:")
    print("-" * 50)
    
    for cls in [1, 2, 3, 4]:
        print(f"\nクラス {cls}:")
        
        # Excel側の分析
        excel_before_cls = df_before_excel[df_before_excel['クラス'] == cls]
        excel_after_cls = df_after_excel[df_after_excel['クラス'] == cls]
        excel_ids = sorted(excel_before_cls['整理番号'].tolist())
        
        # CSV側の分析
        csv_before_cls = df_before_csv[df_before_csv['class'] == cls]
        csv_after_cls = df_after_csv[df_after_csv['class'] == cls]
        page_ids = sorted(csv_before_cls['Page_ID'].tolist())
        
        print(f"  Excel ID範囲: {excel_ids[:3]}...{excel_ids[-3:]} (計{len(excel_ids)}人)")
        print(f"  CSV Page_ID範囲: {page_ids[:3]}...{page_ids[-3:]} (計{len(page_ids)}人)")
        
        # 人数が一致する場合の詳細確認
        if len(excel_ids) == len(page_ids):
            print("  ✅ 人数一致 - マッピング可能")
            
            # 最初の3人の詳細データを比較
            for i in range(min(3, len(excel_ids))):
                excel_id = excel_ids[i]
                page_id = page_ids[i]
                
                # Excel授業前データ
                excel_row = df_before_excel[df_before_excel['整理番号'] == excel_id].iloc[0]
                quiz1_excel = excel_row['クイズ1（〇が1，×が0）']
                quiz2_excel = excel_row['クイズ2']
                
                # CSV授業前データ
                csv_row = df_before_csv[df_before_csv['Page_ID'] == page_id].iloc[0]
                quiz1_csv = csv_row['Q1_Saltwater_Response']
                quiz2_csv = csv_row['Q1_Sugarwater_Response']
                
                print(f"    ID{excel_id}→Page_ID{page_id}: クイズ1 Excel:{quiz1_excel} CSV:{quiz1_csv}, クイズ2 Excel:{quiz2_excel} CSV:{quiz2_csv}")
        else:
            print("  ❌ 人数不一致")

def find_content_based_mapping(df_before_excel, df_before_csv, df_after_excel, df_after_csv):
    """データ内容に基づく対応関係の発見"""
    print("\n\nデータ内容による対応関係の推定:")
    print("-" * 50)
    
    # 特徴的なパターンを使って対応関係を推定
    mapping_candidates = {}
    
    # クラス1のデータで詳細調査
    excel_cls1 = df_before_excel[df_before_excel['クラス'] == 1].copy()
    csv_cls1 = df_before_csv[df_before_csv['class'] == 1].copy()
    
    print(f"クラス1の詳細対応調査:")
    print(f"  Excel側: {len(excel_cls1)}人")
    print(f"  CSV側: {len(csv_cls1)}人")
    
    # 各Excelの行について、最も類似するCSV行を探す
    for _, excel_row in excel_cls1.iterrows():
        excel_id = excel_row['整理番号']
        excel_pattern = create_pattern(excel_row)
        
        best_match_score = -1
        best_match_page_id = None
        
        for _, csv_row in csv_cls1.iterrows():
            page_id = csv_row['Page_ID']
            csv_pattern = create_csv_pattern(csv_row)
            
            # パターンマッチングスコアを計算
            score = calculate_similarity(excel_pattern, csv_pattern)
            
            if score > best_match_score:
                best_match_score = score
                best_match_page_id = page_id
        
        mapping_candidates[excel_id] = {
            'page_id': best_match_page_id,
            'score': best_match_score,
            'excel_pattern': excel_pattern,
            'csv_pattern': create_csv_pattern(csv_cls1[csv_cls1['Page_ID'] == best_match_page_id].iloc[0])
        }
    
    # 結果表示
    print("\n対応関係の推定結果（クラス1、上位5件）:")
    sorted_mapping = sorted(mapping_candidates.items(), key=lambda x: x[1]['score'], reverse=True)
    
    for excel_id, match_info in sorted_mapping[:5]:
        print(f"  Excel ID {excel_id} → Page_ID {match_info['page_id']} (スコア: {match_info['score']:.3f})")
        print(f"    Excel: {match_info['excel_pattern']}")
        print(f"    CSV:   {match_info['csv_pattern']}")
        print()

def create_pattern(excel_row):
    """Excel行からパターンを作成"""
    pattern = []
    pattern.append(int(excel_row['クイズ1（〇が1，×が0）']) if pd.notna(excel_row['クイズ1（〇が1，×が0）']) else 0)
    pattern.append(int(excel_row['クイズ2']) if pd.notna(excel_row['クイズ2']) else 0)
    pattern.append(int(excel_row['クイズ3']) if pd.notna(excel_row['クイズ3']) else 0)
    pattern.append(int(excel_row['クイズ4']) if pd.notna(excel_row['クイズ4']) else 0)
    pattern.append(int(excel_row['クイズ5']) if pd.notna(excel_row['クイズ5']) else 0)
    pattern.append(int(excel_row['クイズ6']) if pd.notna(excel_row['クイズ6']) else 0)
    return pattern

def create_csv_pattern(csv_row):
    """CSV行からパターンを作成"""
    pattern = []
    pattern.append(1 if csv_row['Q1_Saltwater_Response'] else 0)
    pattern.append(1 if csv_row['Q1_Sugarwater_Response'] else 0)
    pattern.append(1 if csv_row['Q1_Muddywater_Response'] else 0)
    pattern.append(1 if csv_row['Q1_Ink_Response'] else 0)
    pattern.append(1 if csv_row['Q1_MisoSoup_Response'] else 0)
    pattern.append(1 if csv_row['Q1_SoySauce_Response'] else 0)
    return pattern

def calculate_similarity(pattern1, pattern2):
    """2つのパターンの類似度を計算"""
    if len(pattern1) != len(pattern2):
        return 0
    
    matches = sum(1 for a, b in zip(pattern1, pattern2) if a == b)
    return matches / len(pattern1)

def create_correct_mapping():
    """正しいIDマッピングを作成"""
    print("\n\n=== 正しいIDマッピングの作成 ===")
    
    # バックアップからオリジナルデータを読み込み
    df_before_excel = pd.read_excel("refference/250226アンケートデータ/250226アンケートデータ.xlsx", sheet_name="授業前")
    df_before_csv = pd.read_csv("backup/before.csv_20250530_154506")
    
    correct_mapping = {}
    
    # 全クラスについて内容ベースのマッピングを実行
    for cls in [1, 2, 3, 4]:
        excel_cls = df_before_excel[df_before_excel['クラス'] == cls].copy()
        csv_cls = df_before_csv[df_before_csv['class'] == cls].copy()
        
        print(f"\nクラス{cls}のマッピング:")
        
        # 使用済みPage_IDを追跡
        used_page_ids = set()
        
        for _, excel_row in excel_cls.iterrows():
            excel_id = excel_row['整理番号']
            excel_pattern = create_pattern(excel_row)
            
            best_score = -1
            best_page_id = None
            
            for _, csv_row in csv_cls.iterrows():
                page_id = csv_row['Page_ID']
                if page_id in used_page_ids:
                    continue
                
                csv_pattern = create_csv_pattern(csv_row)
                score = calculate_similarity(excel_pattern, csv_pattern)
                
                if score > best_score:
                    best_score = score
                    best_page_id = page_id
            
            if best_page_id is not None:
                correct_mapping[excel_id] = best_page_id
                used_page_ids.add(best_page_id)
                print(f"  ID{excel_id} → Page_ID{best_page_id} (スコア: {best_score:.3f})")
    
    # マッピングをファイルに保存
    mapping_df = pd.DataFrame([
        {'Excel_ID': k, 'Page_ID': v} for k, v in correct_mapping.items()
    ])
    mapping_df.to_csv("correct_id_mapping.csv", index=False)
    print(f"\n正しいIDマッピングを保存しました: correct_id_mapping.csv")
    print(f"マッピング数: {len(correct_mapping)}")
    
    return correct_mapping

if __name__ == "__main__":
    investigate_id_mapping()
    correct_mapping = create_correct_mapping()