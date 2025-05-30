#!/usr/bin/env python3
"""
Excel完全準拠CSV生成スクリプト
手動入力Excelデータを100%正確に反映したCSVファイルを生成
効率的な作業フローでExcel→中間CSV→最終CSVの順で処理
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

class ExcelCompliantCSVGenerator:
    def __init__(self):
        self.excel_file = "refference/250226アンケートデータ/250226アンケートデータ.xlsx"
        self.current_before_csv = "before.csv"
        self.current_after_csv = "after.csv"
        self.current_comment_csv = "comment.csv"
        
        # 出力ファイル
        self.output_before = "before_excel_compliant.csv"
        self.output_after = "after_excel_compliant.csv"
        self.output_comment = "comment_excel_compliant.csv"
        self.mapping_file = "excel_csv_mapping.csv"
        
    def step1_analyze_excel_data(self):
        """Step 1: Excelデータの完全調査"""
        print("=== Step 1: Excelデータの完全調査 ===")
        
        # 各シートの読み込み
        df_before_excel = pd.read_excel(self.excel_file, sheet_name="授業前")
        df_after_excel = pd.read_excel(self.excel_file, sheet_name="授業後")
        df_comments_excel = pd.read_excel(self.excel_file, sheet_name="お礼の手紙の記述")
        
        print(f"Excel読み込み結果:")
        print(f"  授業前: {len(df_before_excel)}行")
        print(f"  授業後: {len(df_after_excel)}行")
        print(f"  感想文: {len(df_comments_excel)}行")
        
        # 実際のデータ存在確認
        print("\\n実際のデータ存在確認:")
        
        # 授業前データの実データ数
        before_real_data = self._count_real_data(df_before_excel)
        after_real_data = self._count_real_data(df_after_excel)
        comments_real_data = df_comments_excel.dropna(subset=['記述']).shape[0]
        
        print(f"  授業前実データ: {before_real_data}行")
        print(f"  授業後実データ: {after_real_data}行")
        print(f"  感想文実データ: {comments_real_data}行")
        
        return {
            'before_excel': df_before_excel,
            'after_excel': df_after_excel,
            'comments_excel': df_comments_excel,
            'before_real_count': before_real_data,
            'after_real_count': after_real_data,
            'comments_real_count': comments_real_data
        }
    
    def _count_real_data(self, df):
        """実際のデータ存在数をカウント"""
        # 整理番号とクラスが両方存在する行をカウント
        return df.dropna(subset=['整理番号', 'クラス']).shape[0]
    
    def step2_create_intermediate_csv(self, data):
        """Step 2: Excel→CSV中間ファイル作成"""
        print("\\n=== Step 2: Excel→CSV中間ファイル作成 ===")
        
        # シンプルな変換で中間CSVを作成
        intermediate_dir = Path("intermediate_csv")
        intermediate_dir.mkdir(exist_ok=True)
        
        # 授業前
        before_path = intermediate_dir / "before_raw.csv"
        data['before_excel'].to_csv(before_path, index=False, encoding='utf-8')
        print(f"授業前中間CSV: {before_path}")
        
        # 授業後
        after_path = intermediate_dir / "after_raw.csv"
        data['after_excel'].to_csv(after_path, index=False, encoding='utf-8')
        print(f"授業後中間CSV: {after_path}")
        
        # 感想文
        comments_path = intermediate_dir / "comments_raw.csv"
        data['comments_excel'].to_csv(comments_path, index=False, encoding='utf-8')
        print(f"感想文中間CSV: {comments_path}")
        
        return {
            'before_path': before_path,
            'after_path': after_path,
            'comments_path': comments_path
        }
    
    def step3_analyze_current_csv_structure(self):
        """Step 3: 現在のCSVのPage_ID方式の詳細分析"""
        print("\\n=== Step 3: 現在のCSVのPage_ID方式分析 ===")
        
        # 現在のCSVを読み込み
        curr_before = pd.read_csv(self.current_before_csv)
        curr_after = pd.read_csv(self.current_after_csv)
        curr_comment = pd.read_csv(self.current_comment_csv)
        
        print("現在のCSV構造:")
        print(f"  before.csv: {len(curr_before)}行, カラム: {len(curr_before.columns)}")
        print(f"  after.csv: {len(curr_after)}行, カラム: {len(curr_after.columns)}")
        print(f"  comment.csv: {len(curr_comment)}行, カラム: {len(curr_comment.columns)}")
        
        # Page_ID分析
        print("\\nPage_ID分析:")
        for dataset_name, df in [("before", curr_before), ("after", curr_after)]:
            page_ids = df['Page_ID'].tolist()
            classes = df['class'].tolist()
            
            print(f"  {dataset_name}: Page_ID {min(page_ids)}-{max(page_ids)}")
            
            # クラス別Page_ID分布
            for cls in sorted(set(classes)):
                cls_page_ids = [pid for pid, c in zip(page_ids, classes) if c == cls]
                print(f"    クラス{cls}: Page_ID {min(cls_page_ids)}-{max(cls_page_ids)} ({len(cls_page_ids)}人)")
        
        return {
            'current_before': curr_before,
            'current_after': curr_after,
            'current_comment': curr_comment
        }
    
    def step4_define_complete_mapping(self, current_csv_data):
        """Step 4: ExcelからCSVスキーマへの完全マッピング定義"""
        print("\\n=== Step 4: 完全マッピング定義 ===")
        
        # beforeデータのマッピング
        before_mapping = {
            '整理番号': None,  # Page_IDに変換（特別処理）
            'クラス': 'class',
            'クイズ1（〇が1，×が0）': 'Q1_Saltwater_Response',
            'クイズ2': 'Q1_Sugarwater_Response',
            'クイズ3': 'Q1_Muddywater_Response',
            'クイズ4': 'Q1_Ink_Response',
            'クイズ5': 'Q1_MisoSoup_Response',
            'クイズ6': 'Q1_SoySauce_Response',
            '味噌汁記述': 'Q2_MisoSalty_Reason',
            'お茶クイズ1（いる1，いない0）': 'Q3_TeaLeavesDissolve',
            'お茶クイズ2（いる1，いない0）': 'Q3_TeaComponentsDissolve'
        }
        
        # afterデータのマッピング（beforeに加えて）
        after_mapping = before_mapping.copy()
        after_mapping.update({
            'おもしろさ': 'Q4_ExperimentInterestRating',
            '新発見': 'Q5_NewLearningsRating',
            '理解': 'Q6_DissolvingUnderstandingRating'
        })
        
        # commentsデータのマッピング
        comments_mapping = {
            '整理番号': None,  # 特別処理
            '記述': 'comment'
        }
        
        # CSV専用項目（Excelにない項目）を特定
        before_csv_cols = set(current_csv_data['current_before'].columns)
        after_csv_cols = set(current_csv_data['current_after'].columns)
        comment_csv_cols = set(current_csv_data['current_comment'].columns)
        
        before_mapped = set(before_mapping.values()) - {None}
        after_mapped = set(after_mapping.values()) - {None}
        comments_mapped = set(comments_mapping.values()) - {None}
        
        csv_only_before = before_csv_cols - before_mapped - {'Page_ID'}
        csv_only_after = after_csv_cols - after_mapped - {'Page_ID'}
        csv_only_comments = comment_csv_cols - comments_mapped - {'page-ID'}
        
        print("CSV専用項目（Excelにない項目）:")
        print(f"  before: {csv_only_before}")
        print(f"  after: {csv_only_after}")
        print(f"  comments: {csv_only_comments}")
        
        return {
            'before_mapping': before_mapping,
            'after_mapping': after_mapping,
            'comments_mapping': comments_mapping,
            'csv_only_before': csv_only_before,
            'csv_only_after': csv_only_after,
            'csv_only_comments': csv_only_comments
        }
    
    def step5_select_top_99_data(self, intermediate_paths):
        """Step 5: 上から順に99行までのデータ選択"""
        print("\\n=== Step 5: 上から順に99行データ選択 ===")
        
        # 中間CSVからデータを読み込み
        df_before_raw = pd.read_csv(intermediate_paths['before_path'])
        df_after_raw = pd.read_csv(intermediate_paths['after_path'])
        df_comments_raw = pd.read_csv(intermediate_paths['comments_path'])
        
        # 実データのみフィルタリング（NaNが多い行を除外）
        before_filtered = df_before_raw.dropna(subset=['整理番号', 'クラス']).head(99)
        after_filtered = df_after_raw.dropna(subset=['整理番号', 'クラス']).head(99)
        
        # 感想文は記述がある行のみ
        comments_filtered = df_comments_raw.dropna(subset=['記述'])
        
        print(f"フィルタリング結果:")
        print(f"  授業前: {len(before_filtered)}行（99行まで選択）")
        print(f"  授業後: {len(after_filtered)}行（99行まで選択）")
        print(f"  感想文: {len(comments_filtered)}行（実データのみ）")
        
        return {
            'before_selected': before_filtered,
            'after_selected': after_filtered,
            'comments_selected': comments_filtered
        }
    
    def step6_generate_page_ids(self, selected_data, current_csv_data):
        """Page_ID生成（現在のCSV方式に合わせる）"""
        print("\\n=== Step 6: Page_ID生成 ===")
        
        # 現在のCSVのPage_ID方式を分析
        current_before = current_csv_data['current_before']
        
        # クラス別のPage_ID範囲を取得
        class_page_id_ranges = {}
        for cls in sorted(current_before['class'].unique()):
            cls_data = current_before[current_before['class'] == cls]
            class_page_id_ranges[cls] = {
                'min': cls_data['Page_ID'].min(),
                'max': cls_data['Page_ID'].max(),
                'count': len(cls_data)
            }
        
        print("現在のCSVのクラス別Page_ID範囲:")
        for cls, info in class_page_id_ranges.items():
            print(f"  クラス{cls}: {info['min']}-{info['max']} ({info['count']}人)")
        
        # 選択されたデータにPage_IDを割り当て
        selected_data['before_selected'] = self._assign_page_ids(
            selected_data['before_selected'], class_page_id_ranges)
        selected_data['after_selected'] = self._assign_page_ids(
            selected_data['after_selected'], class_page_id_ranges)
        
        return selected_data
    
    def _assign_page_ids(self, df, class_ranges):
        """Page_IDの割り当て"""
        df = df.copy()
        df['Page_ID'] = None
        
        for cls in sorted(df['クラス'].unique()):
            if pd.isna(cls):
                continue
                
            cls_mask = df['クラス'] == cls
            cls_count = cls_mask.sum()
            
            if cls in class_ranges:
                # 既存の範囲に合わせてPage_IDを生成
                start_id = class_ranges[cls]['min']
                page_ids = list(range(start_id, start_id + cls_count))
            else:
                # 新しいクラスの場合は1から開始
                page_ids = list(range(1, cls_count + 1))
            
            df.loc[cls_mask, 'Page_ID'] = page_ids
        
        return df
    
    def run_complete_generation(self):
        """完全な生成プロセスの実行"""
        print("Excel完全準拠CSV生成を開始します...\\n")
        
        # Step 1: Excelデータ分析
        excel_data = self.step1_analyze_excel_data()
        
        # Step 2: 中間CSV作成
        intermediate_paths = self.step2_create_intermediate_csv(excel_data)
        
        # Step 3: 現在のCSV構造分析
        current_csv_data = self.step3_analyze_current_csv_structure()
        
        # Step 4: マッピング定義
        mapping_data = self.step4_define_complete_mapping(current_csv_data)
        
        # Step 5: データ選択
        selected_data = self.step5_select_top_99_data(intermediate_paths)
        
        # Step 6: Page_ID生成
        final_data = self.step6_generate_page_ids(selected_data, current_csv_data)
        
        return {
            'excel_data': excel_data,
            'mapping_data': mapping_data,
            'final_data': final_data,
            'current_csv_data': current_csv_data
        }

if __name__ == "__main__":
    generator = ExcelCompliantCSVGenerator()
    results = generator.run_complete_generation()
    
    print("\\n=== 生成プロセス完了 ===")
    print("次のステップ: データ変換とCSV出力")