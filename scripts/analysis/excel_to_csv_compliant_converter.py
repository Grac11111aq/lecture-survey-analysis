#!/usr/bin/env python3
"""
Excel完全準拠CSV最終変換・生成スクリプト
前段階の分析結果を基に、最終的なCSVファイルを生成
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

class ExcelToCSVCompliantConverter:
    def __init__(self):
        # 中間データの読み込み
        self.intermediate_dir = Path("intermediate_csv")
        self.current_before_csv = "before.csv"
        self.current_after_csv = "after.csv"
        self.current_comment_csv = "comment.csv"
        
        # 出力ファイル
        self.output_before = "before_excel_compliant.csv"
        self.output_after = "after_excel_compliant.csv"
        self.output_comment = "comment_excel_compliant.csv"
        
    def load_intermediate_data(self):
        """中間データの読み込み"""
        print("=== 中間データ読み込み ===")
        
        # 中間CSVファイル
        df_before_raw = pd.read_csv(self.intermediate_dir / "before_raw.csv")
        df_after_raw = pd.read_csv(self.intermediate_dir / "after_raw.csv")
        df_comments_raw = pd.read_csv(self.intermediate_dir / "comments_raw.csv")
        
        # 現在のCSVファイル（スキーマ参照用）
        curr_before = pd.read_csv(self.current_before_csv)
        curr_after = pd.read_csv(self.current_after_csv)
        curr_comment = pd.read_csv(self.current_comment_csv)
        
        # 実データのみフィルタリング（上から99行）
        before_filtered = df_before_raw.dropna(subset=['整理番号', 'クラス']).head(99)
        after_filtered = df_after_raw.dropna(subset=['整理番号', 'クラス']).head(99)
        comments_filtered = df_comments_raw.dropna(subset=['記述'])
        
        print(f"処理対象データ:")
        print(f"  授業前: {len(before_filtered)}行")
        print(f"  授業後: {len(after_filtered)}行")
        print(f"  感想文: {len(comments_filtered)}行")
        
        return {
            'before_excel': before_filtered,
            'after_excel': after_filtered,
            'comments_excel': comments_filtered,
            'current_before': curr_before,
            'current_after': curr_after,
            'current_comment': curr_comment
        }
    
    def convert_before_data(self, data):
        """授業前データの変換"""
        print("\\n=== 授業前データ変換 ===")
        
        df_excel = data['before_excel']
        df_current = data['current_before']
        
        # 新しいDataFrameを作成（現在のCSVスキーマに合わせる）
        df_new = pd.DataFrame()
        
        # Page_IDの生成（クラス内循環方式）
        df_new['Page_ID'] = self._generate_page_ids(df_excel)
        
        # 基本マッピング
        mapping = {
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
        
        # データ変換
        for excel_col, csv_col in mapping.items():
            if excel_col in df_excel.columns:
                if 'クイズ' in excel_col and csv_col != 'Q2_MisoSalty_Reason':
                    # ブール値変換: 1.0→True, 0.0→False
                    df_new[csv_col] = df_excel[excel_col].apply(
                        lambda x: bool(int(x)) if pd.notna(x) else False
                    )
                else:
                    # そのまま転写
                    df_new[csv_col] = df_excel[excel_col]
        
        # カラム順序を現在のCSVに合わせる
        df_new = df_new.reindex(columns=df_current.columns, fill_value="")
        
        print(f"変換完了: {len(df_new)}行, {len(df_new.columns)}カラム")
        return df_new
    
    def convert_after_data(self, data):
        """授業後データの変換"""
        print("\\n=== 授業後データ変換 ===")
        
        df_excel = data['after_excel']
        df_current = data['current_after']
        
        # 新しいDataFrameを作成
        df_new = pd.DataFrame()
        
        # Page_IDの生成
        df_new['Page_ID'] = self._generate_page_ids(df_excel)
        
        # 基本マッピング（授業前＋授業後専用項目）
        mapping = {
            'クラス': 'class',
            'クイズ1（〇が1，×が0）': 'Q1_Saltwater',
            'クイズ2': 'Q1_Sugarwater',
            'クイズ3': 'Q1_Muddywater',
            'クイズ4': 'Q1_Ink',
            'クイズ5': 'Q1_MisoSoup',
            'クイズ6': 'Q1_SoySauce',
            '味噌汁記述': 'Q2_MisoSaltyReason',
            'お茶クイズ1（いる1，いない0）': 'Q3_TeaLeaves_DissolveInWater',
            'お茶クイズ2（いる1，いない0）': 'Q3_TeaComponents_DissolveInWater',
            'おもしろさ': 'Q4_ExperimentInterestRating',
            '新発見': 'Q5_NewLearningsRating',
            '理解': 'Q6_DissolvingUnderstandingRating'
        }
        
        # データ変換
        for excel_col, csv_col in mapping.items():
            if excel_col in df_excel.columns:
                if 'クイズ' in excel_col and 'お茶' not in excel_col:
                    # Q1系ブール値変換
                    df_new[csv_col] = df_excel[excel_col].apply(
                        lambda x: bool(int(x)) if pd.notna(x) else False
                    )
                elif 'お茶' in excel_col:
                    # Q3系ブール値変換
                    df_new[csv_col] = df_excel[excel_col].apply(
                        lambda x: bool(int(x)) if pd.notna(x) else False
                    )
                elif excel_col in ['おもしろさ', '新発見', '理解']:
                    # 評価値は数値のまま
                    df_new[csv_col] = df_excel[excel_col].apply(
                        lambda x: int(x) if pd.notna(x) else 1
                    )
                else:
                    # その他（テキスト項目）
                    df_new[csv_col] = df_excel[excel_col]
        
        # CSV専用項目のデフォルト値設定
        csv_only_columns = [
            'Q4_ExperimentInterestComment',
            'Q6_DissolvingUnderstandingComment', 
            'GeneralPageComments'
        ]
        
        for col in csv_only_columns:
            df_new[col] = ""  # 空文字列をデフォルト
        
        # カラム順序を現在のCSVに合わせる
        df_new = df_new.reindex(columns=df_current.columns, fill_value="")
        
        print(f"変換完了: {len(df_new)}行, {len(df_new.columns)}カラム")
        return df_new
    
    def convert_comment_data(self, data):
        """感想文データの変換"""
        print("\\n=== 感想文データ変換 ===")
        
        df_excel = data['comments_excel']
        df_current = data['current_comment']
        
        # 新しいDataFrameを作成
        df_new = pd.DataFrame()
        
        # 基本マッピング
        df_new['page-ID'] = self._generate_comment_page_ids(df_excel)
        df_new['comment'] = df_excel['記述']
        
        # CSV専用項目
        df_new['class'] = 3  # 感想文は主にクラス3と仮定
        df_new['LR'] = ""    # 空文字列をデフォルト
        
        # カラム順序を現在のCSVに合わせる
        df_new = df_new.reindex(columns=df_current.columns, fill_value="")
        
        print(f"変換完了: {len(df_new)}行, {len(df_new.columns)}カラム")
        return df_new
    
    def _generate_page_ids(self, df_excel):
        """クラス内循環式Page_ID生成"""
        page_ids = []
        
        for _, row in df_excel.iterrows():
            cls = row['クラス']
            if pd.isna(cls):
                page_ids.append(1)
                continue
                
            # 現在のクラス内での順序を取得
            cls_rows = df_excel[df_excel['クラス'] == cls]
            cls_index = cls_rows.index.get_loc(row.name)
            
            # Page_IDは1から開始
            page_id = cls_index + 1
            page_ids.append(page_id)
        
        return page_ids
    
    def _generate_comment_page_ids(self, df_excel):
        """感想文用Page_ID生成"""
        # 感想文の場合は、整理番号から推定
        page_ids = []
        
        for _, row in df_excel.iterrows():
            excel_id = row['整理番号']
            if pd.notna(excel_id):
                # 簡単な変換ルール（調整可能）
                page_id = int(excel_id) if excel_id <= 26 else (int(excel_id) % 26) + 1
            else:
                page_id = 1
            page_ids.append(page_id)
        
        return page_ids
    
    def save_compliant_csvs(self, converted_data):
        """完全準拠CSVの保存"""
        print("\\n=== 完全準拠CSV保存 ===")
        
        # 保存
        converted_data['before'].to_csv(self.output_before, index=False, encoding='utf-8')
        converted_data['after'].to_csv(self.output_after, index=False, encoding='utf-8')
        converted_data['comment'].to_csv(self.output_comment, index=False, encoding='utf-8')
        
        print(f"保存完了:")
        print(f"  {self.output_before}: {len(converted_data['before'])}行")
        print(f"  {self.output_after}: {len(converted_data['after'])}行")
        print(f"  {self.output_comment}: {len(converted_data['comment'])}行")
        
        return {
            'before_file': self.output_before,
            'after_file': self.output_after,
            'comment_file': self.output_comment
        }
    
    def run_complete_conversion(self):
        """完全な変換プロセスの実行"""
        print("Excel完全準拠CSV変換を開始します...\\n")
        
        # データ読み込み
        data = self.load_intermediate_data()
        
        # 各データセットの変換
        converted_data = {
            'before': self.convert_before_data(data),
            'after': self.convert_after_data(data),
            'comment': self.convert_comment_data(data)
        }
        
        # CSVファイル保存
        output_files = self.save_compliant_csvs(converted_data)
        
        print("\\n=== 変換プロセス完了 ===")
        return {
            'converted_data': converted_data,
            'output_files': output_files
        }

if __name__ == "__main__":
    converter = ExcelToCSVCompliantConverter()
    results = converter.run_complete_conversion()
    
    print("\\nExcel完全準拠CSVファイルが生成されました！")