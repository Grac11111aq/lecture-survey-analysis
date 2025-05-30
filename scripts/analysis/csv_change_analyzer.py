#!/usr/bin/env python3
"""
CSV変更分析スクリプト
オリジナルCSVと現在のCSVの差分を詳細分析し、
Excel非対応部分の変更を特定する
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class CSVChangeAnalyzer:
    def __init__(self):
        # データソースの定義
        self.original_before = "backup/before.csv_20250530_154506"
        self.original_after = "backup/after.csv_20250530_154506"
        self.current_before = "before.csv"
        self.current_after = "after.csv"
        self.id_mapping_file = "correct_id_mapping.csv"
        
        # データを読み込み
        self.load_all_data()
        
    def load_all_data(self):
        """全データの読み込み"""
        print("=== データ読み込み ===")
        
        # オリジナルCSV
        self.orig_before = pd.read_csv(self.original_before)
        self.orig_after = pd.read_csv(self.original_after)
        
        # 現在のCSV
        self.curr_before = pd.read_csv(self.current_before)
        self.curr_after = pd.read_csv(self.current_after)
        
        # IDマッピング
        self.id_mapping = pd.read_csv(self.id_mapping_file)
        self.excel_corresponding_page_ids = set(self.id_mapping['Page_ID'].tolist())
        
        print(f"オリジナル - before: {len(self.orig_before)}行, after: {len(self.orig_after)}行")
        print(f"現在 - before: {len(self.curr_before)}行, after: {len(self.curr_after)}行")
        print(f"Excel対応Page_ID: {len(self.excel_corresponding_page_ids)}個")
        print()

    def analyze_data_structure(self):
        """Phase 1: データ構造比較"""
        print("=== Phase 1: データ構造比較 ===")
        
        structure_analysis = {
            'before': self._compare_structure(self.orig_before, self.curr_before, "before"),
            'after': self._compare_structure(self.orig_after, self.curr_after, "after")
        }
        
        return structure_analysis
    
    def _compare_structure(self, orig_df, curr_df, dataset_name):
        """データ構造の比較"""
        print(f"\n--- {dataset_name}データの構造比較 ---")
        
        analysis = {
            'rows': {'original': len(orig_df), 'current': len(curr_df)},
            'columns': {'original': len(orig_df.columns), 'current': len(curr_df.columns)},
            'dtypes_changes': {},
            'nan_changes': {}
        }
        
        # 行数・カラム数の変化
        print(f"行数: {analysis['rows']['original']} → {analysis['rows']['current']}")
        print(f"カラム数: {analysis['columns']['original']} → {analysis['columns']['current']}")
        
        # データ型の変化
        orig_dtypes = orig_df.dtypes.to_dict()
        curr_dtypes = curr_df.dtypes.to_dict()
        
        for col in orig_df.columns:
            if col in curr_df.columns:
                if str(orig_dtypes[col]) != str(curr_dtypes[col]):
                    analysis['dtypes_changes'][col] = {
                        'original': str(orig_dtypes[col]),
                        'current': str(curr_dtypes[col])
                    }
                    print(f"データ型変化 {col}: {orig_dtypes[col]} → {curr_dtypes[col]}")
        
        # NaN値の変化
        orig_nan = orig_df.isnull().sum().to_dict()
        curr_nan = curr_df.isnull().sum().to_dict()
        
        for col in orig_df.columns:
            if col in curr_df.columns:
                if orig_nan[col] != curr_nan[col]:
                    analysis['nan_changes'][col] = {
                        'original': orig_nan[col],
                        'current': curr_nan[col]
                    }
                    print(f"NaN値変化 {col}: {orig_nan[col]} → {curr_nan[col]}")
        
        return analysis

    def identify_non_excel_rows(self):
        """Phase 2: Excel非対応部分の特定"""
        print("\n=== Phase 2: Excel非対応部分の特定 ===")
        
        # Excel非対応Page_IDを特定
        all_page_ids_before = set(self.curr_before['Page_ID'].tolist())
        all_page_ids_after = set(self.curr_after['Page_ID'].tolist())
        
        non_excel_before = all_page_ids_before - self.excel_corresponding_page_ids
        non_excel_after = all_page_ids_after - self.excel_corresponding_page_ids
        
        print(f"授業前データ:")
        print(f"  全Page_ID: {len(all_page_ids_before)}個")
        print(f"  Excel対応: {len(self.excel_corresponding_page_ids & all_page_ids_before)}個")
        print(f"  Excel非対応: {len(non_excel_before)}個")
        
        print(f"授業後データ:")
        print(f"  全Page_ID: {len(all_page_ids_after)}個")
        print(f"  Excel対応: {len(self.excel_corresponding_page_ids & all_page_ids_after)}個")
        print(f"  Excel非対応: {len(non_excel_after)}個")
        
        # Excel非対応行のDataFrameを作成
        self.non_excel_orig_before = self.orig_before[~self.orig_before['Page_ID'].isin(self.excel_corresponding_page_ids)]
        self.non_excel_curr_before = self.curr_before[~self.curr_before['Page_ID'].isin(self.excel_corresponding_page_ids)]
        self.non_excel_orig_after = self.orig_after[~self.orig_after['Page_ID'].isin(self.excel_corresponding_page_ids)]
        self.non_excel_curr_after = self.curr_after[~self.curr_after['Page_ID'].isin(self.excel_corresponding_page_ids)]
        
        print(f"\nExcel非対応行のデータフレーム作成完了")
        print(f"  before: オリジナル{len(self.non_excel_orig_before)}行, 現在{len(self.non_excel_curr_before)}行")
        print(f"  after: オリジナル{len(self.non_excel_orig_after)}行, 現在{len(self.non_excel_curr_after)}行")
        
        return {
            'non_excel_page_ids_before': non_excel_before,
            'non_excel_page_ids_after': non_excel_after
        }

    def analyze_non_excel_changes(self):
        """Phase 3: Excel非対応行での変更内容の詳細分析"""
        print("\n=== Phase 3: Excel非対応行での変更内容分析 ===")
        
        changes_analysis = {
            'before': self._analyze_changes(
                self.non_excel_orig_before, 
                self.non_excel_curr_before, 
                "before"
            ),
            'after': self._analyze_changes(
                self.non_excel_orig_after, 
                self.non_excel_curr_after, 
                "after"
            )
        }
        
        return changes_analysis
    
    def _analyze_changes(self, orig_df, curr_df, dataset_name):
        """個別データセットでの変更分析"""
        print(f"\n--- {dataset_name}データのExcel非対応行変更分析 ---")
        
        if len(orig_df) == 0 and len(curr_df) == 0:
            print("Excel非対応行がありません")
            return {'total_changes': 0, 'changes_by_column': {}, 'changes_detail': []}
        
        changes = []
        changes_by_column = {}
        
        # Page_IDで突合して比較
        for page_id in orig_df['Page_ID'].unique():
            orig_row = orig_df[orig_df['Page_ID'] == page_id]
            curr_row = curr_df[curr_df['Page_ID'] == page_id]
            
            if len(orig_row) == 0 or len(curr_row) == 0:
                continue
                
            orig_row = orig_row.iloc[0]
            curr_row = curr_row.iloc[0]
            
            # 各カラムを比較
            for col in orig_df.columns:
                if col == 'Page_ID':
                    continue
                    
                orig_val = orig_row[col]
                curr_val = curr_row[col]
                
                # 値の比較（NaN同士は同じとみなす）
                if pd.isna(orig_val) and pd.isna(curr_val):
                    continue
                elif orig_val != curr_val:
                    change_detail = {
                        'page_id': page_id,
                        'column': col,
                        'original': orig_val,
                        'current': curr_val,
                        'change_type': self._classify_change(orig_val, curr_val)
                    }
                    changes.append(change_detail)
                    
                    if col not in changes_by_column:
                        changes_by_column[col] = 0
                    changes_by_column[col] += 1
        
        print(f"総変更数: {len(changes)}")
        print("カラム別変更数:")
        for col, count in sorted(changes_by_column.items(), key=lambda x: x[1], reverse=True):
            print(f"  {col}: {count}件")
        
        # 最初の5件の詳細を表示
        if changes:
            print("\n変更例（最初の5件）:")
            for change in changes[:5]:
                print(f"  Page_ID {change['page_id']}: {change['column']} = '{change['original']}' → '{change['current']}'")
        
        return {
            'total_changes': len(changes),
            'changes_by_column': changes_by_column,
            'changes_detail': changes
        }
    
    def _classify_change(self, orig_val, curr_val):
        """変更タイプの分類"""
        if pd.isna(orig_val) and not pd.isna(curr_val):
            return "NaN_to_Value"
        elif not pd.isna(orig_val) and pd.isna(curr_val):
            return "Value_to_NaN"
        elif isinstance(orig_val, bool) and isinstance(curr_val, bool):
            return "Boolean_Change"
        elif isinstance(orig_val, (int, float)) and isinstance(curr_val, (int, float)):
            return "Numeric_Change"
        else:
            return "Text_Change"

    def analyze_by_column_type(self):
        """Phase 4: カラム別変更分析"""
        print("\n=== Phase 4: カラム別変更分析 ===")
        
        column_analysis = {
            'before': self._analyze_by_column_type(
                self.non_excel_orig_before, 
                self.non_excel_curr_before, 
                "before"
            ),
            'after': self._analyze_by_column_type(
                self.non_excel_orig_after, 
                self.non_excel_curr_after, 
                "after"
            )
        }
        
        return column_analysis
    
    def _analyze_by_column_type(self, orig_df, curr_df, dataset_name):
        """カラムタイプ別の分析"""
        print(f"\n--- {dataset_name}データのカラムタイプ別分析 ---")
        
        if len(orig_df) == 0:
            return {}
        
        # カラムをタイプ別に分類
        bool_columns = []
        text_columns = []
        numeric_columns = []
        
        for col in orig_df.columns:
            if col == 'Page_ID' or col == 'class':
                continue
                
            dtype = str(orig_df[col].dtype)
            if 'bool' in dtype:
                bool_columns.append(col)
            elif 'object' in dtype:
                text_columns.append(col)
            elif 'int' in dtype or 'float' in dtype:
                numeric_columns.append(col)
        
        print(f"ブール値カラム: {len(bool_columns)}個")
        print(f"テキストカラム: {len(text_columns)}個")
        print(f"数値カラム: {len(numeric_columns)}個")
        
        # 各タイプでの変更を分析
        analysis = {}
        
        if bool_columns:
            analysis['boolean'] = self._analyze_boolean_changes(orig_df, curr_df, bool_columns)
        if text_columns:
            analysis['text'] = self._analyze_text_changes(orig_df, curr_df, text_columns)
        if numeric_columns:
            analysis['numeric'] = self._analyze_numeric_changes(orig_df, curr_df, numeric_columns)
        
        return analysis
    
    def _analyze_boolean_changes(self, orig_df, curr_df, bool_columns):
        """ブール値カラムの変更分析"""
        print("\nブール値カラムの分析:")
        
        changes = 0
        for col in bool_columns:
            if col in curr_df.columns:
                for page_id in orig_df['Page_ID'].unique():
                    orig_row = orig_df[orig_df['Page_ID'] == page_id]
                    curr_row = curr_df[curr_df['Page_ID'] == page_id]
                    
                    if len(orig_row) > 0 and len(curr_row) > 0:
                        orig_val = orig_row.iloc[0][col]
                        curr_val = curr_row.iloc[0][col]
                        
                        if orig_val != curr_val:
                            changes += 1
        
        print(f"  ブール値変更: {changes}件")
        return {'changes': changes, 'columns': bool_columns}
    
    def _analyze_text_changes(self, orig_df, curr_df, text_columns):
        """テキストカラムの変更分析"""
        print("\nテキストカラムの分析:")
        
        changes = 0
        for col in text_columns:
            if col in curr_df.columns:
                for page_id in orig_df['Page_ID'].unique():
                    orig_row = orig_df[orig_df['Page_ID'] == page_id]
                    curr_row = curr_df[curr_df['Page_ID'] == page_id]
                    
                    if len(orig_row) > 0 and len(curr_row) > 0:
                        orig_val = orig_row.iloc[0][col]
                        curr_val = curr_row.iloc[0][col]
                        
                        if str(orig_val) != str(curr_val):
                            changes += 1
        
        print(f"  テキスト変更: {changes}件")
        return {'changes': changes, 'columns': text_columns}
    
    def _analyze_numeric_changes(self, orig_df, curr_df, numeric_columns):
        """数値カラムの変更分析"""
        print("\n数値カラムの分析:")
        
        changes = 0
        for col in numeric_columns:
            if col in curr_df.columns:
                for page_id in orig_df['Page_ID'].unique():
                    orig_row = orig_df[orig_df['Page_ID'] == page_id]
                    curr_row = curr_df[curr_df['Page_ID'] == page_id]
                    
                    if len(orig_row) > 0 and len(curr_row) > 0:
                        orig_val = orig_row.iloc[0][col]
                        curr_val = curr_row.iloc[0][col]
                        
                        if orig_val != curr_val:
                            changes += 1
        
        print(f"  数値変更: {changes}件")
        return {'changes': changes, 'columns': numeric_columns}

    def run_full_analysis(self):
        """全分析の実行"""
        print("CSV変更分析を開始します...\n")
        
        # Phase 1: データ構造比較
        structure_analysis = self.analyze_data_structure()
        
        # Phase 2: Excel非対応部分の特定
        non_excel_identification = self.identify_non_excel_rows()
        
        # Phase 3: Excel非対応行での変更分析
        non_excel_changes = self.analyze_non_excel_changes()
        
        # Phase 4: カラム別変更分析
        column_analysis = self.analyze_by_column_type()
        
        # 結果をまとめる
        full_analysis = {
            'timestamp': datetime.now().isoformat(),
            'structure_analysis': structure_analysis,
            'non_excel_identification': non_excel_identification,
            'non_excel_changes': non_excel_changes,
            'column_analysis': column_analysis
        }
        
        return full_analysis

if __name__ == "__main__":
    analyzer = CSVChangeAnalyzer()
    results = analyzer.run_full_analysis()
    
    # 結果をJSONファイルに保存
    with open("csv_change_analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print("\n=== 分析完了 ===")
    print("結果は csv_change_analysis_results.json に保存されました")