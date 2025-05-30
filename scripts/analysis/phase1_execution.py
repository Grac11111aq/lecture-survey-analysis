#!/usr/bin/env python3
"""
Phase 1: データ品質確認と基礎統計の実行スクリプト
ANALYSIS_PLAN.md の Phase 1 に従い実施
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class Phase1Analyzer:
    def __init__(self, data_dir='data/analysis/'):
        self.data_dir = data_dir
        self.before_df = None
        self.after_df = None
        self.comment_df = None
        
    def load_data(self):
        """データ読み込み"""
        print("=== データ読み込み開始 ===")
        
        try:
            self.before_df = pd.read_csv(os.path.join(self.data_dir, 'before_excel_compliant.csv'))
            self.after_df = pd.read_csv(os.path.join(self.data_dir, 'after_excel_compliant.csv'))
            self.comment_df = pd.read_csv(os.path.join(self.data_dir, 'comment.csv'))
            
            print(f"授業前データ: {self.before_df.shape}")
            print(f"授業後データ: {self.after_df.shape}")
            print(f"感想データ: {self.comment_df.shape}")
            print("✅ データ読み込み完了\n")
            
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            return False
        return True
    
    def analyze_page_id_matching(self):
        """Page_IDマッチング確認"""
        print("=== Page_IDマッチング分析 ===")
        
        before_ids = set(self.before_df['Page_ID'])
        after_ids = set(self.after_df['Page_ID'])
        comment_ids = set(self.comment_df['page-ID']) if 'page-ID' in self.comment_df.columns else set(self.comment_df.get('Page_ID', []))
        
        print(f"授業前データ Page_ID数: {len(before_ids)}")
        print(f"授業後データ Page_ID数: {len(after_ids)}")
        print(f"感想データ Page_ID数: {len(comment_ids)}")
        
        # 共通のPage_ID
        common_ids = before_ids & after_ids
        matching_rate = len(common_ids)/max(len(before_ids), len(after_ids))*100
        print(f"\n前後共通 Page_ID数: {len(common_ids)}")
        print(f"マッチング率: {matching_rate:.1f}%")
        
        # マッチングしないPage_ID
        before_only = before_ids - after_ids
        after_only = after_ids - before_ids
        
        if before_only:
            print(f"授業前のみ: {sorted(list(before_only))}")
        if after_only:
            print(f"授業後のみ: {sorted(list(after_only))}")
        
        # クラス別分布
        print("\n=== クラス別分布 ===")
        print("授業前:")
        print(self.before_df['class'].value_counts().sort_index())
        print("\n授業後:")
        print(self.after_df['class'].value_counts().sort_index())
        print()
        
        return common_ids, matching_rate
    
    def analyze_missing_values(self):
        """欠損値分析"""
        print("=== 欠損値分析 ===")
        
        results = {}
        
        for name, df in [("授業前", self.before_df), ("授業後", self.after_df), ("感想", self.comment_df)]:
            print(f"\n--- {name}データ ---")
            
            missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
            missing_counts = df.isnull().sum()
            
            missing_info = pd.DataFrame({
                '欠損数': missing_counts,
                '欠損率(%)': missing_pct
            })
            
            # 欠損値がある項目のみ表示
            missing_info = missing_info[missing_info['欠損数'] > 0]
            
            if len(missing_info) > 0:
                print(missing_info.sort_values('欠損率(%)', ascending=False))
                
                # 20%以上の欠損がある項目
                high_missing = missing_info[missing_info['欠損率(%)'] > 20]
                if len(high_missing) > 0:
                    print(f"⚠️ 20%以上欠損の項目: {list(high_missing.index)}")
                else:
                    print("✅ 20%以上欠損項目なし")
            else:
                print("✅ 欠損値なし")
            
            results[name] = missing_info
        
        print()
        return results
    
    def analyze_q1_responses(self):
        """Q1項目（水溶液認識）の分析"""
        print("=== Q1項目（水溶液認識）分析 ===")
        
        # Q1項目の特定
        q1_cols_before = [col for col in self.before_df.columns if col.startswith('Q1_')]
        q1_cols_after = [col for col in self.after_df.columns if col.startswith('Q1_')]
        
        print(f"授業前 Q1項目: {q1_cols_before}")
        print(f"授業後 Q1項目: {q1_cols_after}")
        
        # 項目対応関係の確立
        substances = ['Saltwater', 'Sugarwater', 'Muddywater', 'Ink', 'MisoSoup', 'SoySauce']
        correspondence = {}
        
        for substance in substances:
            before_col = None
            after_col = None
            
            for col in q1_cols_before:
                if substance in col:
                    before_col = col
                    break
            
            for col in q1_cols_after:
                if substance in col:
                    after_col = col
                    break
            
            if before_col and after_col:
                correspondence[substance] = {'before': before_col, 'after': after_col}
                print(f"{substance}: {before_col} ↔ {after_col}")
        
        # 正答基準
        correct_answers = {
            'Saltwater': True,
            'Sugarwater': True, 
            'Muddywater': False,
            'Ink': False,
            'MisoSoup': True,
            'SoySauce': True
        }
        
        print("\n=== 正答率分析 ===")
        results = []
        
        for substance, cols in correspondence.items():
            before_col = cols['before']
            after_col = cols['after']
            correct = correct_answers.get(substance)
            
            # 授業前正答率
            before_correct = (self.before_df[before_col] == correct).sum()
            before_total = self.before_df[before_col].notna().sum()
            before_rate = before_correct / before_total * 100 if before_total > 0 else 0
            
            # 授業後正答率
            after_correct = (self.after_df[after_col] == correct).sum()
            after_total = self.after_df[after_col].notna().sum()
            after_rate = after_correct / after_total * 100 if after_total > 0 else 0
            
            change = after_rate - before_rate
            
            results.append({
                '物質': substance,
                '正答': correct,
                '授業前_正答数': before_correct,
                '授業前_総数': before_total,
                '授業前_正答率': round(before_rate, 1),
                '授業後_正答数': after_correct,
                '授業後_総数': after_total,
                '授業後_正答率': round(after_rate, 1),
                '変化': round(change, 1)
            })
        
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        print()
        
        return results_df, correspondence
    
    def calculate_descriptive_stats(self):
        """記述統計量の算出"""
        print("=== 記述統計量算出 ===")
        
        for name, df in [("授業前", self.before_df), ("授業後", self.after_df)]:
            print(f"\n--- {name}データ ---")
            
            # 数値変数の記述統計
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                print("数値変数 基本統計量:")
                stats_df = df[numeric_cols].describe()
                print(stats_df.round(2))
                
                # 歪度・尖度
                print("\n歪度・尖度:")
                skew_kurt = pd.DataFrame({
                    '歪度': df[numeric_cols].skew(),
                    '尖度': df[numeric_cols].kurtosis()
                })
                print(skew_kurt.round(3))
            
            # カテゴリ変数の度数分布
            categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
            categorical_cols = [col for col in categorical_cols if col != 'Page_ID']
            
            if len(categorical_cols) > 0:
                print(f"\nカテゴリ変数度数分布:")
                for col in categorical_cols[:5]:  # 最初の5項目のみ表示
                    print(f"\n{col}:")
                    print(df[col].value_counts(dropna=False))
        
        print()
    
    def analyze_evaluation_items(self):
        """授業後評価項目の分析"""
        print("=== 授業後評価項目分析 ===")
        
        evaluation_items = {
            'Q4_ExperimentInterestRating': '実験への興味度',
            'Q5_NewLearningsRating': '新しい学び',
            'Q6_DissolvingUnderstandingRating': '理解度'
        }
        
        for col, label in evaluation_items.items():
            if col in self.after_df.columns:
                print(f"\n{label} ({col}):")
                counts = self.after_df[col].value_counts().sort_index()
                for val, count in counts.items():
                    pct = count / len(self.after_df) * 100
                    print(f"  {val}: {count}名 ({pct:.1f}%)")
                
                # 基本統計
                if pd.api.types.is_numeric_dtype(self.after_df[col]):
                    mean_val = self.after_df[col].mean()
                    median_val = self.after_df[col].median()
                    std_val = self.after_df[col].std()
                    print(f"  平均: {mean_val:.2f}, 中央値: {median_val:.1f}, 標準偏差: {std_val:.2f}")
        
        print()
    
    def generate_quality_summary(self, common_ids, matching_rate, missing_results, q1_results):
        """データ品質サマリーの生成"""
        print("=" * 60)
        print("Phase 1 データ品質確認 結果サマリー")
        print("=" * 60)
        
        print(f"\n📊 データ規模:")
        print(f"・授業前回答者: {len(self.before_df)}名")
        print(f"・授業後回答者: {len(self.after_df)}名")
        print(f"・前後マッチング: {len(common_ids)}名 ({matching_rate:.1f}%)")
        
        print(f"\n🔍 データ品質:")
        
        # 20%以上欠損項目のチェック
        high_missing_items = []
        for dataset, missing_info in missing_results.items():
            if len(missing_info) > 0:
                high_missing = missing_info[missing_info['欠損率(%)'] > 20]
                if len(high_missing) > 0:
                    high_missing_items.extend([f"{dataset}: {item}" for item in high_missing.index])
        
        if not high_missing_items:
            print("・20%以上の欠損項目: なし ✅")
        else:
            print("・20%以上の欠損項目: あり ⚠️")
            for item in high_missing_items:
                print(f"  {item}")
        
        print(f"\n📈 主要結果:")
        if len(q1_results) > 0:
            print(f"・Q1水溶液認識項目: {len(q1_results)}項目で前後比較可能")
            avg_change = q1_results['変化'].mean()
            print(f"・平均正答率変化: {avg_change:.1f}ポイント")
            positive_changes = (q1_results['変化'] > 0).sum()
            print(f"・改善項目数: {positive_changes}/{len(q1_results)}項目")
            
            # 最も改善した項目
            best_improvement = q1_results.loc[q1_results['変化'].idxmax()]
            print(f"・最大改善: {best_improvement['物質']} (+{best_improvement['変化']:.1f}ポイント)")
        
        print(f"\n✅ Phase 1 完了判定:")
        
        # Phase 2進行の条件チェック
        can_proceed = True
        reasons = []
        
        if matching_rate < 80:
            can_proceed = False
            reasons.append("マッチング率が低い (< 80%)")
        
        if high_missing_items:
            can_proceed = False
            reasons.append("高欠損項目あり")
        
        if len(q1_results) < 4:
            can_proceed = False
            reasons.append("比較可能Q1項目が少ない")
        
        if can_proceed:
            print("🟢 Phase 2 (統計的検証) に進行可能")
        else:
            print("🟡 Phase 2 進行前に以下の課題を検討:")
            for reason in reasons:
                print(f"  - {reason}")
        
        print("=" * 60)
        return can_proceed
    
    def run_full_analysis(self):
        """Phase 1 フル分析の実行"""
        print("Phase 1: データ品質確認と基礎統計 実行開始")
        print("=" * 60)
        
        # 1. データ読み込み
        if not self.load_data():
            return False
        
        # 2. Page_IDマッチング確認
        common_ids, matching_rate = self.analyze_page_id_matching()
        
        # 3. 欠損値分析
        missing_results = self.analyze_missing_values()
        
        # 4. Q1項目分析
        q1_results, q1_correspondence = self.analyze_q1_responses()
        
        # 5. 記述統計量算出
        self.calculate_descriptive_stats()
        
        # 6. 評価項目分析
        self.analyze_evaluation_items()
        
        # 7. 品質サマリー
        can_proceed = self.generate_quality_summary(common_ids, matching_rate, missing_results, q1_results)
        
        return can_proceed, {
            'common_ids': common_ids,
            'matching_rate': matching_rate,
            'missing_results': missing_results,
            'q1_results': q1_results,
            'q1_correspondence': q1_correspondence,
            'can_proceed_phase2': can_proceed
        }

def main():
    """メイン実行関数"""
    analyzer = Phase1Analyzer()
    success, results = analyzer.run_full_analysis()
    
    if success:
        print("\n🎉 Phase 1 分析完了!")
        print("次ステップ: Phase 2 (教育効果の統計的検証) の実行")
    else:
        print("\n⚠️ Phase 1 でデータ品質課題を検出")
        print("データ修正またはPhase 2 での対応策検討が必要")
    
    return results

if __name__ == "__main__":
    results = main()