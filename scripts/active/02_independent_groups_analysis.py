#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小学校出前授業アンケート Phase 2（修正版）: 独立群比較による効果検証
====================================================

重要: Page_IDは個人識別子ではなく、単なるページ番号である。
そのため、授業前後の個人追跡は不可能であり、
独立した2群の比較として分析を行う。

実施内容:
- χ²検定による独立性の検定
- Mann-Whitney U検定による順序尺度データの比較
- 効果量の算出（Cohen's h, Cramer's V）
- 群間差の解釈

Author: Claude Code Analysis (Revised)
Date: 2025-05-31
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, fisher_exact, kruskal
import warnings
from pathlib import Path
import json
from datetime import datetime

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
warnings.filterwarnings('ignore')

class Phase2IndependentGroupsAnalyzer:
    """Phase 2修正版: 独立群比較分析クラス"""
    
    def __init__(self, data_dir="data/analysis"):
        self.data_dir = Path(data_dir)
        self.results = {}
        self.before_df = None
        self.after_df = None
        self.alpha = 0.05
        
    def load_data(self):
        """データの読み込み（独立群として）"""
        try:
            self.before_df = pd.read_csv(self.data_dir / "before_excel_compliant.csv")
            self.after_df = pd.read_csv(self.data_dir / "after_excel_compliant.csv")
            
            print("✓ データ読み込み完了")
            print(f"  - 授業前群: {len(self.before_df)} 名")
            print(f"  - 授業後群: {len(self.after_df)} 名")
            print("\n⚠️  重要: これらは独立した2群であり、個人の追跡は不可能です")
            
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            raise
    
    def chi_square_analysis(self):
        """χ²検定による独立性の検定"""
        print("\n" + "="*50)
        print("χ²検定による独立群比較分析")
        print("="*50)
        
        chi_square_results = {
            'q1_results': {},
            'q3_results': {},
            'summary': {}
        }
        
        # Q1項目の分析
        print("\n1. Q1: 水溶液認識項目の群間比較")
        print("-" * 30)
        
        q1_items = {
            'Saltwater': ('Q1_Saltwater_Response', 'Q1_Saltwater'),
            'Sugarwater': ('Q1_Sugarwater_Response', 'Q1_Sugarwater'),
            'Muddywater': ('Q1_Muddywater_Response', 'Q1_Muddywater'),
            'Ink': ('Q1_Ink_Response', 'Q1_Ink'),
            'MisoSoup': ('Q1_MisoSoup_Response', 'Q1_MisoSoup'),
            'SoySauce': ('Q1_SoySauce_Response', 'Q1_SoySauce')
        }
        
        for item_name, (before_col, after_col) in q1_items.items():
            if before_col in self.before_df.columns and after_col in self.after_df.columns:
                result = self.perform_chi_square_test(
                    self.before_df[before_col],
                    self.after_df[after_col],
                    item_name
                )
                chi_square_results['q1_results'][item_name] = result
        
        # Q3項目の分析
        print("\n2. Q3: お茶の理解度項目の群間比較")
        print("-" * 30)
        
        q3_items = {
            'TeaLeavesDissolve': ('Q3_TeaLeavesDissolve', 'Q3_TeaLeaves_DissolveInWater'),
            'TeaComponentsDissolve': ('Q3_TeaComponentsDissolve', 'Q3_TeaComponents_DissolveInWater')
        }
        
        for item_name, (before_col, after_col) in q3_items.items():
            if before_col in self.before_df.columns and after_col in self.after_df.columns:
                result = self.perform_chi_square_test(
                    self.before_df[before_col],
                    self.after_df[after_col],
                    item_name
                )
                chi_square_results['q3_results'][item_name] = result
        
        # 多重比較補正
        print("\n3. 多重比較補正")
        print("-" * 30)
        
        correction_results = self.apply_multiple_testing_correction(chi_square_results)
        chi_square_results['multiple_testing'] = correction_results
        
        self.results['chi_square_analysis'] = chi_square_results
        return chi_square_results
    
    def perform_chi_square_test(self, before_data, after_data, item_name):
        """個別項目のχ²検定実行"""
        # データ準備
        before_clean = before_data.dropna()
        after_clean = after_data.dropna()
        
        # 2x2分割表の作成
        before_true = (before_clean == True).sum()
        before_false = (before_clean == False).sum()
        after_true = (after_clean == True).sum()
        after_false = (after_clean == False).sum()
        
        contingency_table = np.array([
            [before_false, before_true],
            [after_false, after_true]
        ])
        
        # χ²検定
        if min(contingency_table.flatten()) < 5:
            # Fisher's exact test for small samples
            odds_ratio, p_value = fisher_exact(contingency_table)
            test_type = "Fisher's exact"
            test_statistic = odds_ratio
        else:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            test_type = "Chi-square"
            test_statistic = chi2
            
        # 効果量（Cohen's h）の計算
        p1 = before_true / (before_true + before_false)
        p2 = after_true / (after_true + after_false)
        cohens_h = self.calculate_cohens_h(p1, p2)
        
        # Cramer's V（分割表の効果量）
        n = contingency_table.sum()
        if test_type == "Chi-square":
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
        else:
            cramers_v = np.nan
        
        result = {
            'before_n': len(before_clean),
            'after_n': len(after_clean),
            'before_true_rate': float(p1),
            'after_true_rate': float(p2),
            'difference': float(p2 - p1),
            'contingency_table': contingency_table.tolist(),
            'test_type': test_type,
            'test_statistic': float(test_statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'cohens_h': float(cohens_h),
            'cramers_v': float(cramers_v) if not np.isnan(cramers_v) else None,
            'interpretation': self.interpret_chi_square_result(p_value, cohens_h)
        }
        
        # 結果表示
        direction = "高い" if p2 > p1 else "低い" if p2 < p1 else "同等"
        sig_symbol = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        
        print(f"  {item_name:20}: 授業前 {p1:.3f} vs 授業後 {p2:.3f}")
        print(f"    {test_type}: χ² = {test_statistic:.3f}, p = {p_value:.4f} {sig_symbol}")
        print(f"    授業後群は授業前群より{direction} (差: {p2-p1:+.3f})")
        print(f"    効果量 Cohen's h = {cohens_h:.3f}")
        
        return result
    
    def calculate_cohens_h(self, p1, p2):
        """Cohen's h（割合の効果量）の計算"""
        # アークサイン変換
        phi1 = 2 * np.arcsin(np.sqrt(p1))
        phi2 = 2 * np.arcsin(np.sqrt(p2))
        return phi2 - phi1
    
    def interpret_chi_square_result(self, p_value, cohens_h):
        """χ²検定結果の解釈"""
        # 有意性の判定
        if p_value < 0.001:
            significance = "極めて有意な差"
        elif p_value < 0.01:
            significance = "高度に有意な差"
        elif p_value < 0.05:
            significance = "有意な差"
        else:
            significance = "有意差なし"
        
        # 効果量の判定
        abs_h = abs(cohens_h)
        if abs_h < 0.2:
            effect_interpretation = "効果なし/小"
        elif abs_h < 0.5:
            effect_interpretation = "中程度の効果"
        elif abs_h < 0.8:
            effect_interpretation = "大きな効果"
        else:
            effect_interpretation = "極めて大きな効果"
        
        return f"{significance}, {effect_interpretation}"
    
    def mann_whitney_analysis(self):
        """Mann-Whitney U検定による順序尺度データの比較"""
        print("\n" + "="*50)
        print("Mann-Whitney U検定による評価項目の群間比較")
        print("="*50)
        
        mw_results = {}
        
        # 授業後のみの評価項目
        evaluation_items = {
            'Q4_ExperimentInterestRating': '実験への興味',
            'Q5_NewLearningsRating': '新しい学び',
            'Q6_DissolvingUnderstandingRating': '溶解理解度'
        }
        
        print("\n授業後評価項目の分析")
        print("（注: これらは授業後のみのデータのため、クラス間比較として分析）")
        print("-" * 30)
        
        for col_name, label in evaluation_items.items():
            if col_name in self.after_df.columns:
                # クラス間比較として分析
                result = self.analyze_evaluation_by_class(col_name, label)
                mw_results[col_name] = result
        
        self.results['mann_whitney_analysis'] = mw_results
        return mw_results
    
    def analyze_evaluation_by_class(self, column, label):
        """評価項目のクラス間比較"""
        data = self.after_df[[column, 'class']].dropna()
        
        if len(data) < 10:
            return {"error": f"サンプルサイズ不足 (n={len(data)})"}
        
        # クラス別の基本統計
        class_stats = {}
        classes = sorted(data['class'].unique())
        
        for cls in classes:
            class_data = data[data['class'] == cls][column]
            class_stats[cls] = {
                'n': len(class_data),
                'mean': float(class_data.mean()),
                'median': float(class_data.median()),
                'std': float(class_data.std()),
                'min': float(class_data.min()),
                'max': float(class_data.max())
            }
        
        # Kruskal-Wallis検定（3群以上の比較）
        class_groups = [data[data['class'] == cls][column].values for cls in classes]
        h_stat, p_value = kruskal(*class_groups)
        
        result = {
            'variable': column,
            'label': label,
            'n_total': len(data),
            'class_statistics': class_stats,
            'test_type': 'Kruskal-Wallis',
            'h_statistic': float(h_stat),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'interpretation': self.interpret_kruskal_wallis(p_value)
        }
        
        # 結果表示
        print(f"\n{label}:")
        for cls, stats in class_stats.items():
            print(f"  クラス{cls}: 平均 {stats['mean']:.2f}, 中央値 {stats['median']:.1f} (n={stats['n']})")
        print(f"  Kruskal-Wallis H = {h_stat:.3f}, p = {p_value:.4f}")
        
        return result
    
    def interpret_kruskal_wallis(self, p_value):
        """Kruskal-Wallis検定結果の解釈"""
        if p_value < 0.001:
            return "クラス間に極めて有意な差"
        elif p_value < 0.01:
            return "クラス間に高度に有意な差"
        elif p_value < 0.05:
            return "クラス間に有意な差"
        else:
            return "クラス間に有意差なし"
    
    def composite_score_analysis(self):
        """総合スコアの独立群比較"""
        print("\n" + "="*50)
        print("総合スコアによる独立群比較")
        print("="*50)
        
        composite_results = {}
        
        # Q1総合スコア
        print("\n1. Q1総合スコア分析")
        print("-" * 30)
        
        q1_composite = self.calculate_composite_score_independent('Q1')
        composite_results['q1_composite'] = q1_composite
        
        # Q3総合スコア  
        print("\n2. Q3総合スコア分析")
        print("-" * 30)
        
        q3_composite = self.calculate_composite_score_independent('Q3')
        composite_results['q3_composite'] = q3_composite
        
        self.results['composite_analysis'] = composite_results
        return composite_results
    
    def calculate_composite_score_independent(self, category):
        """独立群での総合スコア比較"""
        if category == 'Q1':
            before_cols = [col for col in self.before_df.columns 
                          if col.startswith('Q1_') and col.endswith('_Response')]
            after_cols = [col for col in self.after_df.columns 
                         if col.startswith('Q1_') and not col.endswith('_Response') 
                         and not col.endswith('Reason')]
        else:  # Q3
            before_cols = ['Q3_TeaLeavesDissolve', 'Q3_TeaComponentsDissolve']
            after_cols = ['Q3_TeaLeaves_DissolveInWater', 'Q3_TeaComponents_DissolveInWater']
        
        # スコア計算
        before_scores = self.before_df[before_cols].sum(axis=1)
        after_scores = self.after_df[after_cols].sum(axis=1)
        
        # Mann-Whitney U検定
        u_stat, p_value = mannwhitneyu(before_scores, after_scores, alternative='two-sided')
        
        # 効果量（r = Z / sqrt(N)）
        n1, n2 = len(before_scores), len(after_scores)
        z_score = stats.norm.ppf(p_value / 2)  # 両側検定
        effect_size_r = abs(z_score) / np.sqrt(n1 + n2)
        
        # Cohen's d（独立群）
        pooled_std = np.sqrt(((n1-1)*before_scores.var() + (n2-1)*after_scores.var()) / (n1+n2-2))
        cohens_d = (after_scores.mean() - before_scores.mean()) / pooled_std if pooled_std > 0 else 0
        
        results = {
            'before_n': n1,
            'after_n': n2,
            'before_mean': float(before_scores.mean()),
            'after_mean': float(after_scores.mean()),
            'before_std': float(before_scores.std()),
            'after_std': float(after_scores.std()),
            'mean_difference': float(after_scores.mean() - before_scores.mean()),
            'u_statistic': float(u_stat),
            'p_value': float(p_value),
            'effect_size_r': float(effect_size_r),
            'cohens_d': float(cohens_d),
            'significant': p_value < self.alpha,
            'interpretation': self.interpret_mann_whitney(p_value, cohens_d)
        }
        
        # 結果表示
        direction = "高い" if results['mean_difference'] > 0 else "低い" if results['mean_difference'] < 0 else "同等"
        sig_symbol = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        
        print(f"  {category}総合スコア:")
        print(f"    授業前群: {results['before_mean']:.2f} ± {results['before_std']:.2f} (n={n1})")
        print(f"    授業後群: {results['after_mean']:.2f} ± {results['after_std']:.2f} (n={n2})")
        print(f"    授業後群は授業前群より{direction} (差: {results['mean_difference']:.2f})")
        print(f"    Mann-Whitney U = {u_stat:.1f}, p = {p_value:.4f} {sig_symbol}")
        print(f"    効果量: r = {effect_size_r:.3f}, Cohen's d = {cohens_d:.3f}")
        
        return results
    
    def interpret_mann_whitney(self, p_value, cohens_d):
        """Mann-Whitney検定結果の解釈"""
        # 有意性
        if p_value < 0.001:
            significance = "極めて有意な差"
        elif p_value < 0.01:
            significance = "高度に有意な差"
        elif p_value < 0.05:
            significance = "有意な差"
        else:
            significance = "有意差なし"
        
        # 効果量
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            effect = "効果なし/小"
        elif abs_d < 0.5:
            effect = "中程度の効果"
        elif abs_d < 0.8:
            effect = "大きな効果"
        else:
            effect = "極めて大きな効果"
        
        return f"{significance}, {effect}"
    
    def apply_multiple_testing_correction(self, chi_square_results):
        """多重比較補正"""
        from statsmodels.stats.multitest import multipletests
        
        # p値の収集
        p_values = []
        test_names = []
        
        for category in ['q1_results', 'q3_results']:
            for item, result in chi_square_results[category].items():
                p_values.append(result['p_value'])
                test_names.append(f"{category}_{item}")
        
        # Benjamini-Hochberg法による補正
        rejected, p_adjusted, alpha_sidak, alpha_bonf = multipletests(
            p_values, method='fdr_bh', alpha=self.alpha
        )
        
        # Bonferroni補正も計算
        bonferroni_p = np.minimum(np.array(p_values) * len(p_values), 1.0)
        
        correction_results = {
            'n_tests': len(p_values),
            'method': 'Benjamini-Hochberg (FDR)',
            'corrected_results': {}
        }
        
        for i, test_name in enumerate(test_names):
            correction_results['corrected_results'][test_name] = {
                'original_p': p_values[i],
                'fdr_adjusted_p': p_adjusted[i],
                'bonferroni_p': bonferroni_p[i],
                'fdr_significant': rejected[i],
                'bonferroni_significant': bonferroni_p[i] < self.alpha
            }
        
        # 結果表示
        print(f"総検定数: {len(p_values)}")
        fdr_sig_count = sum(rejected)
        bonf_sig_count = sum(1 for p in bonferroni_p if p < self.alpha)
        print(f"FDR補正後の有意な検定数: {fdr_sig_count}")
        print(f"Bonferroni補正後の有意な検定数: {bonf_sig_count}")
        
        return correction_results
    
    def create_visualizations(self):
        """可視化の作成"""
        print("\n" + "="*50)
        print("独立群比較結果の可視化")
        print("="*50)
        
        output_dir = Path("outputs/phase2_revised_figures")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # χ²検定結果の可視化
        self.plot_chi_square_results(output_dir)
        
        # 総合スコアの比較
        self.plot_composite_scores(output_dir)
        
        # 効果量の比較
        self.plot_effect_sizes(output_dir)
        
        print(f"✓ 図表を {output_dir} に保存しました")
    
    def plot_chi_square_results(self, output_dir):
        """χ²検定結果の可視化"""
        if 'chi_square_analysis' not in self.results:
            return
        
        # データ収集
        items = []
        before_rates = []
        after_rates = []
        p_values = []
        
        for category in ['q1_results', 'q3_results']:
            for item, result in self.results['chi_square_analysis'][category].items():
                items.append(item)
                before_rates.append(result['before_true_rate'] * 100)
                after_rates.append(result['after_true_rate'] * 100)
                p_values.append(result['p_value'])
        
        if not items:
            return
        
        # 図の作成
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 正答率比較
        x = np.arange(len(items))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, before_rates, width, label='Before Class Group', alpha=0.8)
        bars2 = ax1.bar(x + width/2, after_rates, width, label='After Class Group', alpha=0.8)
        
        ax1.set_xlabel('Items')
        ax1.set_ylabel('True Response Rate (%)')
        ax1.set_title('Independent Groups Comparison (Before vs After)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(items, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 有意性マーク
        for i, p_val in enumerate(p_values):
            if p_val < 0.001:
                sig_text = '***'
            elif p_val < 0.01:
                sig_text = '**'
            elif p_val < 0.05:
                sig_text = '*'
            else:
                sig_text = 'ns'
            
            max_height = max(before_rates[i], after_rates[i])
            ax1.text(i, max_height + 2, sig_text, ha='center', va='bottom', fontweight='bold')
        
        # 効果量（Cohen's h）
        cohen_h_values = [self.results['chi_square_analysis'][cat][item]['cohens_h'] 
                         for cat in ['q1_results', 'q3_results'] 
                         for item in self.results['chi_square_analysis'][cat].keys()]
        
        colors = ['green' if h > 0 else 'red' for h in cohen_h_values]
        ax2.bar(range(len(items)), cohen_h_values, color=colors, alpha=0.7)
        ax2.set_xlabel('Items')
        ax2.set_ylabel("Cohen's h (Effect Size)")
        ax2.set_title('Effect Sizes for Each Item')
        ax2.set_xticks(range(len(items)))
        ax2.set_xticklabels(items, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
        ax2.axhline(y=-0.2, color='gray', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "chi_square_results.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_composite_scores(self, output_dir):
        """総合スコア比較の可視化"""
        if 'composite_analysis' not in self.results:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, (category, result) in enumerate(self.results['composite_analysis'].items()):
            ax = axes[i]
            
            # ボックスプロット風の表現
            groups = ['Before', 'After']
            means = [result['before_mean'], result['after_mean']]
            stds = [result['before_std'], result['after_std']]
            
            # バープロット
            bars = ax.bar(groups, means, alpha=0.7, color=['lightblue', 'lightcoral'])
            
            # エラーバー
            ax.errorbar(groups, means, yerr=stds, fmt='o', color='black', capsize=5)
            
            # 統計情報の表示
            sig_text = '***' if result['p_value'] < 0.001 else '**' if result['p_value'] < 0.01 else '*' if result['p_value'] < 0.05 else 'ns'
            ax.text(0.5, max(means) * 1.1, sig_text, ha='center', transform=ax.transAxes, fontweight='bold')
            
            ax.set_ylabel('Composite Score')
            ax.set_title(f'{category.upper()}\np = {result["p_value"]:.4f}, d = {result["cohens_d"]:.3f}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "composite_scores_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_effect_sizes(self, output_dir):
        """効果量の統合的可視化"""
        # すべての効果量を収集
        all_effects = []
        labels = []
        
        # χ²検定からのCohen's h
        if 'chi_square_analysis' in self.results:
            for category in ['q1_results', 'q3_results']:
                for item, result in self.results['chi_square_analysis'][category].items():
                    all_effects.append(result['cohens_h'])
                    labels.append(f"{item} (h)")
        
        # 総合スコアからのCohen's d
        if 'composite_analysis' in self.results:
            for category, result in self.results['composite_analysis'].items():
                all_effects.append(result['cohens_d'])
                labels.append(f"{category} (d)")
        
        if not all_effects:
            return
        
        # 図の作成
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['green' if e > 0 else 'red' for e in all_effects]
        y_pos = np.arange(len(labels))
        
        bars = ax.barh(y_pos, all_effects, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Effect Size')
        ax.set_title('Effect Sizes Summary (Independent Groups Comparison)')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # 効果量の基準線
        ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Small')
        ax.axvline(x=-0.2, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='Medium')
        ax.axvline(x=-0.5, color='gray', linestyle=':', alpha=0.5)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "effect_sizes_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self):
        """Phase 2修正版レポート生成"""
        print("\n" + "="*50)
        print("Phase 2修正版レポート生成")
        print("="*50)
        
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # JSON形式で詳細結果を保存
        with open(output_dir / "phase2_revised_results.json", 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        
        # テキスト形式でサマリーレポートを生成
        report_content = self.create_summary_report()
        
        with open(output_dir / "phase2_revised_report.txt", 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✓ レポートを {output_dir} に保存しました")
        print(f"  - 詳細結果: phase2_revised_results.json")
        print(f"  - サマリー: phase2_revised_report.txt")
        
        return report_content
    
    def create_summary_report(self):
        """サマリーレポートの作成"""
        report = []
        report.append("="*60)
        report.append("小学校出前授業アンケート Phase 2（修正版）独立群比較結果")
        report.append("="*60)
        report.append(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"有意水準: α = {self.alpha}")
        report.append("")
        
        report.append("【重要な前提】")
        report.append("Page_IDは個人識別子ではなく単なるページ番号であるため、")
        report.append("授業前後の個人追跡は不可能です。")
        report.append("本分析は独立した2群（授業前群 vs 授業後群）の比較として実施しました。")
        report.append("")
        
        # サンプルサイズ
        report.append("【サンプルサイズ】")
        report.append(f"授業前群: {len(self.before_df)} 名")
        report.append(f"授業後群: {len(self.after_df)} 名")
        report.append("")
        
        # χ²検定結果
        if 'chi_square_analysis' in self.results:
            report.append("【χ²検定結果】")
            
            for category, label in [('q1_results', 'Q1: 水溶液認識'), ('q3_results', 'Q3: お茶の理解度')]:
                report.append(f"\n{label}:")
                
                results = self.results['chi_square_analysis'][category]
                significant_items = []
                
                for item, result in results.items():
                    direction = "高い" if result['difference'] > 0 else "低い" if result['difference'] < 0 else "同等"
                    
                    report.append(f"  {item:15}: 授業前 {result['before_true_rate']:.3f} vs 授業後 {result['after_true_rate']:.3f}")
                    report.append(f"    授業後群は{direction} (差: {result['difference']:+.3f})")
                    report.append(f"    p = {result['p_value']:.4f}, Cohen's h = {result['cohens_h']:.3f}")
                    
                    if result['significant']:
                        significant_items.append(item)
                
                if significant_items:
                    report.append(f"  有意な群間差: {', '.join(significant_items)}")
                else:
                    report.append("  有意な群間差なし")
            
            # 多重比較補正
            if 'multiple_testing' in self.results['chi_square_analysis']:
                correction = self.results['chi_square_analysis']['multiple_testing']
                report.append(f"\n多重比較補正 (検定数: {correction['n_tests']}):")
                
                fdr_sig = sum(1 for r in correction['corrected_results'].values() 
                            if r['fdr_significant'])
                bonf_sig = sum(1 for r in correction['corrected_results'].values() 
                             if r['bonferroni_significant'])
                
                report.append(f"  FDR補正後有意: {fdr_sig}項目")
                report.append(f"  Bonferroni補正後有意: {bonf_sig}項目")
            
            report.append("")
        
        # 総合スコア分析
        if 'composite_analysis' in self.results:
            report.append("【総合スコア分析（Mann-Whitney U検定）】")
            
            for category, result in self.results['composite_analysis'].items():
                category_label = "Q1総合" if "q1" in category else "Q3総合"
                direction = "高い" if result['mean_difference'] > 0 else "低い" if result['mean_difference'] < 0 else "同等"
                
                report.append(f"\n{category_label}:")
                report.append(f"  授業前群: {result['before_mean']:.2f} ± {result['before_std']:.2f} (n={result['before_n']})")
                report.append(f"  授業後群: {result['after_mean']:.2f} ± {result['after_std']:.2f} (n={result['after_n']})")
                report.append(f"  授業後群は{direction} (差: {result['mean_difference']:.2f})")
                report.append(f"  U = {result['u_statistic']:.1f}, p = {result['p_value']:.4f}")
                report.append(f"  効果量: r = {result['effect_size_r']:.3f}, Cohen's d = {result['cohens_d']:.3f}")
                report.append(f"  結果: {result['interpretation']}")
            
            report.append("")
        
        # 主要な発見事項
        report.append("【主要な発見事項】")
        
        # 有意な群間差があったかどうか
        significant_count = 0
        if 'chi_square_analysis' in self.results:
            for category in ['q1_results', 'q3_results']:
                for result in self.results['chi_square_analysis'][category].values():
                    if result['significant']:
                        significant_count += 1
        
        if significant_count > 0:
            report.append(f"✓ {significant_count}項目で有意な群間差を検出")
        else:
            report.append("• 統計的に有意な群間差は検出されず")
        
        # 効果量の評価
        large_effects = 0
        if 'chi_square_analysis' in self.results:
            for category in ['q1_results', 'q3_results']:
                for result in self.results['chi_square_analysis'][category].values():
                    if abs(result.get('cohens_h', 0)) > 0.5:
                        large_effects += 1
        
        if large_effects > 0:
            report.append(f"✓ {large_effects}項目で大きな効果量を確認")
        else:
            report.append("• 実質的に大きな効果は確認されず")
        
        report.append("")
        report.append("【解釈上の注意】")
        report.append("1. これは独立群比較であり、個人の変化を示すものではない")
        report.append("2. 群間差は授業効果以外の要因による可能性もある")
        report.append("3. ランダム割付でないため因果推論には限界がある")
        report.append("")
        
        report.append("【Phase 3への推奨事項】")
        report.append("1. クラス間差異を独立群として分析")
        report.append("2. 変化量分析は実施不可（ペアデータなし）")
        report.append("3. 群間差に影響する要因の探索")
        
        return "\n".join(report)
    
    def run_complete_analysis(self):
        """Phase 2修正版完全分析実行"""
        print("小学校出前授業アンケート Phase 2（修正版）: 独立群比較による効果検証")
        print("実行開始:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("="*60)
        
        try:
            # データ読み込み
            self.load_data()
            
            # χ²検定分析
            self.chi_square_analysis()
            
            # Mann-Whitney U検定分析
            self.mann_whitney_analysis()
            
            # 総合スコア分析
            self.composite_score_analysis()
            
            # 可視化作成
            self.create_visualizations()
            
            # レポート生成
            summary_report = self.generate_report()
            
            print("\n" + "="*60)
            print("Phase 2修正版分析完了!")
            print("="*60)
            print(summary_report)
            
            return self.results
            
        except Exception as e:
            print(f"❌ Phase 2修正版分析中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """メイン実行関数"""
    analyzer = Phase2IndependentGroupsAnalyzer()
    results = analyzer.run_complete_analysis()
    return results

if __name__ == "__main__":
    main()