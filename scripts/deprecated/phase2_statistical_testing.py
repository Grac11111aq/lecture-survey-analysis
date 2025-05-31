#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小学校出前授業アンケート Phase 2: 教育効果の統計的検証
==================================================

実施内容:
- McNemar検定による前後比較
- 対応のあるt検定
- 効果量の算出（Cohen's d）
- 多重比較補正
- 検出力分析

Author: Claude Code Analysis
Date: 2025-05-31
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
from pathlib import Path
import json
from datetime import datetime
from statsmodels.stats.contingency_tables import mcnemar as mcnemar_exact
from statsmodels.stats.power import TTestPower
from statsmodels.stats.multitest import multipletests

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
warnings.filterwarnings('ignore')

class Phase2StatisticalTester:
    """Phase 2: 統計的検証クラス"""
    
    def __init__(self, data_dir="data/analysis"):
        self.data_dir = Path(data_dir)
        self.results = {}
        self.before_df = None
        self.after_df = None
        self.alpha = 0.05
        self.paired_data = None
        
    def load_data(self):
        """データの読み込みとペアリング"""
        try:
            self.before_df = pd.read_csv(self.data_dir / "before_excel_compliant.csv")
            self.after_df = pd.read_csv(self.data_dir / "after_excel_compliant.csv")
            
            print("✓ データ読み込み完了")
            print(f"  - 授業前: {len(self.before_df)} 行")
            print(f"  - 授業後: {len(self.after_df)} 行")
            
            # Page_IDでペアリング
            self.create_paired_dataset()
            
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            raise
    
    def create_paired_dataset(self):
        """前後データのペアリング"""
        # Page_IDで結合（重複がある場合は最初の記録を使用）
        before_unique = self.before_df.drop_duplicates(subset=['Page_ID'])
        after_unique = self.after_df.drop_duplicates(subset=['Page_ID'])
        
        # 内部結合でペアデータを作成
        self.paired_data = pd.merge(
            before_unique, 
            after_unique, 
            on='Page_ID', 
            suffixes=('_before', '_after')
        )
        
        print(f"✓ ペアリング完了: {len(self.paired_data)} ペア")
        
        # ペアリング可能な項目を特定
        self.identify_matching_columns()
    
    def identify_matching_columns(self):
        """前後で対応する列の特定"""
        # Q1項目のマッピング
        self.q1_mapping = {
            'Q1_Saltwater_Response': 'Q1_Saltwater',
            'Q1_Sugarwater_Response': 'Q1_Sugarwater', 
            'Q1_Muddywater_Response': 'Q1_Muddywater',
            'Q1_Ink_Response': 'Q1_Ink',
            'Q1_MisoSoup_Response': 'Q1_MisoSoup',
            'Q1_SoySauce_Response': 'Q1_SoySauce'
        }
        
        # Q3項目のマッピング
        self.q3_mapping = {
            'Q3_TeaLeavesDissolve': 'Q3_TeaLeaves_DissolveInWater',
            'Q3_TeaComponentsDissolve': 'Q3_TeaComponents_DissolveInWater'
        }
        
        print("✓ 列マッピング特定完了")
    
    def mcnemar_test_analysis(self):
        """McNemar検定による前後比較分析"""
        print("\n" + "="*50)
        print("McNemar検定による前後比較分析")
        print("="*50)
        
        mcnemar_results = {
            'q1_results': {},
            'q3_results': {},
            'summary': {}
        }
        
        # Q1項目のMcNemar検定
        print("\n1. Q1: 水溶液認識項目")
        print("-" * 30)
        
        q1_results = self.perform_mcnemar_tests(self.q1_mapping, "Q1")
        mcnemar_results['q1_results'] = q1_results
        
        # Q3項目のMcNemar検定
        print("\n2. Q3: お茶の理解度項目")
        print("-" * 30)
        
        q3_results = self.perform_mcnemar_tests(self.q3_mapping, "Q3")
        mcnemar_results['q3_results'] = q3_results
        
        # 多重比較補正
        print("\n3. 多重比較補正")
        print("-" * 30)
        
        correction_results = self.apply_multiple_testing_correction(mcnemar_results)
        mcnemar_results['multiple_testing'] = correction_results
        
        self.results['mcnemar_analysis'] = mcnemar_results
        return mcnemar_results
    
    def perform_mcnemar_tests(self, mapping, category):
        """特定カテゴリのMcNemar検定実行"""
        results = {}
        
        for before_col, after_col in mapping.items():
            if before_col in self.paired_data.columns and after_col in self.paired_data.columns:
                # データ準備
                before_data = self.paired_data[before_col]
                after_data = self.paired_data[after_col]
                
                # 欠損値を除外
                valid_mask = ~(before_data.isna() | after_data.isna())
                before_clean = before_data[valid_mask]
                after_clean = after_data[valid_mask]
                
                if len(before_clean) < 10:
                    print(f"  ⚠️  {before_col}: サンプルサイズ不足 (n={len(before_clean)})")
                    continue
                
                # 分割表作成
                crosstab = pd.crosstab(before_clean, after_clean)
                
                # McNemar検定実行
                test_result = self.run_mcnemar_test(crosstab, before_col)
                
                # 効果量計算
                effect_size = self.calculate_effect_size_mcnemar(before_clean, after_clean)
                
                results[before_col] = {
                    'n_pairs': len(before_clean),
                    'before_true_rate': float(before_clean.mean()),
                    'after_true_rate': float(after_clean.mean()),
                    'change': float(after_clean.mean() - before_clean.mean()),
                    'crosstab': crosstab.to_dict(),
                    'mcnemar_statistic': test_result['statistic'],
                    'p_value': test_result['p_value'],
                    'significant': test_result['p_value'] < self.alpha,
                    'effect_size': effect_size,
                    'interpretation': self.interpret_mcnemar_result(test_result, effect_size)
                }
                
                # 結果表示
                change_symbol = "↗️" if results[before_col]['change'] > 0.05 else "↘️" if results[before_col]['change'] < -0.05 else "→"
                sig_symbol = "***" if test_result['p_value'] < 0.001 else "**" if test_result['p_value'] < 0.01 else "*" if test_result['p_value'] < 0.05 else ""
                
                print(f"  {before_col:20}: {results[before_col]['before_true_rate']:.3f} → {results[before_col]['after_true_rate']:.3f} {change_symbol}")
                print(f"    McNemar χ² = {test_result['statistic']:.3f}, p = {test_result['p_value']:.4f} {sig_symbol}")
                print(f"    効果量 = {effect_size:.3f}, n = {len(before_clean)}")
        
        return results
    
    def run_mcnemar_test(self, crosstab, item_name):
        """McNemar検定の実行"""
        try:
            # 2x2分割表の確認
            if crosstab.shape != (2, 2):
                # 不完全な分割表の場合、0で埋める
                crosstab = crosstab.reindex([False, True], fill_value=0)
                crosstab = crosstab.reindex([False, True], axis=1, fill_value=0)
            
            # McNemar検定（正確検定を使用）
            result = mcnemar_exact(crosstab.values, exact=True)
            
            return {
                'statistic': float(result.statistic),
                'p_value': float(result.pvalue)
            }
            
        except Exception as e:
            print(f"    ❌ McNemar検定エラー ({item_name}): {e}")
            return {
                'statistic': np.nan,
                'p_value': np.nan
            }
    
    def calculate_effect_size_mcnemar(self, before_data, after_data):
        """McNemar検定の効果量計算（Cohen's g）"""
        try:
            # 不一致ペアの数
            before_true_after_false = ((before_data == True) & (after_data == False)).sum()
            before_false_after_true = ((before_data == False) & (after_data == True)).sum()
            
            total_discordant = before_true_after_false + before_false_after_true
            
            if total_discordant == 0:
                return 0.0
            
            # Cohen's g = (b - c) / sqrt(b + c)
            # b = False→True, c = True→False
            effect_size = (before_false_after_true - before_true_after_false) / np.sqrt(total_discordant)
            
            return float(effect_size)
            
        except Exception:
            return np.nan
    
    def interpret_mcnemar_result(self, test_result, effect_size):
        """McNemar検定結果の解釈"""
        p_val = test_result['p_value']
        
        # 有意性の判定
        if np.isnan(p_val):
            significance = "解析不可"
        elif p_val < 0.001:
            significance = "極めて有意"
        elif p_val < 0.01:
            significance = "高度に有意"
        elif p_val < 0.05:
            significance = "有意"
        else:
            significance = "非有意"
        
        # 効果量の判定
        if np.isnan(effect_size):
            effect_interpretation = "効果量算出不可"
        elif abs(effect_size) < 0.2:
            effect_interpretation = "効果なし/小"
        elif abs(effect_size) < 0.5:
            effect_interpretation = "中程度の効果"
        else:
            effect_interpretation = "大きな効果"
        
        return f"{significance}, {effect_interpretation}"
    
    def apply_multiple_testing_correction(self, mcnemar_results):
        """多重比較補正の適用"""
        # 全てのp値を収集
        all_p_values = []
        all_tests = []
        
        for category in ['q1_results', 'q3_results']:
            for item, result in mcnemar_results[category].items():
                if not np.isnan(result['p_value']):
                    all_p_values.append(result['p_value'])
                    all_tests.append(f"{category}_{item}")
        
        if not all_p_values:
            return {"method": "No valid tests", "corrected_results": {}}
        
        # Bonferroni補正
        bonferroni_corrected = multipletests(all_p_values, method='bonferroni')[1]
        
        # FDR (Benjamini-Hochberg) 補正
        fdr_corrected = multipletests(all_p_values, method='fdr_bh')[1]
        
        correction_results = {
            "method": "Bonferroni + FDR",
            "total_tests": len(all_p_values),
            "corrected_results": {}
        }
        
        print(f"総検定数: {len(all_p_values)}")
        print("補正結果:")
        
        for i, test_name in enumerate(all_tests):
            original_p = all_p_values[i]
            bonf_p = bonferroni_corrected[i]
            fdr_p = fdr_corrected[i]
            
            correction_results["corrected_results"][test_name] = {
                "original_p": float(original_p),
                "bonferroni_p": float(bonf_p),
                "fdr_p": float(fdr_p),
                "bonferroni_significant": bonf_p < self.alpha,
                "fdr_significant": fdr_p < self.alpha
            }
            
            print(f"  {test_name:30}: p={original_p:.4f} → Bonf={bonf_p:.4f}, FDR={fdr_p:.4f}")
        
        return correction_results
    
    def composite_score_analysis(self):
        """総合スコアの分析"""
        print("\n" + "="*50)
        print("総合スコアによる対応のあるt検定")
        print("="*50)
        
        composite_results = {}
        
        # Q1総合スコア
        print("\n1. Q1総合スコア分析")
        print("-" * 30)
        
        q1_composite = self.calculate_composite_score(self.q1_mapping, "Q1")
        composite_results['q1_composite'] = q1_composite
        
        # Q3総合スコア  
        print("\n2. Q3総合スコア分析")
        print("-" * 30)
        
        q3_composite = self.calculate_composite_score(self.q3_mapping, "Q3")
        composite_results['q3_composite'] = q3_composite
        
        self.results['composite_analysis'] = composite_results
        return composite_results
    
    def calculate_composite_score(self, mapping, category):
        """総合スコアの計算と統計検定"""
        # スコア計算
        before_scores = []
        after_scores = []
        
        for before_col, after_col in mapping.items():
            if before_col in self.paired_data.columns and after_col in self.paired_data.columns:
                before_scores.append(self.paired_data[before_col].astype(float))
                after_scores.append(self.paired_data[after_col].astype(float))
        
        if not before_scores:
            return {"error": "No valid columns found"}
        
        # 総合スコア作成
        before_total = pd.concat(before_scores, axis=1).sum(axis=1)
        after_total = pd.concat(after_scores, axis=1).sum(axis=1)
        
        # 欠損値除外
        valid_mask = ~(before_total.isna() | after_total.isna())
        before_clean = before_total[valid_mask]
        after_clean = after_total[valid_mask]
        
        # 対応のあるt検定
        t_stat, p_value = stats.ttest_rel(before_clean, after_clean)
        
        # 効果量計算（Cohen's d）
        diff = after_clean - before_clean
        pooled_std = np.sqrt((before_clean.var() + after_clean.var()) / 2)
        cohens_d = diff.mean() / pooled_std if pooled_std > 0 else 0
        
        # 95%信頼区間
        diff_mean = diff.mean()
        diff_se = diff.std() / np.sqrt(len(diff))
        t_critical = stats.t.ppf(0.975, len(diff) - 1)
        ci_lower = diff_mean - t_critical * diff_se
        ci_upper = diff_mean + t_critical * diff_se
        
        results = {
            'n_pairs': len(before_clean),
            'before_mean': float(before_clean.mean()),
            'after_mean': float(after_clean.mean()),
            'before_std': float(before_clean.std()),
            'after_std': float(after_clean.std()),
            'mean_difference': float(diff_mean),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'ci_95_lower': float(ci_lower),
            'ci_95_upper': float(ci_upper),
            'significant': p_value < self.alpha,
            'effect_interpretation': self.interpret_effect_size(cohens_d)
        }
        
        # 結果表示
        direction = "改善" if diff_mean > 0 else "悪化" if diff_mean < 0 else "変化なし"
        sig_symbol = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        
        print(f"  {category}総合スコア:")
        print(f"    授業前: {results['before_mean']:.2f} ± {results['before_std']:.2f}")
        print(f"    授業後: {results['after_mean']:.2f} ± {results['after_std']:.2f}")
        print(f"    差分: {results['mean_difference']:.2f} [{results['ci_95_lower']:.2f}, {results['ci_95_upper']:.2f}] ({direction})")
        print(f"    t({len(diff)-1}) = {t_stat:.3f}, p = {p_value:.4f} {sig_symbol}")
        print(f"    Cohen's d = {cohens_d:.3f} ({results['effect_interpretation']})")
        
        return results
    
    def interpret_effect_size(self, cohens_d):
        """効果量の解釈"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "効果なし/小"
        elif abs_d < 0.5:
            return "中程度の効果"
        elif abs_d < 0.8:
            return "大きな効果"
        else:
            return "極めて大きな効果"
    
    def power_analysis(self):
        """検出力分析"""
        print("\n" + "="*50)
        print("検出力分析")
        print("="*50)
        
        power_results = {}
        
        # 現在のサンプルサイズでの検出力
        n = len(self.paired_data)
        effect_sizes = [0.2, 0.5, 0.8]  # 小、中、大の効果量
        
        power_calc = TTestPower()
        
        print(f"現在のサンプルサイズ (n={n}) での検出力:")
        
        for effect_size in effect_sizes:
            power = power_calc.solve_power(effect_size=effect_size, nobs=n, alpha=self.alpha)
            power_results[f"effect_{effect_size}"] = {
                "effect_size": effect_size,
                "sample_size": n,
                "power": float(power),
                "adequate": power >= 0.8
            }
            
            adequacy = "✓ 十分" if power >= 0.8 else "⚠️ 不足"
            print(f"  効果量 d={effect_size}: 検出力 = {power:.3f} {adequacy}")
        
        # 十分な検出力(0.8)を得るために必要なサンプルサイズ
        print("\n検出力0.8を得るために必要なサンプルサイズ:")
        
        for effect_size in effect_sizes:
            required_n = power_calc.solve_power(effect_size=effect_size, power=0.8, alpha=self.alpha)
            power_results[f"required_n_effect_{effect_size}"] = {
                "effect_size": effect_size,
                "required_n": int(np.ceil(required_n)),
                "current_n": n,
                "sufficient": n >= required_n
            }
            
            status = "✓ 十分" if n >= required_n else f"⚠️ 不足 (あと{int(np.ceil(required_n)) - n}名)"
            print(f"  効果量 d={effect_size}: n = {int(np.ceil(required_n))} {status}")
        
        self.results['power_analysis'] = power_results
        return power_results
    
    def create_visualizations(self):
        """可視化の作成"""
        print("\n" + "="*50)
        print("統計検定結果の可視化")
        print("="*50)
        
        output_dir = Path("outputs/phase2_figures")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # McNemar検定結果の可視化
        self.plot_mcnemar_results(output_dir)
        
        # 総合スコアの変化
        self.plot_composite_scores(output_dir)
        
        # 効果量の比較
        self.plot_effect_sizes(output_dir)
        
        # 検出力分析
        self.plot_power_analysis(output_dir)
        
        print(f"✓ 図表を {output_dir} に保存しました")
    
    def plot_mcnemar_results(self, output_dir):
        """McNemar検定結果の可視化"""
        if 'mcnemar_analysis' not in self.results:
            return
        
        # Q1とQ3の結果を統合
        all_results = {}
        all_results.update(self.results['mcnemar_analysis']['q1_results'])
        all_results.update(self.results['mcnemar_analysis']['q3_results'])
        
        if not all_results:
            return
        
        # データ準備
        items = list(all_results.keys())
        before_rates = [all_results[item]['before_true_rate'] * 100 for item in items]
        after_rates = [all_results[item]['after_true_rate'] * 100 for item in items]
        p_values = [all_results[item]['p_value'] for item in items]
        
        # 図の作成
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 前後比較
        x = np.arange(len(items))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, before_rates, width, label='Before', alpha=0.8, color='lightblue')
        bars2 = ax1.bar(x + width/2, after_rates, width, label='After', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('Items')
        ax1.set_ylabel('True Response Rate (%)')
        ax1.set_title('Before vs After Comparison with McNemar Test Results')
        ax1.set_xticks(x)
        ax1.set_xticklabels([item.replace('Q1_', '').replace('_Response', '') for item in items], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 有意性マーク
        for i, p_val in enumerate(p_values):
            if not np.isnan(p_val):
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
        
        # p値の分布
        valid_p_values = [p for p in p_values if not np.isnan(p)]
        if valid_p_values:
            ax2.bar(range(len(valid_p_values)), valid_p_values, alpha=0.7, color='steelblue')
            ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
            ax2.set_xlabel('Test Number')
            ax2.set_ylabel('p-value')
            ax2.set_title('McNemar Test p-values')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "mcnemar_results.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_composite_scores(self, output_dir):
        """総合スコア変化の可視化"""
        if 'composite_analysis' not in self.results:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, (category, result) in enumerate(self.results['composite_analysis'].items()):
            if 'error' in result:
                continue
            
            # ボックスプロット用データ準備（実際のデータが必要なため、平均値で代用）
            before_mean = result['before_mean']
            after_mean = result['after_mean']
            
            categories = ['Before', 'After']
            values = [before_mean, after_mean]
            
            axes[i].bar(categories, values, alpha=0.7, color=['lightblue', 'lightcoral'])
            axes[i].set_ylabel('Composite Score')
            axes[i].set_title(f'{category.upper()} Composite Score\n(p = {result["p_value"]:.4f}, d = {result["cohens_d"]:.3f})')
            axes[i].grid(True, alpha=0.3)
            
            # 誤差棒
            stds = [result['before_std'], result['after_std']]
            axes[i].errorbar(categories, values, yerr=stds, fmt='o', color='black', capsize=5)
            
            # 信頼区間を表示
            if result['ci_95_lower'] * result['ci_95_upper'] > 0:  # 0を含まない
                significance = "Significant"
                color = 'red'
            else:
                significance = "Not Significant"
                color = 'gray'
            
            axes[i].text(0.5, max(values) * 1.1, significance, 
                        ha='center', va='center', transform=axes[i].transData,
                        color=color, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / "composite_scores.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_effect_sizes(self, output_dir):
        """効果量の比較可視化"""
        if 'mcnemar_analysis' not in self.results:
            return
        
        # 効果量データの収集
        effect_sizes = {}
        
        for category in ['q1_results', 'q3_results']:
            for item, result in self.results['mcnemar_analysis'][category].items():
                if not np.isnan(result['effect_size']):
                    effect_sizes[item.replace('Q1_', '').replace('Q3_', '').replace('_Response', '')] = result['effect_size']
        
        if not effect_sizes:
            return
        
        # 図の作成
        items = list(effect_sizes.keys())
        values = list(effect_sizes.values())
        colors = ['red' if v < -0.2 else 'green' if v > 0.2 else 'gray' for v in values]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(items, values, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Small effect (+)')
        ax.axhline(y=-0.2, color='red', linestyle='--', alpha=0.5, label='Small effect (-)')
        ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.5, label='Medium effect (+)')
        ax.axhline(y=-0.5, color='red', linestyle=':', alpha=0.5, label='Medium effect (-)')
        
        ax.set_ylabel('Effect Size (Cohen\'s g)')
        ax.set_title('Effect Sizes for Each Item (McNemar Test)')
        ax.set_xticklabels(items, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "effect_sizes.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_power_analysis(self, output_dir):
        """検出力分析の可視化"""
        if 'power_analysis' not in self.results:
            return
        
        # 検出力データの準備
        effect_sizes = [0.2, 0.5, 0.8]
        current_powers = [self.results['power_analysis'][f'effect_{es}']['power'] for es in effect_sizes]
        required_ns = [self.results['power_analysis'][f'required_n_effect_{es}']['required_n'] for es in effect_sizes]
        current_n = len(self.paired_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 現在の検出力
        bars1 = ax1.bar(effect_sizes, current_powers, alpha=0.7, color='steelblue')
        ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Adequate Power (0.8)')
        ax1.set_xlabel('Effect Size (Cohen\'s d)')
        ax1.set_ylabel('Statistical Power')
        ax1.set_title(f'Current Statistical Power (n={current_n})')
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 値をバーの上に表示
        for bar, power in zip(bars1, current_powers):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{power:.3f}', ha='center', va='bottom')
        
        # 必要サンプルサイズ
        bars2 = ax2.bar(effect_sizes, required_ns, alpha=0.7, color='lightcoral')
        ax2.axhline(y=current_n, color='blue', linestyle='--', alpha=0.7, label=f'Current n={current_n}')
        ax2.set_xlabel('Effect Size (Cohen\'s d)')
        ax2.set_ylabel('Required Sample Size')
        ax2.set_title('Required Sample Size for Power = 0.8')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 値をバーの上に表示
        for bar, n in zip(bars2, required_ns):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(required_ns)*0.02, 
                    f'{n}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / "power_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self):
        """Phase 2 レポート生成"""
        print("\n" + "="*50)
        print("Phase 2 レポート生成")
        print("="*50)
        
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # JSON形式で詳細結果を保存
        with open(output_dir / "phase2_detailed_results.json", 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        
        # テキスト形式でサマリーレポートを生成
        report_content = self.create_summary_report()
        
        with open(output_dir / "phase2_summary_report.txt", 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✓ レポートを {output_dir} に保存しました")
        print(f"  - 詳細結果: phase2_detailed_results.json")
        print(f"  - サマリー: phase2_summary_report.txt")
        
        return report_content
    
    def create_summary_report(self):
        """サマリーレポートの作成"""
        report = []
        report.append("="*60)
        report.append("小学校出前授業アンケート Phase 2 統計的検証結果")
        report.append("="*60)
        report.append(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"有意水準: α = {self.alpha}")
        report.append("")
        
        # サンプルサイズ
        report.append("【サンプルサイズ】")
        report.append(f"ペア数: {len(self.paired_data)}")
        report.append("")
        
        # McNemar検定結果
        if 'mcnemar_analysis' in self.results:
            report.append("【McNemar検定結果】")
            
            for category, label in [('q1_results', 'Q1: 水溶液認識'), ('q3_results', 'Q3: お茶の理解度')]:
                report.append(f"\n{label}:")
                
                results = self.results['mcnemar_analysis'][category]
                significant_items = []
                
                for item, result in results.items():
                    change_direction = "改善" if result['change'] > 0 else "悪化" if result['change'] < 0 else "変化なし"
                    item_short = item.replace('Q1_', '').replace('Q3_', '').replace('_Response', '')
                    
                    report.append(f"  {item_short:15}: {result['before_true_rate']:.3f} → {result['after_true_rate']:.3f} ({change_direction})")
                    report.append(f"    p = {result['p_value']:.4f}, 効果量 = {result['effect_size']:.3f}")
                    
                    if result['significant']:
                        significant_items.append(item_short)
                
                if significant_items:
                    report.append(f"  有意な変化: {', '.join(significant_items)}")
                else:
                    report.append("  有意な変化なし")
            
            # 多重比較補正
            if 'multiple_testing' in self.results['mcnemar_analysis']:
                correction = self.results['mcnemar_analysis']['multiple_testing']
                if 'total_tests' in correction:
                    report.append(f"\n多重比較補正 (検定数: {correction['total_tests']}):")
                    
                    bonf_significant = sum(1 for result in correction['corrected_results'].values() 
                                         if result['bonferroni_significant'])
                    fdr_significant = sum(1 for result in correction['corrected_results'].values() 
                                        if result['fdr_significant'])
                    
                    report.append(f"  Bonferroni補正後有意: {bonf_significant}項目")
                    report.append(f"  FDR補正後有意: {fdr_significant}項目")
            
            report.append("")
        
        # 総合スコア分析
        if 'composite_analysis' in self.results:
            report.append("【総合スコア分析（対応のあるt検定）】")
            
            for category, result in self.results['composite_analysis'].items():
                if 'error' in result:
                    continue
                    
                category_label = "Q1総合" if "q1" in category else "Q3総合"
                change_direction = "改善" if result['mean_difference'] > 0 else "悪化" if result['mean_difference'] < 0 else "変化なし"
                
                report.append(f"\n{category_label}:")
                report.append(f"  授業前: {result['before_mean']:.2f} ± {result['before_std']:.2f}")
                report.append(f"  授業後: {result['after_mean']:.2f} ± {result['after_std']:.2f}")
                report.append(f"  平均差: {result['mean_difference']:.2f} [{result['ci_95_lower']:.2f}, {result['ci_95_upper']:.2f}]")
                report.append(f"  t({result['n_pairs']-1}) = {result['t_statistic']:.3f}, p = {result['p_value']:.4f}")
                report.append(f"  Cohen's d = {result['cohens_d']:.3f} ({result['effect_interpretation']})")
                report.append(f"  結果: {change_direction}, {'有意' if result['significant'] else '非有意'}")
            
            report.append("")
        
        # 検出力分析
        if 'power_analysis' in self.results:
            report.append("【検出力分析】")
            report.append(f"現在のサンプルサイズ (n={len(self.paired_data)}) での検出力:")
            
            for effect_size in [0.2, 0.5, 0.8]:
                power_key = f"effect_{effect_size}"
                if power_key in self.results['power_analysis']:
                    power = self.results['power_analysis'][power_key]['power']
                    adequate = "十分" if power >= 0.8 else "不足"
                    report.append(f"  効果量 d={effect_size}: {power:.3f} ({adequate})")
            
            report.append("")
        
        # 主要な結論
        report.append("【主要な結論】")
        
        # 有意な効果があったかどうか
        significant_count = 0
        if 'mcnemar_analysis' in self.results:
            for category in ['q1_results', 'q3_results']:
                for result in self.results['mcnemar_analysis'][category].values():
                    if result['significant']:
                        significant_count += 1
        
        if significant_count > 0:
            report.append(f"✓ {significant_count}項目で有意な変化を検出")
        else:
            report.append("• 統計的に有意な変化は検出されず")
        
        # 効果量の評価
        large_effects = 0
        if 'mcnemar_analysis' in self.results:
            for category in ['q1_results', 'q3_results']:
                for result in self.results['mcnemar_analysis'][category].values():
                    if abs(result.get('effect_size', 0)) > 0.5:
                        large_effects += 1
        
        if large_effects > 0:
            report.append(f"✓ {large_effects}項目で大きな効果量を確認")
        else:
            report.append("• 実質的に大きな効果は確認されず")
        
        # サンプルサイズの評価
        adequate_power = sum(1 for es in [0.2, 0.5, 0.8] 
                           if self.results.get('power_analysis', {}).get(f'effect_{es}', {}).get('adequate', False))
        
        if adequate_power >= 2:
            report.append("✓ 十分な検出力を確保")
        else:
            report.append("⚠️  検出力不足の可能性")
        
        report.append("")
        report.append("【Phase 3への推奨事項】")
        report.append("1. クラス間差異の分析実施")
        report.append("2. 個人差要因の探索")
        if significant_count == 0:
            report.append("3. 効果が小さい理由の質的分析")
        report.append("4. テキストデータによる補完的分析")
        
        return "\n".join(report)
    
    def run_complete_analysis(self):
        """Phase 2 完全分析実行"""
        print("小学校出前授業アンケート Phase 2: 教育効果の統計的検証")
        print("実行開始:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("="*60)
        
        try:
            # データ読み込み
            self.load_data()
            
            # McNemar検定分析
            self.mcnemar_test_analysis()
            
            # 総合スコア分析
            self.composite_score_analysis()
            
            # 検出力分析
            self.power_analysis()
            
            # 可視化作成
            self.create_visualizations()
            
            # レポート生成
            summary_report = self.generate_report()
            
            print("\n" + "="*60)
            print("Phase 2 分析完了!")
            print("="*60)
            print(summary_report)
            
            return self.results
            
        except Exception as e:
            print(f"❌ Phase 2 分析中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """メイン実行関数"""
    tester = Phase2StatisticalTester()
    results = tester.run_complete_analysis()
    return results

if __name__ == "__main__":
    main()