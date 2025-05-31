#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統計的検出力分析とサンプルサイズ計算
===================================

実施した分析の統計的妥当性を検証し、将来の研究設計への提言を提供。

機能:
- 各統計検定の事後検出力計算
- 効果量別の必要サンプルサイズ計算
- Bootstrap法による信頼区間推定
- 統計的有意性の妥当性評価
- 将来研究への設計提言

対象分析:
- χ²検定の検出力
- Mann-Whitney U検定の検出力
- 相関分析の検出力
- 機械学習モデルの統計的信頼性

Author: Claude Code Analysis (Power Analysis Implementation)
Date: 2025-05-31
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, pearsonr
import statsmodels.stats.power as smp
import statsmodels.stats.contingency_tables as smt

# Bootstrap分析用
from sklearn.utils import resample

# 警告抑制
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class PowerAnalysisEvaluator:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data" / "analysis"
        self.output_dir = self.project_root / "outputs" / "current" / "05_advanced_analysis"
        self.figures_dir = self.project_root / "outputs" / "figures" / "current" / "05_advanced_analysis"
        
        # 既存結果の読み込みパス
        self.sem_results_path = self.output_dir / "structural_equation_modeling_results.json"
        self.ml_results_path = self.output_dir / "machine_learning_prediction_results.json"
        
        # 出力ディレクトリ作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        
    def load_data_and_results(self):
        """データと既存分析結果の読み込み"""
        print("📊 データと既存結果読み込み中...")
        
        # データ読み込み
        before_path = self.data_dir / "before_excel_compliant.csv"
        after_path = self.data_dir / "after_excel_compliant.csv"
        
        if not before_path.exists() or not after_path.exists():
            raise FileNotFoundError("必要なデータファイルが見つかりません")
        
        self.before_data = pd.read_csv(before_path, encoding='utf-8')
        self.after_data = pd.read_csv(after_path, encoding='utf-8')
        
        print(f"✓ 授業前データ: {len(self.before_data)} 件")
        print(f"✓ 授業後データ: {len(self.after_data)} 件")
        
        # 既存分析結果読み込み
        self._load_existing_results()
        
    def _load_existing_results(self):
        """既存分析結果読み込み"""
        self.existing_results = {}
        
        # SEM結果
        if self.sem_results_path.exists():
            with open(self.sem_results_path, 'r', encoding='utf-8') as f:
                self.existing_results['sem'] = json.load(f)
            print("✓ SEM分析結果読み込み完了")
        
        # ML結果
        if self.ml_results_path.exists():
            with open(self.ml_results_path, 'r', encoding='utf-8') as f:
                self.existing_results['ml'] = json.load(f)
            print("✓ ML分析結果読み込み完了")
    
    def calculate_effect_sizes(self):
        """効果量計算"""
        print("\n📏 効果量計算中...")
        
        effect_sizes = {}
        
        # Q1総合スコア効果量計算
        effect_sizes['q1_composite'] = self._calculate_q1_effect_size()
        
        # Q3総合スコア効果量計算
        effect_sizes['q3_composite'] = self._calculate_q3_effect_size()
        
        # カテゴリカル変数の効果量（Cramér's V）
        effect_sizes['categorical'] = self._calculate_categorical_effect_sizes()
        
        self.results['effect_sizes'] = effect_sizes
        
        print("✓ 効果量計算完了")
        
    def _calculate_q1_effect_size(self):
        """Q1総合スコア効果量計算"""
        # Q1スコア計算
        q1_before_cols = ['Q1_Saltwater_Response', 'Q1_Sugarwater_Response', 'Q1_Muddywater_Response', 
                         'Q1_Ink_Response', 'Q1_MisoSoup_Response', 'Q1_SoySauce_Response']
        q1_after_cols = ['Q1_Saltwater', 'Q1_Sugarwater', 'Q1_Muddywater',
                        'Q1_Ink', 'Q1_MisoSoup', 'Q1_SoySauce']
        
        before_scores = self.before_data[q1_before_cols].sum(axis=1)
        after_scores = self.after_data[q1_after_cols].sum(axis=1)
        
        # Cohen's d計算
        mean_diff = after_scores.mean() - before_scores.mean()
        pooled_std = np.sqrt(((len(before_scores) - 1) * before_scores.var() + 
                             (len(after_scores) - 1) * after_scores.var()) / 
                            (len(before_scores) + len(after_scores) - 2))
        
        cohens_d = mean_diff / pooled_std
        
        # Mann-Whitney U検定
        u_stat, u_p = mannwhitneyu(before_scores, after_scores, alternative='two-sided')
        
        # 効果量r
        n_total = len(before_scores) + len(after_scores)
        z_score = stats.norm.ppf(u_p/2)  # 近似
        effect_r = abs(z_score) / np.sqrt(n_total)
        
        return {
            'before_mean': before_scores.mean(),
            'after_mean': after_scores.mean(),
            'before_std': before_scores.std(),
            'after_std': after_scores.std(),
            'mean_difference': mean_diff,
            'cohens_d': cohens_d,
            'mann_whitney_u': u_stat,
            'mann_whitney_p': u_p,
            'effect_size_r': effect_r,
            'sample_size_before': len(before_scores),
            'sample_size_after': len(after_scores)
        }
    
    def _calculate_q3_effect_size(self):
        """Q3総合スコア効果量計算"""
        # Q3スコア計算
        q3_before_cols = ['Q3_TeaLeavesDissolve', 'Q3_TeaComponentsDissolve']
        q3_after_cols = ['Q3_TeaLeaves_DissolveInWater', 'Q3_TeaComponents_DissolveInWater']
        
        before_scores = self.before_data[q3_before_cols].sum(axis=1)
        after_scores = self.after_data[q3_after_cols].sum(axis=1)
        
        # Cohen's d計算
        mean_diff = after_scores.mean() - before_scores.mean()
        pooled_std = np.sqrt(((len(before_scores) - 1) * before_scores.var() + 
                             (len(after_scores) - 1) * after_scores.var()) / 
                            (len(before_scores) + len(after_scores) - 2))
        
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # Mann-Whitney U検定
        u_stat, u_p = mannwhitneyu(before_scores, after_scores, alternative='two-sided')
        
        return {
            'before_mean': before_scores.mean(),
            'after_mean': after_scores.mean(),
            'before_std': before_scores.std(),
            'after_std': after_scores.std(),
            'mean_difference': mean_diff,
            'cohens_d': cohens_d,
            'mann_whitney_u': u_stat,
            'mann_whitney_p': u_p,
            'sample_size_before': len(before_scores),
            'sample_size_after': len(after_scores)
        }
    
    def _calculate_categorical_effect_sizes(self):
        """カテゴリカル変数効果量計算"""
        categorical_effects = {}
        
        # Q1各項目のCramér's V
        q1_items = [
            ('Saltwater', 'Q1_Saltwater_Response', 'Q1_Saltwater'),
            ('Sugarwater', 'Q1_Sugarwater_Response', 'Q1_Sugarwater'),
            ('Muddywater', 'Q1_Muddywater_Response', 'Q1_Muddywater'),
            ('Ink', 'Q1_Ink_Response', 'Q1_Ink'),
            ('MisoSoup', 'Q1_MisoSoup_Response', 'Q1_MisoSoup'),
            ('SoySauce', 'Q1_SoySauce_Response', 'Q1_SoySauce')
        ]
        
        for item_name, before_col, after_col in q1_items:
            if before_col in self.before_data.columns and after_col in self.after_data.columns:
                # クロス集計表作成
                before_counts = self.before_data[before_col].value_counts()
                after_counts = self.after_data[after_col].value_counts()
                
                # 共通のカテゴリで集計表作成
                all_categories = sorted(set(before_counts.index) | set(after_counts.index))
                
                contingency_table = []
                for cat in all_categories:
                    before_count = before_counts.get(cat, 0)
                    after_count = after_counts.get(cat, 0)
                    contingency_table.append([before_count, after_count])
                
                contingency_table = np.array(contingency_table).T
                
                # χ²検定とCramér's V
                if contingency_table.sum() > 0:
                    try:
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                        n = contingency_table.sum()
                        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                        
                        categorical_effects[item_name] = {
                            'chi2': chi2,
                            'p_value': p_value,
                            'degrees_of_freedom': dof,
                            'cramers_v': cramers_v,
                            'sample_size': n,
                            'contingency_table': contingency_table.tolist()
                        }
                    except ValueError as e:
                        categorical_effects[item_name] = {'error': str(e)}
        
        return categorical_effects
    
    def calculate_post_hoc_power(self):
        """事後検出力計算"""
        print("\n⚡ 事後検出力計算中...")
        
        power_results = {}
        
        # t検定の検出力（Q1, Q3総合スコア）
        power_results['t_tests'] = self._calculate_t_test_power()
        
        # χ²検定の検出力
        power_results['chi2_tests'] = self._calculate_chi2_power()
        
        # 相関分析の検出力
        power_results['correlation_tests'] = self._calculate_correlation_power()
        
        self.results['power_analysis'] = power_results
        
        print("✓ 事後検出力計算完了")
    
    def _calculate_t_test_power(self):
        """t検定検出力計算"""
        t_test_power = {}
        
        # Q1総合スコア
        if 'q1_composite' in self.results['effect_sizes']:
            q1_effect = self.results['effect_sizes']['q1_composite']
            
            # 独立t検定の検出力
            power_q1 = smp.ttest_power(
                effect_size=abs(q1_effect['cohens_d']),
                nobs=q1_effect['sample_size_before'],
                alpha=0.05,
                alternative='two-sided'
            )
            
            t_test_power['Q1_composite'] = {
                'effect_size': q1_effect['cohens_d'],
                'power': power_q1,
                'sample_size_before': q1_effect['sample_size_before'],
                'sample_size_after': q1_effect['sample_size_after'],
                'interpretation': self._interpret_power(power_q1)
            }
        
        # Q3総合スコア
        if 'q3_composite' in self.results['effect_sizes']:
            q3_effect = self.results['effect_sizes']['q3_composite']
            
            power_q3 = smp.ttest_power(
                effect_size=abs(q3_effect['cohens_d']),
                nobs=q3_effect['sample_size_before'],
                alpha=0.05,
                alternative='two-sided'
            )
            
            t_test_power['Q3_composite'] = {
                'effect_size': q3_effect['cohens_d'],
                'power': power_q3,
                'sample_size_before': q3_effect['sample_size_before'],
                'sample_size_after': q3_effect['sample_size_after'],
                'interpretation': self._interpret_power(power_q3)
            }
        
        return t_test_power
    
    def _calculate_chi2_power(self):
        """χ²検定検出力計算"""
        chi2_power = {}
        
        if 'categorical' in self.results['effect_sizes']:
            for item_name, effect_data in self.results['effect_sizes']['categorical'].items():
                if 'error' not in effect_data:
                    # χ²検定の検出力計算（近似）
                    cramers_v = effect_data['cramers_v']
                    n = effect_data['sample_size']
                    df = effect_data['degrees_of_freedom']
                    
                    # 効果量をw（Cohenのw）に変換
                    w = cramers_v * np.sqrt(df)
                    
                    # 検出力計算
                    try:
                        power = smp.GofChisquarePower().solve_power(
                            effect_size=w,
                            nobs=n,
                            alpha=0.05,
                            power=None
                        )
                        
                        chi2_power[item_name] = {
                            'cramers_v': cramers_v,
                            'cohens_w': w,
                            'power': power,
                            'sample_size': n,
                            'p_value': effect_data['p_value'],
                            'interpretation': self._interpret_power(power)
                        }
                    except Exception as e:
                        chi2_power[item_name] = {'error': f'検出力計算エラー: {e}'}
        
        return chi2_power
    
    def _calculate_correlation_power(self):
        """相関分析検出力計算"""
        correlation_power = {}
        
        # 授業後データでの相関分析
        after_data = self.after_data.copy()
        
        # Q1総合スコア計算
        q1_cols = ['Q1_Saltwater', 'Q1_Sugarwater', 'Q1_Muddywater',
                   'Q1_Ink', 'Q1_MisoSoup', 'Q1_SoySauce']
        after_data['Q1_total'] = after_data[q1_cols].sum(axis=1)
        
        # 相関分析対象
        correlation_pairs = [
            ('Q1_total', 'Q4_ExperimentInterestRating'),
            ('Q1_total', 'Q6_DissolvingUnderstandingRating'),
            ('Q4_ExperimentInterestRating', 'Q5_NewLearningsRating'),
            ('Q5_NewLearningsRating', 'Q6_DissolvingUnderstandingRating')
        ]
        
        for var1, var2 in correlation_pairs:
            if var1 in after_data.columns and var2 in after_data.columns:
                # 欠損値除去
                data_pair = after_data[[var1, var2]].dropna()
                
                if len(data_pair) > 5:  # 最小サンプルサイズチェック
                    r, p_value = pearsonr(data_pair[var1], data_pair[var2])
                    
                    # 相関の検出力計算
                    try:
                        power = smp.ttest_power(
                            effect_size=abs(r),  # 相関係数を効果量として使用
                            nobs=len(data_pair),
                            alpha=0.05,
                            alternative='two-sided'
                        )
                        
                        correlation_power[f"{var1}_vs_{var2}"] = {
                            'correlation': r,
                            'p_value': p_value,
                            'power': power,
                            'sample_size': len(data_pair),
                            'interpretation': self._interpret_power(power)
                        }
                    except Exception as e:
                        correlation_power[f"{var1}_vs_{var2}"] = {'error': f'検出力計算エラー: {e}'}
        
        return correlation_power
    
    def _interpret_power(self, power):
        """検出力解釈"""
        if power >= 0.8:
            return "十分な検出力"
        elif power >= 0.6:
            return "中程度の検出力"
        else:
            return "検出力不足"
    
    def calculate_required_sample_sizes(self):
        """必要サンプルサイズ計算"""
        print("\n📊 必要サンプルサイズ計算中...")
        
        sample_size_recommendations = {}
        
        # 異なる効果量での必要サンプルサイズ
        effect_sizes = [0.2, 0.5, 0.8]  # 小・中・大効果
        desired_power = 0.8
        alpha = 0.05
        
        for effect_size in effect_sizes:
            # t検定
            try:
                n_t_test = smp.TTestPower().solve_power(
                    effect_size=effect_size,
                    power=desired_power,
                    alpha=alpha,
                    alternative='two-sided'
                )
            except:
                n_t_test = None
            
            # χ²検定（自由度1として近似）
            try:
                n_chi2_test = smp.GofChisquarePower().solve_power(
                    effect_size=effect_size,
                    power=desired_power,
                    alpha=alpha,
                    n_bins=2
                )
            except:
                n_chi2_test = None
            
            # 相関分析
            try:
                n_correlation = smp.TTestPower().solve_power(
                    effect_size=effect_size,
                    power=desired_power,
                    alpha=alpha,
                    alternative='two-sided'
                )
            except:
                n_correlation = None
            
            sample_size_recommendations[f"effect_size_{effect_size}"] = {
                'effect_size': effect_size,
                'effect_interpretation': self._interpret_effect_size(effect_size),
                't_test_n_per_group': n_t_test,
                'chi2_test_n_total': n_chi2_test,
                'correlation_n_total': n_correlation,
                'power': desired_power,
                'alpha': alpha
            }
        
        self.results['sample_size_recommendations'] = sample_size_recommendations
        
        print("✓ 必要サンプルサイズ計算完了")
    
    def _interpret_effect_size(self, effect_size):
        """効果量解釈"""
        if effect_size >= 0.8:
            return "大効果"
        elif effect_size >= 0.5:
            return "中効果"
        elif effect_size >= 0.2:
            return "小効果"
        else:
            return "効果なし/極小"
    
    def bootstrap_confidence_intervals(self):
        """Bootstrap法による信頼区間推定"""
        print("\n🔄 Bootstrap信頼区間推定中...")
        
        bootstrap_results = {}
        
        # Q1総合スコアのBootstrap
        bootstrap_results['q1_composite'] = self._bootstrap_q1_difference()
        
        # Q3総合スコアのBootstrap
        bootstrap_results['q3_composite'] = self._bootstrap_q3_difference()
        
        self.results['bootstrap_analysis'] = bootstrap_results
        
        print("✓ Bootstrap分析完了")
    
    def _bootstrap_q1_difference(self, n_bootstrap=1000):
        """Q1スコア差のBootstrap信頼区間"""
        # Q1スコア計算
        q1_before_cols = ['Q1_Saltwater_Response', 'Q1_Sugarwater_Response', 'Q1_Muddywater_Response', 
                         'Q1_Ink_Response', 'Q1_MisoSoup_Response', 'Q1_SoySauce_Response']
        q1_after_cols = ['Q1_Saltwater', 'Q1_Sugarwater', 'Q1_Muddywater',
                        'Q1_Ink', 'Q1_MisoSoup', 'Q1_SoySauce']
        
        before_scores = self.before_data[q1_before_cols].sum(axis=1).values
        after_scores = self.after_data[q1_after_cols].sum(axis=1).values
        
        # Bootstrap標本抽出
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            before_sample = resample(before_scores, n_samples=len(before_scores))
            after_sample = resample(after_scores, n_samples=len(after_scores))
            diff = after_sample.mean() - before_sample.mean()
            bootstrap_diffs.append(diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # 信頼区間計算
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)
        
        return {
            'observed_difference': after_scores.mean() - before_scores.mean(),
            'bootstrap_mean': bootstrap_diffs.mean(),
            'bootstrap_std': bootstrap_diffs.std(),
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'n_bootstrap': n_bootstrap,
            'significant': not (ci_lower <= 0 <= ci_upper)  # 0を含まなければ有意
        }
    
    def _bootstrap_q3_difference(self, n_bootstrap=1000):
        """Q3スコア差のBootstrap信頼区間"""
        # Q3スコア計算
        q3_before_cols = ['Q3_TeaLeavesDissolve', 'Q3_TeaComponentsDissolve']
        q3_after_cols = ['Q3_TeaLeaves_DissolveInWater', 'Q3_TeaComponents_DissolveInWater']
        
        before_scores = self.before_data[q3_before_cols].sum(axis=1).values
        after_scores = self.after_data[q3_after_cols].sum(axis=1).values
        
        # Bootstrap標本抽出
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            before_sample = resample(before_scores, n_samples=len(before_scores))
            after_sample = resample(after_scores, n_samples=len(after_scores))
            diff = after_sample.mean() - before_sample.mean()
            bootstrap_diffs.append(diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # 信頼区間計算
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)
        
        return {
            'observed_difference': after_scores.mean() - before_scores.mean(),
            'bootstrap_mean': bootstrap_diffs.mean(),
            'bootstrap_std': bootstrap_diffs.std(),
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'n_bootstrap': n_bootstrap,
            'significant': not (ci_lower <= 0 <= ci_upper)
        }
    
    def create_visualizations(self):
        """可視化作成"""
        print("\n📊 可視化作成中...")
        
        # 1. 検出力分析結果
        self._create_power_analysis_plot()
        
        # 2. 効果量可視化
        self._create_effect_size_plot()
        
        # 3. サンプルサイズ推奨
        self._create_sample_size_recommendation_plot()
        
        # 4. Bootstrap信頼区間
        self._create_bootstrap_plot()
        
    def _create_power_analysis_plot(self):
        """検出力分析プロット"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # t検定検出力
        if 'power_analysis' in self.results and 't_tests' in self.results['power_analysis']:
            t_tests = self.results['power_analysis']['t_tests']
            
            test_names = list(t_tests.keys())
            powers = [t_tests[name]['power'] for name in test_names]
            effect_sizes = [abs(t_tests[name]['effect_size']) for name in test_names]
            
            axes[0,0].bar(test_names, powers, color='skyblue', alpha=0.7)
            axes[0,0].axhline(y=0.8, color='red', linestyle='--', label='推奨検出力(0.8)')
            axes[0,0].set_title('t検定の検出力')
            axes[0,0].set_ylabel('検出力')
            axes[0,0].legend()
            axes[0,0].set_ylim(0, 1)
            
            # 効果量も表示
            for i, (power, es) in enumerate(zip(powers, effect_sizes)):
                axes[0,0].text(i, power + 0.02, f'ES={es:.3f}', ha='center', fontsize=8)
        
        # χ²検定検出力
        if 'chi2_tests' in self.results['power_analysis']:
            chi2_tests = self.results['power_analysis']['chi2_tests']
            
            valid_tests = {k: v for k, v in chi2_tests.items() if 'error' not in v}
            if valid_tests:
                test_names = list(valid_tests.keys())
                powers = [valid_tests[name]['power'] for name in test_names]
                
                axes[0,1].bar(test_names, powers, color='lightcoral', alpha=0.7)
                axes[0,1].axhline(y=0.8, color='red', linestyle='--', label='推奨検出力(0.8)')
                axes[0,1].set_title('χ²検定の検出力')
                axes[0,1].set_ylabel('検出力')
                axes[0,1].legend()
                axes[0,1].set_ylim(0, 1)
                axes[0,1].tick_params(axis='x', rotation=45)
        
        # 相関分析検出力
        if 'correlation_tests' in self.results['power_analysis']:
            corr_tests = self.results['power_analysis']['correlation_tests']
            
            valid_tests = {k: v for k, v in corr_tests.items() if 'error' not in v}
            if valid_tests:
                test_names = [name.replace('_vs_', ' - ') for name in valid_tests.keys()]
                powers = [valid_tests[list(valid_tests.keys())[i]]['power'] for i in range(len(test_names))]
                
                axes[1,0].bar(test_names, powers, color='lightgreen', alpha=0.7)
                axes[1,0].axhline(y=0.8, color='red', linestyle='--', label='推奨検出力(0.8)')
                axes[1,0].set_title('相関分析の検出力')
                axes[1,0].set_ylabel('検出力')
                axes[1,0].legend()
                axes[1,0].set_ylim(0, 1)
                axes[1,0].tick_params(axis='x', rotation=45)
        
        # 検出力サマリー
        all_powers = []
        if 't_tests' in self.results['power_analysis']:
            all_powers.extend([v['power'] for v in self.results['power_analysis']['t_tests'].values()])
        
        if all_powers:
            axes[1,1].hist(all_powers, bins=10, alpha=0.7, color='purple')
            axes[1,1].axvline(x=0.8, color='red', linestyle='--', label='推奨検出力(0.8)')
            axes[1,1].set_title('検出力分布')
            axes[1,1].set_xlabel('検出力')
            axes[1,1].set_ylabel('頻度')
            axes[1,1].legend()
        
        plt.tight_layout()
        
        output_path = self.figures_dir / "power_analysis_results.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 検出力分析図保存: {output_path}")
    
    def _create_effect_size_plot(self):
        """効果量可視化"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Cohen's d（連続変数）
        if 'effect_sizes' in self.results:
            es_data = self.results['effect_sizes']
            
            if 'q1_composite' in es_data and 'q3_composite' in es_data:
                variables = ['Q1総合スコア', 'Q3総合スコア']
                cohens_d = [es_data['q1_composite']['cohens_d'], es_data['q3_composite']['cohens_d']]
                
                colors = ['blue' if d > 0 else 'red' for d in cohens_d]
                bars = axes[0].bar(variables, cohens_d, color=colors, alpha=0.7)
                axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                axes[0].axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='小効果(0.2)')
                axes[0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='中効果(0.5)')
                axes[0].axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='大効果(0.8)')
                axes[0].set_title("Cohen's d (効果量)")
                axes[0].set_ylabel("Cohen's d")
                axes[0].legend()
                
                # 値をバーの上に表示
                for bar, d in zip(bars, cohens_d):
                    height = bar.get_height()
                    axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.05,
                               f'{d:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        # Cramér's V（カテゴリカル変数）
        if 'categorical' in self.results['effect_sizes']:
            cat_data = self.results['effect_sizes']['categorical']
            valid_items = {k: v for k, v in cat_data.items() if 'error' not in v}
            
            if valid_items:
                items = list(valid_items.keys())
                cramers_v = [valid_items[item]['cramers_v'] for item in items]
                
                axes[1].bar(items, cramers_v, color='skyblue', alpha=0.7)
                axes[1].axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='小効果(0.1)')
                axes[1].axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='中効果(0.3)')
                axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='大効果(0.5)')
                axes[1].set_title("Cramér's V (カテゴリカル効果量)")
                axes[1].set_ylabel("Cramér's V")
                axes[1].legend()
                axes[1].tick_params(axis='x', rotation=45)
        
        # 効果量比較サマリー
        effect_summary_data = []
        effect_labels = []
        
        if 'effect_sizes' in self.results:
            es_data = self.results['effect_sizes']
            
            if 'q1_composite' in es_data:
                effect_summary_data.append(abs(es_data['q1_composite']['cohens_d']))
                effect_labels.append('Q1 (Cohen\'s d)')
            
            if 'q3_composite' in es_data:
                effect_summary_data.append(abs(es_data['q3_composite']['cohens_d']))
                effect_labels.append('Q3 (Cohen\'s d)')
            
            if 'categorical' in es_data:
                for item, data in es_data['categorical'].items():
                    if 'error' not in data:
                        effect_summary_data.append(data['cramers_v'])
                        effect_labels.append(f'{item} (Cramér\'s V)')
        
        if effect_summary_data:
            axes[2].barh(effect_labels, effect_summary_data, alpha=0.7)
            axes[2].axvline(x=0.2, color='green', linestyle='--', alpha=0.5, label='小効果')
            axes[2].axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='中効果')
            axes[2].axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='大効果')
            axes[2].set_title('効果量一覧')
            axes[2].set_xlabel('効果量')
            axes[2].legend()
        
        plt.tight_layout()
        
        output_path = self.figures_dir / "effect_sizes_summary.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 効果量図保存: {output_path}")
    
    def _create_sample_size_recommendation_plot(self):
        """サンプルサイズ推奨プロット"""
        if 'sample_size_recommendations' not in self.results:
            return
        
        recommendations = self.results['sample_size_recommendations']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 効果量別必要サンプルサイズ
        effect_sizes = []
        t_test_ns = []
        
        for key, data in recommendations.items():
            effect_sizes.append(data['effect_size'])
            t_test_ns.append(data['t_test_n_per_group'])
        
        ax1.plot(effect_sizes, t_test_ns, 'o-', linewidth=2, markersize=8, label='t検定（群あたり）')
        ax1.axhline(y=99, color='red', linestyle='--', alpha=0.7, label='現在のサンプルサイズ')
        ax1.set_xlabel('効果量 (Cohen\'s d)')
        ax1.set_ylabel('必要サンプルサイズ（群あたり）')
        ax1.set_title('効果量別必要サンプルサイズ\n(検出力=0.8, α=0.05)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 現在の分析の検出力評価
        current_powers = []
        current_labels = []
        
        if 'power_analysis' in self.results:
            if 't_tests' in self.results['power_analysis']:
                for test_name, test_data in self.results['power_analysis']['t_tests'].items():
                    current_powers.append(test_data['power'])
                    current_labels.append(test_name)
        
        if current_powers:
            colors = ['green' if p >= 0.8 else 'orange' if p >= 0.6 else 'red' for p in current_powers]
            bars = ax2.bar(current_labels, current_powers, color=colors, alpha=0.7)
            ax2.axhline(y=0.8, color='red', linestyle='--', label='推奨検出力(0.8)')
            ax2.set_title('現在の分析の検出力評価')
            ax2.set_ylabel('検出力')
            ax2.legend()
            ax2.set_ylim(0, 1)
            
            # 値をバーの上に表示
            for bar, power in zip(bars, current_powers):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{power:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        output_path = self.figures_dir / "sample_size_recommendations.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ サンプルサイズ推奨図保存: {output_path}")
    
    def _create_bootstrap_plot(self):
        """Bootstrap信頼区間プロット"""
        if 'bootstrap_analysis' not in self.results:
            return
        
        bootstrap_data = self.results['bootstrap_analysis']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Q1 Bootstrap分布
        if 'q1_composite' in bootstrap_data:
            q1_data = bootstrap_data['q1_composite']
            
            # 分布をプロット（ヒストグラム）
            # Bootstrap分布は実際には保存していないので、正規分布で近似
            x_range = np.linspace(q1_data['ci_95_lower'] - 0.5, q1_data['ci_95_upper'] + 0.5, 100)
            y_values = stats.norm.pdf(x_range, q1_data['bootstrap_mean'], q1_data['bootstrap_std'])
            
            axes[0].plot(x_range, y_values, 'b-', linewidth=2, label='Bootstrap分布')
            axes[0].axvline(q1_data['observed_difference'], color='red', linestyle='-', 
                          linewidth=2, label=f'観測値 ({q1_data["observed_difference"]:.3f})')
            axes[0].axvline(q1_data['ci_95_lower'], color='green', linestyle='--', 
                          label=f'95%CI下限 ({q1_data["ci_95_lower"]:.3f})')
            axes[0].axvline(q1_data['ci_95_upper'], color='green', linestyle='--', 
                          label=f'95%CI上限 ({q1_data["ci_95_upper"]:.3f})')
            axes[0].axvline(0, color='black', linestyle=':', alpha=0.5, label='差なし')
            
            axes[0].set_title('Q1総合スコア差の Bootstrap信頼区間')
            axes[0].set_xlabel('平均スコア差')
            axes[0].set_ylabel('確率密度')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Q3 Bootstrap分布
        if 'q3_composite' in bootstrap_data:
            q3_data = bootstrap_data['q3_composite']
            
            x_range = np.linspace(q3_data['ci_95_lower'] - 0.5, q3_data['ci_95_upper'] + 0.5, 100)
            y_values = stats.norm.pdf(x_range, q3_data['bootstrap_mean'], q3_data['bootstrap_std'])
            
            axes[1].plot(x_range, y_values, 'b-', linewidth=2, label='Bootstrap分布')
            axes[1].axvline(q3_data['observed_difference'], color='red', linestyle='-', 
                          linewidth=2, label=f'観測値 ({q3_data["observed_difference"]:.3f})')
            axes[1].axvline(q3_data['ci_95_lower'], color='green', linestyle='--', 
                          label=f'95%CI下限 ({q3_data["ci_95_lower"]:.3f})')
            axes[1].axvline(q3_data['ci_95_upper'], color='green', linestyle='--', 
                          label=f'95%CI上限 ({q3_data["ci_95_upper"]:.3f})')
            axes[1].axvline(0, color='black', linestyle=':', alpha=0.5, label='差なし')
            
            axes[1].set_title('Q3総合スコア差の Bootstrap信頼区間')
            axes[1].set_xlabel('平均スコア差')
            axes[1].set_ylabel('確率密度')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.figures_dir / "bootstrap_confidence_intervals.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Bootstrap信頼区間図保存: {output_path}")
    
    def generate_recommendations(self):
        """統計的提言生成"""
        print("\n📝 統計的提言生成中...")
        
        recommendations = {
            'current_analysis_evaluation': self._evaluate_current_analysis(),
            'future_study_design': self._design_future_study(),
            'statistical_best_practices': self._statistical_best_practices(),
            'reporting_guidelines': self._reporting_guidelines()
        }
        
        self.results['recommendations'] = recommendations
        
        print("✓ 統計的提言生成完了")
    
    def _evaluate_current_analysis(self):
        """現在の分析評価"""
        evaluation = {
            'strengths': [
                "適切な独立群比較設計の採用",
                "多重比較補正の実施",
                "効果量の報告",
                "非パラメトリック検定の併用"
            ],
            'limitations': [
                f"限定的なサンプルサイズ（N={len(self.before_data)}, {len(self.after_data)}）",
                "一部検定での検出力不足",
                "個人追跡不可による制約",
                "観察研究としての因果推論の限界"
            ],
            'power_summary': self._summarize_power_results()
        }
        
        return evaluation
    
    def _summarize_power_results(self):
        """検出力結果要約"""
        if 'power_analysis' not in self.results:
            return "検出力分析未実施"
        
        power_data = self.results['power_analysis']
        summary = {}
        
        # t検定検出力要約
        if 't_tests' in power_data:
            t_powers = [test['power'] for test in power_data['t_tests'].values()]
            summary['t_tests'] = {
                'mean_power': np.mean(t_powers),
                'adequate_power_count': sum(1 for p in t_powers if p >= 0.8),
                'total_tests': len(t_powers)
            }
        
        return summary
    
    def _design_future_study(self):
        """将来研究設計提言"""
        future_design = {
            'recommended_sample_size': self._recommend_sample_size(),
            'design_improvements': [
                "個人識別可能なデータ収集システム導入",
                "ランダム割付による実験デザイン",
                "統制群の設定",
                "長期追跡調査の実施",
                "多変量調整による交絡制御"
            ],
            'data_collection_enhancements': [
                "より詳細な背景変数収集",
                "テキストデータの体系的収集",
                "授業実施条件の標準化",
                "教師要因の統制・測定"
            ],
            'analysis_strategy': [
                "階層線形モデル（HLM）の適用",
                "傾向スコアマッチング",
                "構造方程式モデリングの発展",
                "機械学習手法との統合"
            ]
        }
        
        return future_design
    
    def _recommend_sample_size(self):
        """推奨サンプルサイズ"""
        if 'sample_size_recommendations' not in self.results:
            return "サンプルサイズ計算未実施"
        
        recommendations = self.results['sample_size_recommendations']
        
        # 中効果（0.5）での推奨サンプルサイズ
        medium_effect = recommendations.get('effect_size_0.5', {})
        
        return {
            'target_effect_size': 0.5,
            'recommended_n_per_group': medium_effect.get('t_test_n_per_group', 'N/A'),
            'current_n_per_group': len(self.before_data),
            'improvement_needed': medium_effect.get('t_test_n_per_group', 0) > len(self.before_data)
        }
    
    def _statistical_best_practices(self):
        """統計的ベストプラクティス"""
        practices = {
            'effect_size_reporting': [
                "すべての検定で効果量を報告",
                "実質的意味のある効果サイズの閾値設定",
                "信頼区間の併記"
            ],
            'multiple_testing': [
                "事前の検定計画立案",
                "適切な多重比較補正法選択",
                "探索的分析と確認的分析の区別"
            ],
            'model_validation': [
                "統計的前提条件の確認",
                "ロバスト性チェック",
                "感度分析の実施"
            ]
        }
        
        return practices
    
    def _reporting_guidelines(self):
        """報告ガイドライン"""
        guidelines = {
            'required_elements': [
                "サンプルサイズとその根拠",
                "効果量と信頼区間",
                "検出力または検出力分析結果",
                "多重比較補正の詳細",
                "分析の限界と解釈上の注意"
            ],
            'transparency_measures': [
                "事前分析計画の公開",
                "データの利用可能性",
                "分析コードの共有",
                "結果の再現性確保"
            ],
            'interpretation_cautions': [
                "因果推論の限界明記",
                "一般化可能性の検討",
                "実践的意義の評価",
                "統計的有意性と実質的意義の区別"
            ]
        }
        
        return guidelines
    
    def save_results(self):
        """結果保存"""
        print("\n💾 結果保存中...")
        
        # 詳細結果
        detailed_results = {
            'metadata': {
                'analysis_type': 'Power Analysis and Statistical Validation',
                'generated_at': datetime.now().isoformat(),
                'sample_size_before': len(self.before_data),
                'sample_size_after': len(self.after_data)
            },
            'effect_sizes': self.results.get('effect_sizes', {}),
            'power_analysis': self.results.get('power_analysis', {}),
            'sample_size_recommendations': self.results.get('sample_size_recommendations', {}),
            'bootstrap_analysis': self.results.get('bootstrap_analysis', {}),
            'recommendations': self.results.get('recommendations', {})
        }
        
        # JSON保存
        output_path = self.output_dir / "power_analysis_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✓ 詳細結果保存: {output_path}")
        
        # 要約レポート作成
        self._create_summary_report(detailed_results)
    
    def _create_summary_report(self, detailed_results):
        """要約レポート作成"""
        report_lines = [
            "# 統計的検出力分析レポート",
            "## 小学校出前授業アンケート - 統計的妥当性検証",
            "",
            f"**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**サンプルサイズ**: 授業前 {detailed_results['metadata']['sample_size_before']} 件, 授業後 {detailed_results['metadata']['sample_size_after']} 件",
            "",
            "## 分析概要",
            "",
            "実施した統計分析の検出力を評価し、将来の研究設計への提言を提供。",
            "効果量、Bootstrap信頼区間、必要サンプルサイズを包括的に検討。",
            ""
        ]
        
        # 効果量サマリー
        if 'effect_sizes' in detailed_results:
            report_lines.extend([
                "## 効果量サマリー",
                ""
            ])
            
            effect_sizes = detailed_results['effect_sizes']
            
            if 'q1_composite' in effect_sizes:
                q1 = effect_sizes['q1_composite']
                report_lines.append(f"**Q1総合スコア**: Cohen's d = {q1['cohens_d']:.3f}")
            
            if 'q3_composite' in effect_sizes:
                q3 = effect_sizes['q3_composite']
                report_lines.append(f"**Q3総合スコア**: Cohen's d = {q3['cohens_d']:.3f}")
            
            report_lines.append("")
        
        # 検出力評価
        if 'power_analysis' in detailed_results:
            report_lines.extend([
                "## 検出力評価",
                ""
            ])
            
            power_data = detailed_results['power_analysis']
            
            if 't_tests' in power_data:
                report_lines.append("### t検定の検出力")
                report_lines.append("")
                
                for test_name, test_data in power_data['t_tests'].items():
                    status = test_data['interpretation']
                    report_lines.append(f"- **{test_name}**: {test_data['power']:.3f} ({status})")
                
                report_lines.append("")
        
        # サンプルサイズ推奨
        if 'sample_size_recommendations' in detailed_results:
            report_lines.extend([
                "## 将来研究への推奨サンプルサイズ",
                "",
                "| 効果量 | 効果の解釈 | 必要サンプルサイズ（群あたり） | 検出力 |",
                "|--------|------------|--------------------------------|--------|"
            ])
            
            for key, rec in detailed_results['sample_size_recommendations'].items():
                if 'effect_size' in rec:
                    report_lines.append(
                        f"| {rec['effect_size']} | {rec['effect_interpretation']} | "
                        f"{rec['t_test_n_per_group']:.0f} | {rec['power']} |"
                    )
            
            report_lines.append("")
        
        # Bootstrap信頼区間
        if 'bootstrap_analysis' in detailed_results:
            report_lines.extend([
                "## Bootstrap信頼区間（95%）",
                ""
            ])
            
            bootstrap_data = detailed_results['bootstrap_analysis']
            
            for variable, data in bootstrap_data.items():
                if 'ci_95_lower' in data:
                    significance = "有意" if data['significant'] else "非有意"
                    report_lines.append(
                        f"**{variable}**: [{data['ci_95_lower']:.3f}, {data['ci_95_upper']:.3f}] ({significance})"
                    )
            
            report_lines.append("")
        
        # 提言
        if 'recommendations' in detailed_results:
            recs = detailed_results['recommendations']
            
            report_lines.extend([
                "## 統計的提言",
                "",
                "### 現在の分析の評価",
                ""
            ])
            
            if 'current_analysis_evaluation' in recs:
                eval_data = recs['current_analysis_evaluation']
                
                if 'strengths' in eval_data:
                    report_lines.append("**強み:**")
                    for strength in eval_data['strengths']:
                        report_lines.append(f"- {strength}")
                    report_lines.append("")
                
                if 'limitations' in eval_data:
                    report_lines.append("**制約事項:**")
                    for limitation in eval_data['limitations']:
                        report_lines.append(f"- {limitation}")
                    report_lines.append("")
            
            if 'future_study_design' in recs:
                future_design = recs['future_study_design']
                
                report_lines.extend([
                    "### 将来研究への提言",
                    ""
                ])
                
                if 'design_improvements' in future_design:
                    report_lines.append("**デザイン改善:**")
                    for improvement in future_design['design_improvements']:
                        report_lines.append(f"- {improvement}")
                    report_lines.append("")
        
        # 結論
        report_lines.extend([
            "## 結論",
            "",
            "本分析は独立群比較として適切に実施されましたが、検出力の観点から改善の余地があります。",
            "将来の研究では、より大きなサンプルサイズと個人追跡可能な研究デザインを推奨します。",
            "",
            "---",
            "",
            "**Generated by**: Claude Code Analysis (Power Analysis Implementation)",
            f"**Output Files**: {self.output_dir}",
            f"**Figures**: {self.figures_dir}"
        ])
        
        # レポート保存
        report_path = self.output_dir / "power_analysis_summary.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✓ 要約レポート保存: {report_path}")
    
    def run_complete_analysis(self):
        """完全検出力分析実行"""
        print("="*60)
        print("統計的検出力分析とサンプルサイズ計算")
        print("="*60)
        print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        try:
            # 分析実行
            self.load_data_and_results()
            self.calculate_effect_sizes()
            self.calculate_post_hoc_power()
            self.calculate_required_sample_sizes()
            self.bootstrap_confidence_intervals()
            self.create_visualizations()
            self.generate_recommendations()
            self.save_results()
            
            print("\n" + "="*60)
            print("🎉 検出力分析完了!")
            print("="*60)
            print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            print("📁 出力ファイル:")
            print(f"  - 詳細結果: {self.output_dir}/power_analysis_results.json")
            print(f"  - 要約レポート: {self.output_dir}/power_analysis_summary.txt")
            print(f"  - 図表: {self.figures_dir}/")
            print()
            print("📊 主要な発見:")
            
            # 主要結果の表示
            if hasattr(self, 'results') and 'effect_sizes' in self.results:
                if 'q1_composite' in self.results['effect_sizes']:
                    q1_d = self.results['effect_sizes']['q1_composite']['cohens_d']
                    print(f"  - Q1総合スコア効果量: Cohen's d = {q1_d:.3f}")
                
                if 'q3_composite' in self.results['effect_sizes']:
                    q3_d = self.results['effect_sizes']['q3_composite']['cohens_d']
                    print(f"  - Q3総合スコア効果量: Cohen's d = {q3_d:.3f}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ 検出力分析エラー: {e}")
            import traceback
            print(f"詳細: {traceback.format_exc()}")
            return False

def main():
    """メイン実行関数"""
    project_root = Path(__file__).parent.parent.parent
    
    analyzer = PowerAnalysisEvaluator(project_root)
    
    try:
        success = analyzer.run_complete_analysis()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 ユーザーによって中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()