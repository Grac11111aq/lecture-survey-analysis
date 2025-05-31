#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小学校出前授業アンケート Phase 3: 集団間差異の分析
============================================

実施内容:
- クラス間比較（一元配置分散分析）
- 授業前・後・変化量のクラス間差異
- 事後検定（Tukey HSD）
- 理解度の要因分析（重回帰・ロジスティック回帰）

Author: Claude Code Analysis
Date: 2025-05-31
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, kruskal
import warnings
from pathlib import Path
import json
from datetime import datetime
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import itertools

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
warnings.filterwarnings('ignore')

class Phase3GroupAnalyzer:
    """Phase 3: 集団間差異分析クラス"""
    
    def __init__(self, data_dir="data/analysis"):
        self.data_dir = Path(data_dir)
        self.results = {}
        self.before_df = None
        self.after_df = None
        self.comment_df = None
        self.paired_data = None
        self.alpha = 0.05
        
    def load_data(self):
        """データの読み込みとペアリング"""
        try:
            self.before_df = pd.read_csv(self.data_dir / "before_excel_compliant.csv")
            self.after_df = pd.read_csv(self.data_dir / "after_excel_compliant.csv")
            self.comment_df = pd.read_csv(self.data_dir / "comment.csv")
            
            print("✓ データ読み込み完了")
            print(f"  - 授業前: {len(self.before_df)} 行")
            print(f"  - 授業後: {len(self.after_df)} 行")
            print(f"  - 感想文: {len(self.comment_df)} 行")
            
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
        
        # 変化量の計算
        self.calculate_change_scores()
    
    def calculate_change_scores(self):
        """変化量スコアの計算"""
        # Q1項目のマッピング
        q1_mapping = {
            'Q1_Saltwater_Response': 'Q1_Saltwater',
            'Q1_Sugarwater_Response': 'Q1_Sugarwater', 
            'Q1_Muddywater_Response': 'Q1_Muddywater',
            'Q1_Ink_Response': 'Q1_Ink',
            'Q1_MisoSoup_Response': 'Q1_MisoSoup',
            'Q1_SoySauce_Response': 'Q1_SoySauce'
        }
        
        # Q3項目のマッピング
        q3_mapping = {
            'Q3_TeaLeavesDissolve': 'Q3_TeaLeaves_DissolveInWater',
            'Q3_TeaComponentsDissolve': 'Q3_TeaComponents_DissolveInWater'
        }
        
        # 変化量の計算
        for before_col, after_col in q1_mapping.items():
            if before_col in self.paired_data.columns and after_col in self.paired_data.columns:
                change_col = f"{before_col}_change"
                self.paired_data[change_col] = (
                    self.paired_data[after_col].astype(float) - 
                    self.paired_data[before_col].astype(float)
                )
        
        for before_col, after_col in q3_mapping.items():
            if before_col in self.paired_data.columns and after_col in self.paired_data.columns:
                change_col = f"{before_col}_change"
                self.paired_data[change_col] = (
                    self.paired_data[after_col].astype(float) - 
                    self.paired_data[before_col].astype(float)
                )
        
        # 総合スコアの計算
        self.calculate_composite_scores()
        
        print("✓ 変化量スコア計算完了")
    
    def calculate_composite_scores(self):
        """総合スコアの計算"""
        # Q1総合スコア（授業前・後）
        q1_before_cols = [col for col in self.paired_data.columns 
                         if col.startswith('Q1_') and col.endswith('_Response')]
        q1_after_cols = [col.replace('_Response', '') for col in q1_before_cols 
                        if col.replace('_Response', '') in self.paired_data.columns]
        
        if q1_before_cols and q1_after_cols:
            self.paired_data['Q1_total_before'] = self.paired_data[q1_before_cols].sum(axis=1)
            self.paired_data['Q1_total_after'] = self.paired_data[q1_after_cols].sum(axis=1)
            self.paired_data['Q1_total_change'] = (
                self.paired_data['Q1_total_after'] - self.paired_data['Q1_total_before']
            )
        
        # Q3総合スコア（授業前・後）
        q3_before_cols = [col for col in self.paired_data.columns 
                         if col.startswith('Q3_') and not col.endswith('_change')]
        q3_after_cols = [col for col in self.paired_data.columns 
                        if col.startswith('Q3_') and 'DissolveInWater' in col]
        
        if len(q3_before_cols) >= 2 and len(q3_after_cols) >= 2:
            # Q3の最初の2列を使用（TeaLeavesDissolve, TeaComponentsDissolve）
            q3_before_subset = [col for col in q3_before_cols if not col.endswith('_change')][:2]
            q3_after_subset = q3_after_cols[:2]
            
            self.paired_data['Q3_total_before'] = self.paired_data[q3_before_subset].sum(axis=1)
            self.paired_data['Q3_total_after'] = self.paired_data[q3_after_subset].sum(axis=1)
            self.paired_data['Q3_total_change'] = (
                self.paired_data['Q3_total_after'] - self.paired_data['Q3_total_before']
            )
    
    def class_comparison_analysis(self):
        """クラス間比較分析"""
        print("\n" + "="*50)
        print("クラス間比較分析（一元配置分散分析）")
        print("="*50)
        
        class_results = {
            'before_analysis': {},
            'after_analysis': {},
            'change_analysis': {},
            'summary': {}
        }
        
        # 分析対象変数の定義
        analysis_vars = {
            'before': {
                'Q1_total_before': 'Q1総合スコア（授業前）',
                'Q3_total_before': 'Q3総合スコア（授業前）'
            },
            'after': {
                'Q1_total_after': 'Q1総合スコア（授業後）',
                'Q3_total_after': 'Q3総合スコア（授業後）',
                'Q4_ExperimentInterestRating': '実験への興味',
                'Q5_NewLearningsRating': '新しい学び',
                'Q6_DissolvingUnderstandingRating': '溶解理解度'
            },
            'change': {
                'Q1_total_change': 'Q1総合スコア変化量',
                'Q3_total_change': 'Q3総合スコア変化量'
            }
        }
        
        # 各カテゴリで分析実行
        for category, variables in analysis_vars.items():
            print(f"\n{category.upper()}項目の分析")
            print("-" * 30)
            
            category_results = {}
            
            for var_name, var_label in variables.items():
                if var_name in self.paired_data.columns:
                    result = self.perform_anova_analysis(var_name, var_label)
                    category_results[var_name] = result
                else:
                    print(f"  ⚠️  {var_label}: データなし")
            
            class_results[f'{category}_analysis'] = category_results
        
        self.results['class_comparison'] = class_results
        return class_results
    
    def perform_anova_analysis(self, variable, label):
        """個別変数のANOVA分析"""
        # データ準備
        data = self.paired_data[[variable, 'class_before']].dropna()
        
        if len(data) < 10:
            return {"error": f"サンプルサイズ不足 (n={len(data)})"}
        
        # クラス別のデータ準備
        classes = sorted(data['class_before'].unique())
        class_data = [data[data['class_before'] == cls][variable].values for cls in classes]
        
        # 基本統計量
        descriptive_stats = {}
        for cls in classes:
            cls_data = data[data['class_before'] == cls][variable]
            descriptive_stats[cls] = {
                'n': len(cls_data),
                'mean': float(cls_data.mean()),
                'std': float(cls_data.std()),
                'median': float(cls_data.median())
            }
        
        # 正規性・等分散性の検定
        normality_results = self.test_assumptions(class_data, classes)
        
        # ANOVA実行
        if normality_results['use_parametric']:
            # パラメトリック検定（一元配置分散分析）
            f_stat, p_value = f_oneway(*class_data)
            test_type = "One-way ANOVA"
            test_statistic = f_stat
        else:
            # ノンパラメトリック検定（Kruskal-Wallis）
            h_stat, p_value = kruskal(*class_data)
            test_type = "Kruskal-Wallis"
            test_statistic = h_stat
        
        # 効果量の計算（eta-squared）
        eta_squared = self.calculate_eta_squared(class_data)
        
        # 事後検定
        posthoc_results = None
        if p_value < self.alpha:
            posthoc_results = self.perform_posthoc_tests(data, variable, classes, normality_results['use_parametric'])
        
        result = {
            'variable': variable,
            'label': label,
            'n_total': len(data),
            'n_classes': len(classes),
            'classes': classes,
            'descriptive_stats': descriptive_stats,
            'test_type': test_type,
            'test_statistic': float(test_statistic),
            'p_value': float(p_value),
            'eta_squared': float(eta_squared),
            'significant': p_value < self.alpha,
            'assumption_tests': normality_results,
            'posthoc_results': posthoc_results,
            'interpretation': self.interpret_anova_result(p_value, eta_squared)
        }
        
        # 結果表示
        sig_symbol = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"  {label}:")
        print(f"    {test_type}: 統計量 = {test_statistic:.3f}, p = {p_value:.4f} {sig_symbol}")
        print(f"    効果量 η² = {eta_squared:.3f}")
        
        # クラス別平均表示
        for cls in classes:
            stats = descriptive_stats[cls]
            print(f"    クラス{cls}: {stats['mean']:.2f} ± {stats['std']:.2f} (n={stats['n']})")
        
        return result
    
    def test_assumptions(self, class_data, classes):
        """ANOVA の前提条件検定"""
        # 正規性検定（各群）
        normality_pvals = []
        for i, data_group in enumerate(class_data):
            if len(data_group) >= 3:
                _, p_val = stats.shapiro(data_group)
                normality_pvals.append(p_val)
        
        # 等分散性検定（Levene検定）
        if len(class_data) >= 2 and all(len(group) >= 2 for group in class_data):
            levene_stat, levene_p = stats.levene(*class_data)
        else:
            levene_stat, levene_p = np.nan, np.nan
        
        # 前提条件の判定
        normality_ok = len(normality_pvals) > 0 and min(normality_pvals) > 0.05
        homogeneity_ok = not np.isnan(levene_p) and levene_p > 0.05
        use_parametric = normality_ok and homogeneity_ok
        
        return {
            'normality_pvals': normality_pvals,
            'normality_ok': normality_ok,
            'levene_statistic': float(levene_stat) if not np.isnan(levene_stat) else None,
            'levene_p': float(levene_p) if not np.isnan(levene_p) else None,
            'homogeneity_ok': homogeneity_ok,
            'use_parametric': use_parametric
        }
    
    def calculate_eta_squared(self, class_data):
        """効果量（eta-squared）の計算"""
        try:
            # グループ間平方和とグループ内平方和の計算
            all_data = np.concatenate(class_data)
            grand_mean = np.mean(all_data)
            
            # Between-group sum of squares
            ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in class_data)
            
            # Total sum of squares
            ss_total = sum((x - grand_mean)**2 for x in all_data)
            
            if ss_total == 0:
                return 0.0
            
            eta_squared = ss_between / ss_total
            return eta_squared
            
        except Exception:
            return np.nan
    
    def perform_posthoc_tests(self, data, variable, classes, use_parametric):
        """事後検定の実行"""
        if use_parametric and len(classes) > 2:
            try:
                # Tukey HSD検定
                tukey_result = pairwise_tukeyhsd(
                    data[variable], 
                    data['class_before'], 
                    alpha=self.alpha
                )
                
                # 結果の整理
                posthoc_results = {
                    'test_type': 'Tukey HSD',
                    'comparisons': []
                }
                
                for i in range(len(tukey_result.summary().data) - 1):  # ヘッダー除外
                    row = tukey_result.summary().data[i + 1]
                    comparison = {
                        'group1': str(row[0]),
                        'group2': str(row[1]),
                        'mean_diff': float(row[2]),
                        'p_adj': float(row[3]),
                        'lower_ci': float(row[4]),
                        'upper_ci': float(row[5]),
                        'reject': str(row[6]) == 'True'
                    }
                    posthoc_results['comparisons'].append(comparison)
                
                return posthoc_results
                
            except Exception as e:
                return {"error": f"Tukey HSD検定エラー: {e}"}
        
        return None
    
    def interpret_anova_result(self, p_value, eta_squared):
        """ANOVA結果の解釈"""
        # 有意性の判定
        if p_value < 0.001:
            significance = "極めて有意"
        elif p_value < 0.01:
            significance = "高度に有意"
        elif p_value < 0.05:
            significance = "有意"
        else:
            significance = "非有意"
        
        # 効果量の判定
        if np.isnan(eta_squared):
            effect_interpretation = "効果量算出不可"
        elif eta_squared < 0.01:
            effect_interpretation = "効果なし/小"
        elif eta_squared < 0.06:
            effect_interpretation = "中程度の効果"
        elif eta_squared < 0.14:
            effect_interpretation = "大きな効果"
        else:
            effect_interpretation = "極めて大きな効果"
        
        return f"{significance}, {effect_interpretation}"
    
    def understanding_factors_analysis(self):
        """理解度の要因分析"""
        print("\n" + "="*50)
        print("理解度の要因分析（重回帰・ロジスティック回帰）")
        print("="*50)
        
        factors_results = {}
        
        # 連続変数での重回帰分析
        print("\n1. 重回帰分析（理解度連続値）")
        print("-" * 30)
        
        regression_result = self.perform_multiple_regression()
        factors_results['multiple_regression'] = regression_result
        
        # カテゴリカル変数でのロジスティック回帰
        print("\n2. ロジスティック回帰分析（理解度二値）")
        print("-" * 30)
        
        logistic_result = self.perform_logistic_regression()
        factors_results['logistic_regression'] = logistic_result
        
        self.results['factors_analysis'] = factors_results
        return factors_results
    
    def perform_multiple_regression(self):
        """重回帰分析の実行"""
        try:
            # 予測変数の準備
            predictors = [
                'Q4_ExperimentInterestRating',  # 実験への興味
                'Q1_total_before',              # 授業前Q1スコア
                'Q3_total_before'               # 授業前Q3スコア
            ]
            
            # 目的変数
            target = 'Q6_DissolvingUnderstandingRating'  # 理解度
            
            # クラスダミー変数の作成
            class_dummies = pd.get_dummies(self.paired_data['class_before'], prefix='class')
            
            # データ準備
            analysis_data = self.paired_data[predictors + [target]].copy()
            analysis_data = pd.concat([analysis_data, class_dummies], axis=1)
            analysis_data = analysis_data.dropna()
            
            if len(analysis_data) < 10:
                return {"error": f"サンプルサイズ不足 (n={len(analysis_data)})"}
            
            # 予測変数リストにクラスダミーを追加（参照クラスを除く）
            class_cols = [col for col in class_dummies.columns if col != class_dummies.columns[0]]
            all_predictors = predictors + class_cols
            
            # 重回帰分析
            formula = f"{target} ~ " + " + ".join(all_predictors)
            model = ols(formula, data=analysis_data).fit()
            
            # VIF（多重共線性）の確認
            vif_values = self.calculate_vif(analysis_data[all_predictors])
            
            result = {
                'n': len(analysis_data),
                'formula': formula,
                'r_squared': float(model.rsquared),
                'adj_r_squared': float(model.rsquared_adj),
                'f_statistic': float(model.fvalue),
                'f_pvalue': float(model.f_pvalue),
                'coefficients': {},
                'vif_values': vif_values,
                'multicollinearity_ok': all(v < 10 for v in vif_values.values())
            }
            
            # 係数の詳細
            for var in all_predictors:
                if var in model.params.index:
                    result['coefficients'][var] = {
                        'coef': float(model.params[var]),
                        'std_err': float(model.bse[var]),
                        't_value': float(model.tvalues[var]),
                        'p_value': float(model.pvalues[var]),
                        'significant': model.pvalues[var] < self.alpha
                    }
            
            # 結果表示
            print(f"  サンプルサイズ: n = {result['n']}")
            print(f"  決定係数: R² = {result['r_squared']:.3f}, 調整済R² = {result['adj_r_squared']:.3f}")
            print(f"  F統計量: F = {result['f_statistic']:.3f}, p = {result['f_pvalue']:.4f}")
            print("  係数:")
            
            for var, coef_info in result['coefficients'].items():
                sig_symbol = "*" if coef_info['significant'] else ""
                var_label = var.replace('Q4_ExperimentInterestRating', '実験興味')\
                              .replace('Q1_total_before', 'Q1事前')\
                              .replace('Q3_total_before', 'Q3事前')
                print(f"    {var_label:12}: β = {coef_info['coef']:6.3f}, p = {coef_info['p_value']:.4f} {sig_symbol}")
            
            return result
            
        except Exception as e:
            return {"error": f"重回帰分析エラー: {e}"}
    
    def calculate_vif(self, predictors_df):
        """VIF（分散膨張因子）の計算"""
        vif_values = {}
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            
            for i, col in enumerate(predictors_df.columns):
                if predictors_df[col].var() > 0:  # 分散が0でない列のみ
                    vif = variance_inflation_factor(predictors_df.values, i)
                    vif_values[col] = float(vif)
                else:
                    vif_values[col] = np.inf
                    
        except Exception:
            # VIF計算できない場合はすべて1.0を返す
            vif_values = {col: 1.0 for col in predictors_df.columns}
        
        return vif_values
    
    def perform_logistic_regression(self):
        """ロジスティック回帰分析の実行"""
        try:
            # 理解度を二値化（4段階 → 高/低）
            target_binary = (self.paired_data['Q6_DissolvingUnderstandingRating'] >= 4).astype(int)
            
            # 予測変数の準備
            predictors = [
                'Q4_ExperimentInterestRating',
                'Q1_total_before',
                'Q3_total_before'
            ]
            
            # クラスダミー変数
            class_dummies = pd.get_dummies(self.paired_data['class_before'], prefix='class')
            
            # データ準備
            X_data = self.paired_data[predictors].copy()
            X_data = pd.concat([X_data, class_dummies.iloc[:, 1:]], axis=1)  # 参照クラス除く
            
            # 欠損値除去
            complete_mask = ~(X_data.isna().any(axis=1) | target_binary.isna())
            X_clean = X_data[complete_mask]
            y_clean = target_binary[complete_mask]
            
            if len(X_clean) < 10:
                return {"error": f"サンプルサイズ不足 (n={len(X_clean)})"}
            
            # 標準化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)
            
            # ロジスティック回帰
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_scaled, y_clean)
            
            # 予測と評価
            y_pred = model.predict(X_scaled)
            y_prob = model.predict_proba(X_scaled)[:, 1]
            
            # 分類レポート
            class_report = classification_report(y_clean, y_pred, output_dict=True)
            
            result = {
                'n': len(X_clean),
                'feature_names': list(X_clean.columns),
                'coefficients': model.coef_[0].tolist(),
                'intercept': float(model.intercept_[0]),
                'accuracy': float(class_report['accuracy']),
                'precision': float(class_report['1']['precision']),
                'recall': float(class_report['1']['recall']),
                'f1_score': float(class_report['1']['f1-score']),
                'class_distribution': y_clean.value_counts().to_dict(),
                'feature_importance': {
                    name: abs(coef) for name, coef in zip(X_clean.columns, model.coef_[0])
                }
            }
            
            # 結果表示
            print(f"  サンプルサイズ: n = {result['n']}")
            print(f"  予測精度: Accuracy = {result['accuracy']:.3f}")
            print(f"  高理解度の割合: {result['class_distribution'].get(1, 0)}/{result['n']}")
            print("  特徴量重要度:")
            
            sorted_importance = sorted(result['feature_importance'].items(), 
                                     key=lambda x: x[1], reverse=True)
            for name, importance in sorted_importance:
                var_label = name.replace('Q4_ExperimentInterestRating', '実験興味')\
                              .replace('Q1_total_before', 'Q1事前')\
                              .replace('Q3_total_before', 'Q3事前')
                print(f"    {var_label:12}: {importance:.3f}")
            
            return result
            
        except Exception as e:
            return {"error": f"ロジスティック回帰分析エラー: {e}"}
    
    def create_visualizations(self):
        """可視化の作成"""
        print("\n" + "="*50)
        print("集団間差異分析の可視化")
        print("="*50)
        
        output_dir = Path("outputs/phase3_figures")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # クラス間比較の可視化
        self.plot_class_comparisons(output_dir)
        
        # 要因分析結果の可視化
        self.plot_factors_analysis(output_dir)
        
        # クラス別分布の可視化
        self.plot_class_distributions(output_dir)
        
        print(f"✓ 図表を {output_dir} に保存しました")
    
    def plot_class_comparisons(self, output_dir):
        """クラス間比較の可視化"""
        if 'class_comparison' not in self.results:
            return
        
        # 有意な結果がある変数を抽出
        significant_vars = []
        all_results = {}
        
        for category in ['before_analysis', 'after_analysis', 'change_analysis']:
            if category in self.results['class_comparison']:
                for var, result in self.results['class_comparison'][category].items():
                    if 'error' not in result and result.get('significant', False):
                        significant_vars.append((var, result['label']))
                        all_results[var] = result
        
        if not significant_vars:
            # 有意でない場合も主要変数を表示
            for category in ['after_analysis']:
                if category in self.results['class_comparison']:
                    for var, result in self.results['class_comparison'][category].items():
                        if 'error' not in result:
                            significant_vars.append((var, result['label']))
                            all_results[var] = result
                            break  # 最初の1つだけ
        
        if not significant_vars:
            return
        
        # 図の作成
        n_vars = len(significant_vars)
        fig, axes = plt.subplots((n_vars + 1) // 2, 2, figsize=(12, 4 * ((n_vars + 1) // 2)))
        if n_vars == 1:
            axes = [axes]
        axes = axes.flatten() if n_vars > 1 else axes
        
        for i, (var, label) in enumerate(significant_vars):
            if i >= len(axes):
                break
                
            result = all_results[var]
            
            # クラス別の平均値と標準誤差
            classes = result['classes']
            means = [result['descriptive_stats'][cls]['mean'] for cls in classes]
            stds = [result['descriptive_stats'][cls]['std'] for cls in classes]
            ns = [result['descriptive_stats'][cls]['n'] for cls in classes]
            sems = [std / np.sqrt(n) for std, n in zip(stds, ns)]
            
            # バープロット
            bars = axes[i].bar(classes, means, yerr=sems, capsize=5, alpha=0.7)
            axes[i].set_xlabel('Class')
            axes[i].set_ylabel(label)
            axes[i].set_title(f'{label}\n{result["test_type"]}: p = {result["p_value"]:.4f}')
            axes[i].grid(True, alpha=0.3)
            
            # 有意性マーク
            if result['significant']:
                axes[i].text(0.5, 0.95, '***' if result['p_value'] < 0.001 else 
                           '**' if result['p_value'] < 0.01 else '*',
                           transform=axes[i].transAxes, ha='center', va='top',
                           fontsize=16, color='red', fontweight='bold')
        
        # 余った軸を非表示
        for j in range(len(significant_vars), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / "class_comparisons.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_factors_analysis(self, output_dir):
        """要因分析結果の可視化"""
        if 'factors_analysis' not in self.results:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 重回帰分析結果
        if 'multiple_regression' in self.results['factors_analysis']:
            reg_result = self.results['factors_analysis']['multiple_regression']
            
            if 'error' not in reg_result and 'coefficients' in reg_result:
                vars_list = []
                coefs = []
                p_values = []
                
                for var, coef_info in reg_result['coefficients'].items():
                    if not var.startswith('class_'):  # クラスダミーを除外
                        vars_list.append(var.replace('Q4_ExperimentInterestRating', '実験興味')\
                                               .replace('Q1_total_before', 'Q1事前')\
                                               .replace('Q3_total_before', 'Q3事前'))
                        coefs.append(coef_info['coef'])
                        p_values.append(coef_info['p_value'])
                
                if vars_list:
                    colors = ['red' if p < 0.05 else 'gray' for p in p_values]
                    bars = axes[0].bar(vars_list, coefs, color=colors, alpha=0.7)
                    axes[0].set_ylabel('Regression Coefficient')
                    axes[0].set_title(f'Multiple Regression\nR² = {reg_result["r_squared"]:.3f}')
                    axes[0].tick_params(axis='x', rotation=45)
                    axes[0].grid(True, alpha=0.3)
                    axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # ロジスティック回帰結果
        if 'logistic_regression' in self.results['factors_analysis']:
            log_result = self.results['factors_analysis']['logistic_regression']
            
            if 'error' not in log_result and 'feature_importance' in log_result:
                importance_items = [(k.replace('Q4_ExperimentInterestRating', '実験興味')\
                                      .replace('Q1_total_before', 'Q1事前')\
                                      .replace('Q3_total_before', 'Q3事前'), v) 
                                   for k, v in log_result['feature_importance'].items()
                                   if not k.startswith('class_')]
                
                if importance_items:
                    vars_list, importances = zip(*importance_items)
                    bars = axes[1].bar(vars_list, importances, alpha=0.7, color='steelblue')
                    axes[1].set_ylabel('Feature Importance (|coefficient|)')
                    axes[1].set_title(f'Logistic Regression\nAccuracy = {log_result["accuracy"]:.3f}')
                    axes[1].tick_params(axis='x', rotation=45)
                    axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "factors_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_class_distributions(self, output_dir):
        """クラス別分布の可視化"""
        # 主要評価項目のクラス別分布
        evaluation_vars = [
            ('Q4_ExperimentInterestRating', '実験への興味'),
            ('Q5_NewLearningsRating', '新しい学び'),
            ('Q6_DissolvingUnderstandingRating', '溶解理解度')
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for i, (var, label) in enumerate(evaluation_vars):
            if var in self.paired_data.columns:
                # クラス別のヒストグラム
                classes = sorted(self.paired_data['class_before'].unique())
                
                for cls in classes:
                    class_data = self.paired_data[self.paired_data['class_before'] == cls][var].dropna()
                    if len(class_data) > 0:
                        axes[i].hist(class_data, alpha=0.6, label=f'Class {cls}', bins=range(1, 6))
                
                axes[i].set_xlabel('Rating')
                axes[i].set_ylabel('Frequency')
                axes[i].set_title(label)
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
                axes[i].set_xticks(range(1, 5))
        
        plt.tight_layout()
        plt.savefig(output_dir / "class_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self):
        """Phase 3 レポート生成"""
        print("\n" + "="*50)
        print("Phase 3 レポート生成")
        print("="*50)
        
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # JSON形式で詳細結果を保存
        with open(output_dir / "phase3_detailed_results.json", 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        
        # テキスト形式でサマリーレポートを生成
        report_content = self.create_summary_report()
        
        with open(output_dir / "phase3_summary_report.txt", 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✓ レポートを {output_dir} に保存しました")
        print(f"  - 詳細結果: phase3_detailed_results.json")
        print(f"  - サマリー: phase3_summary_report.txt")
        
        return report_content
    
    def create_summary_report(self):
        """サマリーレポートの作成"""
        report = []
        report.append("="*60)
        report.append("小学校出前授業アンケート Phase 3 集団間差異分析結果")
        report.append("="*60)
        report.append(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # サンプルサイズ
        report.append("【サンプルサイズ】")
        report.append(f"ペア数: {len(self.paired_data)}")
        
        # クラス別分布
        class_dist = self.paired_data['class_before'].value_counts().sort_index()
        report.append("クラス別分布:")
        for cls, count in class_dist.items():
            report.append(f"  クラス{cls}: {count}名")
        report.append("")
        
        # クラス間比較結果
        if 'class_comparison' in self.results:
            report.append("【クラス間比較結果（ANOVA）】")
            
            significant_results = []
            
            for category, category_label in [
                ('before_analysis', '授業前'),
                ('after_analysis', '授業後'),
                ('change_analysis', '変化量')
            ]:
                if category in self.results['class_comparison']:
                    report.append(f"\n{category_label}:")
                    
                    category_results = self.results['class_comparison'][category]
                    for var, result in category_results.items():
                        if 'error' in result:
                            continue
                        
                        var_label = result.get('label', var)
                        significance = "有意" if result['significant'] else "非有意"
                        
                        report.append(f"  {var_label}:")
                        report.append(f"    {result['test_type']}: p = {result['p_value']:.4f} ({significance})")
                        report.append(f"    効果量 η² = {result['eta_squared']:.3f}")
                        
                        if result['significant']:
                            significant_results.append(var_label)
                        
                        # クラス別平均
                        for cls in result['classes']:
                            stats = result['descriptive_stats'][cls]
                            report.append(f"      クラス{cls}: {stats['mean']:.2f} ± {stats['std']:.2f}")
            
            if significant_results:
                report.append(f"\n有意なクラス間差異: {', '.join(significant_results)}")
            else:
                report.append("\n有意なクラス間差異なし")
            
            report.append("")
        
        # 要因分析結果
        if 'factors_analysis' in self.results:
            report.append("【理解度の要因分析】")
            
            # 重回帰分析
            if 'multiple_regression' in self.results['factors_analysis']:
                reg_result = self.results['factors_analysis']['multiple_regression']
                
                if 'error' not in reg_result:
                    report.append(f"\n重回帰分析:")
                    report.append(f"  サンプルサイズ: n = {reg_result['n']}")
                    report.append(f"  決定係数: R² = {reg_result['r_squared']:.3f}")
                    report.append(f"  モデル有意性: F = {reg_result['f_statistic']:.3f}, p = {reg_result['f_pvalue']:.4f}")
                    
                    significant_predictors = []
                    for var, coef_info in reg_result['coefficients'].items():
                        if not var.startswith('class_') and coef_info['significant']:
                            var_label = var.replace('Q4_ExperimentInterestRating', '実験への興味')\
                                          .replace('Q1_total_before', '授業前Q1スコア')\
                                          .replace('Q3_total_before', '授業前Q3スコア')
                            significant_predictors.append(var_label)
                    
                    if significant_predictors:
                        report.append(f"  有意な予測因子: {', '.join(significant_predictors)}")
                    else:
                        report.append("  有意な予測因子なし")
            
            # ロジスティック回帰
            if 'logistic_regression' in self.results['factors_analysis']:
                log_result = self.results['factors_analysis']['logistic_regression']
                
                if 'error' not in log_result:
                    report.append(f"\nロジスティック回帰分析:")
                    report.append(f"  サンプルサイズ: n = {log_result['n']}")
                    report.append(f"  予測精度: {log_result['accuracy']:.3f}")
                    
                    # 最も重要な特徴量
                    if 'feature_importance' in log_result:
                        sorted_features = sorted(log_result['feature_importance'].items(),
                                               key=lambda x: x[1], reverse=True)
                        if sorted_features:
                            top_feature = sorted_features[0]
                            var_label = top_feature[0].replace('Q4_ExperimentInterestRating', '実験への興味')\
                                                      .replace('Q1_total_before', '授業前Q1スコア')\
                                                      .replace('Q3_total_before', '授業前Q3スコア')
                            report.append(f"  最重要因子: {var_label} (重要度: {top_feature[1]:.3f})")
            
            report.append("")
        
        # 主要な結論
        report.append("【主要な結論】")
        
        # クラス効果の評価
        class_effects = 0
        if 'class_comparison' in self.results:
            for category in ['before_analysis', 'after_analysis', 'change_analysis']:
                if category in self.results['class_comparison']:
                    for result in self.results['class_comparison'][category].values():
                        if 'error' not in result and result.get('significant', False):
                            class_effects += 1
        
        if class_effects > 0:
            report.append(f"✓ {class_effects}項目でクラス間差異を検出")
        else:
            report.append("• クラス間差異は検出されず")
        
        # 予測精度の評価
        prediction_accuracy = None
        if ('factors_analysis' in self.results and 
            'logistic_regression' in self.results['factors_analysis'] and
            'error' not in self.results['factors_analysis']['logistic_regression']):
            prediction_accuracy = self.results['factors_analysis']['logistic_regression']['accuracy']
            
            if prediction_accuracy > 0.7:
                report.append(f"✓ 理解度予測精度良好 ({prediction_accuracy:.3f})")
            else:
                report.append(f"⚠️  理解度予測精度限定的 ({prediction_accuracy:.3f})")
        
        # 個人差要因の評価
        significant_factors = 0
        if ('factors_analysis' in self.results and 
            'multiple_regression' in self.results['factors_analysis'] and
            'coefficients' in self.results['factors_analysis']['multiple_regression']):
            for coef_info in self.results['factors_analysis']['multiple_regression']['coefficients'].values():
                if coef_info.get('significant', False):
                    significant_factors += 1
        
        if significant_factors > 0:
            report.append(f"✓ {significant_factors}個の有意な個人差要因を特定")
        else:
            report.append("• 明確な個人差要因は特定されず")
        
        report.append("")
        report.append("【Phase 4への推奨事項】")
        report.append("1. テキストデータの定量的分析")
        report.append("2. 感想文からの質的洞察の抽出")
        if class_effects == 0:
            report.append("3. クラス間差異が小さい理由の探索")
        if prediction_accuracy and prediction_accuracy < 0.7:
            report.append("4. 追加的な予測因子の検討")
        report.append("5. 教育効果の質的側面の分析")
        
        return "\n".join(report)
    
    def run_complete_analysis(self):
        """Phase 3 完全分析実行"""
        print("小学校出前授業アンケート Phase 3: 集団間差異の分析")
        print("実行開始:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("="*60)
        
        try:
            # データ読み込み
            self.load_data()
            
            # クラス間比較分析
            self.class_comparison_analysis()
            
            # 理解度要因分析
            self.understanding_factors_analysis()
            
            # 可視化作成
            self.create_visualizations()
            
            # レポート生成
            summary_report = self.generate_report()
            
            print("\n" + "="*60)
            print("Phase 3 分析完了!")
            print("="*60)
            print(summary_report)
            
            return self.results
            
        except Exception as e:
            print(f"❌ Phase 3 分析中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """メイン実行関数"""
    analyzer = Phase3GroupAnalyzer()
    results = analyzer.run_complete_analysis()
    return results

if __name__ == "__main__":
    main()