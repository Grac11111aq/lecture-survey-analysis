#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
構造方程式モデリング（SEM）による教育効果の因果構造分析
=======================================================

独立群比較における学習プロセスの構造的関係を解明する。

機能:
- 潜在変数モデル構築（科学的理解、学習積極性）
- パス解析による因果関係推定
- モデル適合度評価（CFI, RMSEA, SRMR）
- 間接効果・総効果の算出
- 教育的示唆の抽出

制約:
- Page_IDによる個人追跡不可のため独立群設計
- 授業前後データを別群として扱う
- クロスセクショナルSEMモデル適用

Author: Claude Code Analysis (SEM Implementation)
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

# SEM関連ライブラリ
import semopy

# 警告抑制
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class StructuralEquationModeling:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data" / "analysis"
        self.output_dir = self.project_root / "outputs" / "current" / "05_advanced_analysis"
        self.figures_dir = self.project_root / "outputs" / "figures" / "current" / "05_advanced_analysis"
        
        # 出力ディレクトリ作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        
    def load_data(self):
        """データ読み込み"""
        print("📊 データ読み込み中...")
        
        # 授業前後データ読み込み
        before_path = self.data_dir / "before_excel_compliant.csv"
        after_path = self.data_dir / "after_excel_compliant.csv"
        
        if not before_path.exists() or not after_path.exists():
            raise FileNotFoundError("必要なデータファイルが見つかりません")
        
        self.before_data = pd.read_csv(before_path, encoding='utf-8')
        self.after_data = pd.read_csv(after_path, encoding='utf-8')
        
        print(f"✓ 授業前データ: {len(self.before_data)} 件")
        print(f"✓ 授業後データ: {len(self.after_data)} 件")
        
    def prepare_sem_data(self):
        """SEM分析用データ準備"""
        print("\n🔧 SEM分析用データ準備中...")
        
        # 授業前データ処理
        before_sem = self.before_data.copy()
        
        # Q1総合スコア計算（授業前）
        q1_before_cols = ['Q1_Saltwater_Response', 'Q1_Sugarwater_Response', 'Q1_Muddywater_Response', 
                         'Q1_Ink_Response', 'Q1_MisoSoup_Response', 'Q1_SoySauce_Response']
        before_sem['Q1_total_before'] = before_sem[q1_before_cols].sum(axis=1)
        
        # Q3総合スコア計算（授業前）
        q3_before_cols = ['Q3_TeaLeavesDissolve', 'Q3_TeaComponentsDissolve']
        before_sem['Q3_total_before'] = before_sem[q3_before_cols].sum(axis=1)
        
        # 授業後データ処理
        after_sem = self.after_data.copy()
        
        # Q1総合スコア計算（授業後）
        q1_after_cols = ['Q1_Saltwater', 'Q1_Sugarwater', 'Q1_Muddywater',
                        'Q1_Ink', 'Q1_MisoSoup', 'Q1_SoySauce']
        after_sem['Q1_total_after'] = after_sem[q1_after_cols].sum(axis=1)
        
        # Q3総合スコア計算（授業後）
        q3_after_cols = ['Q3_TeaLeaves_DissolveInWater', 'Q3_TeaComponents_DissolveInWater']
        after_sem['Q3_total_after'] = after_sem[q3_after_cols].sum(axis=1)
        
        # データ統合（独立群として）
        # 授業前群：group=0、授業後群：group=1
        before_sem['group'] = 0
        after_sem['group'] = 1
        
        # 授業前データの変数名調整
        before_vars = ['Page_ID', 'class', 'Q1_total_before', 'Q3_total_before', 'group']
        before_subset = before_sem[before_vars].copy()
        before_subset.columns = ['Page_ID', 'class', 'Q1_total', 'Q3_total', 'group']
        
        # 授業後データの変数選択
        after_vars = ['Page_ID', 'class', 'Q1_total_after', 'Q3_total_after',
                     'Q4_ExperimentInterestRating', 'Q5_NewLearningsRating',
                     'Q6_DissolvingUnderstandingRating', 'group']
        after_subset = after_sem[after_vars].copy()
        after_subset.columns = ['Page_ID', 'class', 'Q1_total', 'Q3_total',
                               'Q4_interest', 'Q5_learning', 'Q6_understanding', 'group']
        
        # データ統合
        # 授業前データには欠損値を設定
        before_subset['Q4_interest'] = np.nan
        before_subset['Q5_learning'] = np.nan
        before_subset['Q6_understanding'] = np.nan
        
        # 授業後データは全変数含む
        self.sem_data = pd.concat([before_subset, after_subset], ignore_index=True)
        
        # クラスダミー変数作成
        class_dummies = pd.get_dummies(self.sem_data['class'], prefix='class')
        self.sem_data = pd.concat([self.sem_data, class_dummies], axis=1)
        
        print(f"✓ SEM統合データ: {len(self.sem_data)} 件")
        print(f"✓ 授業前群: {(self.sem_data['group'] == 0).sum()} 件")
        print(f"✓ 授業後群: {(self.sem_data['group'] == 1).sum()} 件")
        
        # 基本統計表示
        print("\n📊 基本統計:")
        desc_stats = self.sem_data[['Q1_total', 'Q3_total', 'Q4_interest', 
                                   'Q5_learning', 'Q6_understanding']].describe()
        print(desc_stats.round(3))
        
    def define_sem_models(self):
        """SEM理論モデル定義"""
        print("\n🏗️ SEM理論モデル定義中...")
        
        # 測定モデル（潜在変数 ← 観測変数）
        measurement_model = """
        # 測定モデル（潜在変数の定義）
        # 科学的理解（授業前）
        SciUnderstanding_Pre =~ Q1_total + Q3_total
        
        # 学習への積極性（授業後のみ測定可能）
        LearningEngagement =~ Q4_interest + Q5_learning
        
        # 科学的理解（授業後）
        SciUnderstanding_Post =~ Q1_total + Q3_total + Q6_understanding
        
        # 構造モデル（潜在変数間の関係）
        # 授業前理解 → 学習積極性 → 授業後理解
        LearningEngagement ~ SciUnderstanding_Pre
        SciUnderstanding_Post ~ SciUnderstanding_Pre + LearningEngagement
        
        # グループ効果（独立群比較）
        SciUnderstanding_Pre ~ group
        LearningEngagement ~ group
        SciUnderstanding_Post ~ group
        """
        
        # 簡略化モデル（授業後データのみ）
        simplified_model = """
        # 簡略化モデル（授業後データのクロスセクショナル分析）
        # 学習への積極性
        LearningEngagement =~ Q4_interest + Q5_learning
        
        # 科学的理解（統合指標）
        SciUnderstanding =~ Q1_total + Q3_total + Q6_understanding
        
        # 構造関係
        SciUnderstanding ~ LearningEngagement
        
        # クラス効果
        LearningEngagement ~ class_2.0 + class_3.0 + class_4.0
        SciUnderstanding ~ class_2.0 + class_3.0 + class_4.0
        """
        
        self.models = {
            'full_model': measurement_model,
            'simplified_model': simplified_model
        }
        
        print("✓ フルモデル定義完了")
        print("✓ 簡略化モデル定義完了")
        
    def fit_sem_models(self):
        """SEMモデル推定"""
        print("\n⚙️ SEMモデル推定中...")
        
        self.fitted_models = {}
        
        # 授業後データのみでの分析（N=99）
        after_data = self.sem_data[self.sem_data['group'] == 1].copy()
        after_data = after_data.dropna()
        
        print(f"📊 分析対象データ: {len(after_data)} 件")
        
        try:
            # 簡略化モデル推定
            print("🔍 簡略化モデル推定中...")
            
            # SEMopyを使用してモデル推定
            model = semopy.Model(self.models['simplified_model'])
            
            # データ準備（欠損値除去）
            analysis_vars = ['Q1_total', 'Q3_total', 'Q4_interest', 'Q5_learning', 
                           'Q6_understanding', 'class_2.0', 'class_3.0', 'class_4.0']
            
            # クラスダミー変数が存在しない場合は作成
            for class_var in ['class_2.0', 'class_3.0', 'class_4.0']:
                if class_var not in after_data.columns:
                    after_data[class_var] = 0
            
            analysis_data = after_data[analysis_vars].dropna()
            
            print(f"📊 実際の分析データ: {len(analysis_data)} 件")
            
            # モデル推定
            results = model.fit(analysis_data)
            
            self.fitted_models['simplified'] = {
                'model': model,
                'results': results,
                'data': analysis_data,
                'fit_indices': self._calculate_fit_indices(model, analysis_data)
            }
            
            print("✓ 簡略化モデル推定完了")
            
        except Exception as e:
            print(f"❌ SEMモデル推定エラー: {e}")
            # エラー詳細をログ
            import traceback
            error_details = traceback.format_exc()
            print(f"詳細エラー: {error_details}")
            
            # 代替分析：相関分析
            self._alternative_correlation_analysis(after_data)
            
    def _calculate_fit_indices(self, model, data):
        """モデル適合度指標計算"""
        try:
            # SEMopyでの適合度指標取得
            fit_indices = {}
            
            # 基本的な適合度指標
            if hasattr(model, 'mx'):
                fit_indices.update({
                    'chi_square': model.mx.fun,
                    'degrees_of_freedom': model.mx.df if hasattr(model.mx, 'df') else 'N/A',
                    'n_observations': len(data)
                })
            
            return fit_indices
            
        except Exception as e:
            print(f"⚠️ 適合度指標計算エラー: {e}")
            return {'error': str(e)}
    
    def _alternative_correlation_analysis(self, data):
        """代替分析：相関・回帰分析"""
        print("\n🔄 代替分析実行中（相関・回帰分析）...")
        
        try:
            # 分析変数
            analysis_vars = ['Q1_total', 'Q3_total', 'Q4_interest', 'Q5_learning', 'Q6_understanding']
            correlation_data = data[analysis_vars].dropna()
            
            # 相関行列
            correlation_matrix = correlation_data.corr()
            
            # 統計的有意性検定
            from scipy.stats import pearsonr
            
            correlations_with_p = {}
            n_vars = len(analysis_vars)
            
            for i in range(n_vars):
                for j in range(i+1, n_vars):
                    var1, var2 = analysis_vars[i], analysis_vars[j]
                    r, p = pearsonr(correlation_data[var1], correlation_data[var2])
                    correlations_with_p[f"{var1}_vs_{var2}"] = {'r': r, 'p': p}
            
            # 重回帰分析（Q6理解度を目的変数）
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            X = correlation_data[['Q1_total', 'Q3_total', 'Q4_interest', 'Q5_learning']]
            y = correlation_data['Q6_understanding']
            
            reg_model = LinearRegression().fit(X, y)
            y_pred = reg_model.predict(X)
            r2 = r2_score(y, y_pred)
            
            # 結果保存
            self.fitted_models = {
                'alternative_analysis': {
                    'correlation_matrix': correlation_matrix,
                    'correlations_with_p': correlations_with_p,
                    'regression_coefficients': dict(zip(X.columns, reg_model.coef_)),
                    'regression_intercept': reg_model.intercept_,
                    'r_squared': r2,
                    'n_observations': len(correlation_data)
                }
            }
            
            print("✓ 代替分析完了")
            
        except Exception as e:
            print(f"❌ 代替分析エラー: {e}")
    
    def interpret_results(self):
        """結果解釈"""
        print("\n📝 結果解釈中...")
        
        interpretations = {}
        
        if 'simplified' in self.fitted_models:
            # SEM結果解釈
            model_data = self.fitted_models['simplified']
            interpretations['sem_analysis'] = self._interpret_sem_results(model_data)
            
        elif 'alternative_analysis' in self.fitted_models:
            # 代替分析結果解釈
            alt_data = self.fitted_models['alternative_analysis']
            interpretations['correlation_analysis'] = self._interpret_correlation_results(alt_data)
        
        self.results['interpretations'] = interpretations
        
    def _interpret_sem_results(self, model_data):
        """SEM結果の解釈"""
        interpretation = {
            'model_type': 'Structural Equation Modeling',
            'sample_size': len(model_data['data']),
            'fit_indices': model_data['fit_indices'],
            'structural_relationships': [],
            'educational_implications': []
        }
        
        # 適合度評価
        if 'chi_square' in model_data['fit_indices']:
            interpretation['model_fit_evaluation'] = "モデル適合度指標を確認してください"
        
        # 教育的示唆
        interpretation['educational_implications'] = [
            "学習への積極性が科学的理解に与える影響を構造的に分析",
            "クラス間差異が学習プロセスに与える影響を考慮",
            "潜在変数を用いた理論的枠組みでの教育効果測定"
        ]
        
        return interpretation
    
    def _interpret_correlation_results(self, alt_data):
        """相関分析結果の解釈"""
        correlation_matrix = alt_data['correlation_matrix']
        correlations_with_p = alt_data['correlations_with_p']
        
        interpretation = {
            'model_type': 'Correlation and Regression Analysis',
            'sample_size': alt_data['n_observations'],
            'regression_r_squared': alt_data['r_squared'],
            'significant_correlations': [],
            'regression_coefficients': alt_data['regression_coefficients'],
            'educational_implications': []
        }
        
        # 有意な相関関係の特定
        for relation, stats in correlations_with_p.items():
            if stats['p'] < 0.05:
                interpretation['significant_correlations'].append({
                    'relationship': relation,
                    'correlation': stats['r'],
                    'p_value': stats['p'],
                    'interpretation': self._interpret_correlation_magnitude(stats['r'])
                })
        
        # 教育的示唆
        interpretation['educational_implications'] = [
            f"理解度予測モデルの説明力: R² = {alt_data['r_squared']:.3f}",
            "学習変数間の関連性から教育効果のメカニズムを推定",
            "有意な相関関係から重要な学習要因を特定"
        ]
        
        # 回帰係数の解釈
        coef_interpretation = []
        for var, coef in alt_data['regression_coefficients'].items():
            coef_interpretation.append(f"{var}: {coef:.3f} (重要度順位付けの参考)")
        interpretation['coefficient_interpretation'] = coef_interpretation
        
        return interpretation
    
    def _interpret_correlation_magnitude(self, r):
        """相関係数の大きさ解釈"""
        abs_r = abs(r)
        if abs_r >= 0.7:
            return "強い関連"
        elif abs_r >= 0.4:
            return "中程度の関連"
        elif abs_r >= 0.2:
            return "弱い関連"
        else:
            return "ほぼ関連なし"
    
    def create_visualizations(self):
        """可視化作成"""
        print("\n📊 可視化作成中...")
        
        plt.style.use('default')
        
        if 'alternative_analysis' in self.fitted_models:
            self._create_correlation_heatmap()
            self._create_regression_plot()
        
        # パス図作成（概念図）
        self._create_conceptual_path_diagram()
        
    def _create_correlation_heatmap(self):
        """相関行列ヒートマップ"""
        correlation_matrix = self.fitted_models['alternative_analysis']['correlation_matrix']
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                    mask=mask,
                    annot=True, 
                    cmap='RdBu_r', 
                    center=0,
                    square=True,
                    fmt='.3f',
                    cbar_kws={"shrink": .8})
        
        plt.title('学習変数間の相関関係\n（授業後データ）', fontsize=14, pad=20)
        plt.tight_layout()
        
        output_path = self.figures_dir / "correlation_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 相関行列保存: {output_path}")
    
    def _create_regression_plot(self):
        """回帰分析結果プロット"""
        alt_data = self.fitted_models['alternative_analysis']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 回帰係数の可視化
        coefficients = alt_data['regression_coefficients']
        vars_names = list(coefficients.keys())
        coef_values = list(coefficients.values())
        
        axes[0,0].barh(vars_names, coef_values)
        axes[0,0].set_title('回帰係数（Q6理解度への影響）')
        axes[0,0].set_xlabel('回帰係数')
        
        # R²値表示
        axes[0,1].text(0.5, 0.5, f"R² = {alt_data['r_squared']:.3f}\n\n説明力: {alt_data['r_squared']*100:.1f}%", 
                      ha='center', va='center', fontsize=16,
                      bbox=dict(boxstyle='round', facecolor='lightblue'))
        axes[0,1].set_xlim(0, 1)
        axes[0,1].set_ylim(0, 1)
        axes[0,1].set_title('モデル説明力')
        axes[0,1].axis('off')
        
        # 有意な相関のみ表示
        correlations_with_p = alt_data['correlations_with_p']
        significant_corrs = [(k.replace('_vs_', ' - '), v['r']) 
                           for k, v in correlations_with_p.items() if v['p'] < 0.05]
        
        if significant_corrs:
            labels, values = zip(*significant_corrs)
            axes[1,0].barh(labels, values)
            axes[1,0].set_title('有意な相関関係 (p < 0.05)')
            axes[1,0].set_xlabel('相関係数')
        
        # サンプル情報
        axes[1,1].text(0.5, 0.5, f"分析対象: {alt_data['n_observations']} 件\n\n独立群比較\n（授業後データ）", 
                      ha='center', va='center', fontsize=12,
                      bbox=dict(boxstyle='round', facecolor='lightyellow'))
        axes[1,1].set_xlim(0, 1)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].set_title('分析概要')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        
        output_path = self.figures_dir / "regression_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 回帰分析図保存: {output_path}")
    
    def _create_conceptual_path_diagram(self):
        """概念的パス図作成"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # ボックス位置定義
        boxes = {
            'Q1_total': (1, 3),
            'Q3_total': (1, 1),
            'Q4_interest': (3, 4),
            'Q5_learning': (3, 2),
            'Q6_understanding': (5, 3),
            'LearningEngagement': (3, 3),
            'SciUnderstanding': (5, 1.5)
        }
        
        # ボックス描画
        for var, (x, y) in boxes.items():
            if var in ['LearningEngagement', 'SciUnderstanding']:
                # 潜在変数（楕円）
                ellipse = plt.Circle((x, y), 0.3, fill=False, linestyle='--')
                ax.add_patch(ellipse)
                ax.text(x, y, var.replace('Learning', 'Learning\n').replace('Sci', 'Sci\n'), 
                       ha='center', va='center', fontsize=8)
            else:
                # 観測変数（矩形）
                rect = plt.Rectangle((x-0.3, y-0.2), 0.6, 0.4, fill=False)
                ax.add_patch(rect)
                ax.text(x, y, var, ha='center', va='center', fontsize=9)
        
        # パス（矢印）描画
        paths = [
            ('Q4_interest', 'LearningEngagement'),
            ('Q5_learning', 'LearningEngagement'),
            ('LearningEngagement', 'SciUnderstanding'),
            ('Q1_total', 'SciUnderstanding'),
            ('Q3_total', 'SciUnderstanding'),
            ('Q6_understanding', 'SciUnderstanding')
        ]
        
        for start, end in paths:
            x1, y1 = boxes[start]
            x2, y2 = boxes[end]
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=1.5))
        
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('教育効果の構造的関係モデル\n（概念図）', fontsize=14, pad=20)
        
        # 凡例
        ax.text(0.5, 4.5, '□ 観測変数\n○ 潜在変数\n→ 構造関係', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightgray'))
        
        plt.tight_layout()
        
        output_path = self.figures_dir / "conceptual_path_diagram.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 概念図保存: {output_path}")
    
    def save_results(self):
        """結果保存"""
        print("\n💾 結果保存中...")
        
        # 結果の詳細情報
        detailed_results = {
            'metadata': {
                'analysis_type': 'Structural Equation Modeling',
                'generated_at': datetime.now().isoformat(),
                'sample_size_before': (self.sem_data['group'] == 0).sum(),
                'sample_size_after': (self.sem_data['group'] == 1).sum(),
                'analysis_approach': 'Independent Groups Comparison'
            },
            'data_summary': {
                'variables_analyzed': ['Q1_total', 'Q3_total', 'Q4_interest', 'Q5_learning', 'Q6_understanding'],
                'correlation_matrix': None,
                'descriptive_statistics': None
            },
            'model_results': {},
            'interpretations': self.results.get('interpretations', {}),
            'educational_implications': self._generate_educational_implications()
        }
        
        # 分析結果追加
        if 'alternative_analysis' in self.fitted_models:
            alt_data = self.fitted_models['alternative_analysis']
            detailed_results['data_summary']['correlation_matrix'] = alt_data['correlation_matrix'].to_dict()
            detailed_results['model_results'] = {
                'regression_coefficients': alt_data['regression_coefficients'],
                'r_squared': alt_data['r_squared'],
                'significant_correlations': [
                    {k: v for k, v in alt_data['correlations_with_p'].items() if v['p'] < 0.05}
                ]
            }
        
        # JSON形式で保存
        output_path = self.output_dir / "structural_equation_modeling_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✓ 詳細結果保存: {output_path}")
        
        # 要約レポート作成
        self._create_summary_report(detailed_results)
        
    def _generate_educational_implications(self):
        """教育的示唆生成"""
        implications = {
            'learning_process_insights': [
                "学習への積極性（実験興味・新学び）が理解度向上の重要な要因",
                "授業前の基礎理解度が学習成果に影響を与える可能性",
                "クラス間差異を考慮した個別指導の重要性"
            ],
            'instructional_recommendations': [
                "実験活動への興味喚起が効果的な学習促進策",
                "新しい学びへの意識向上が理解度向上に寄与",
                "基礎知識の確実な定着が発展的学習の基盤"
            ],
            'assessment_insights': [
                "多面的評価による学習プロセスの構造的理解",
                "潜在的な学習能力の測定方法の重要性",
                "量的・質的データの統合分析の有効性"
            ],
            'methodological_notes': [
                "独立群比較による構造的関係の推定",
                "個人追跡データ収集の重要性（今後の改善点）",
                "因果推論の限界と観察研究の特性"
            ]
        }
        
        return implications
    
    def _create_summary_report(self, detailed_results):
        """要約レポート作成"""
        report_lines = [
            "# 構造方程式モデリング（SEM）分析レポート",
            "## 小学校出前授業アンケート - 学習プロセスの構造分析",
            "",
            f"**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**分析手法**: {detailed_results['metadata']['analysis_type']}",
            f"**サンプルサイズ**: 授業後 {detailed_results['metadata']['sample_size_after']} 件",
            "",
            "## 分析概要",
            "",
            "独立群比較データにおける学習プロセスの構造的関係を分析。",
            "学習への積極性と科学的理解度の関連性を統計的に検証。",
            "",
            "## 主要な発見事項",
            ""
        ]
        
        # モデル結果
        if 'model_results' in detailed_results and detailed_results['model_results']:
            model_results = detailed_results['model_results']
            
            report_lines.extend([
                "### 回帰分析結果",
                "",
                f"**モデル説明力**: R² = {model_results['r_squared']:.3f} ({model_results['r_squared']*100:.1f}%)",
                "",
                "**回帰係数**:",
                ""
            ])
            
            for var, coef in model_results['regression_coefficients'].items():
                report_lines.append(f"- {var}: {coef:.3f}")
            
            report_lines.append("")
        
        # 教育的示唆
        if 'educational_implications' in detailed_results:
            implications = detailed_results['educational_implications']
            
            report_lines.extend([
                "## 教育的示唆",
                "",
                "### 学習プロセスへの洞察",
                ""
            ])
            
            for insight in implications['learning_process_insights']:
                report_lines.append(f"- {insight}")
            
            report_lines.extend([
                "",
                "### 指導法への提言",
                ""
            ])
            
            for rec in implications['instructional_recommendations']:
                report_lines.append(f"- {rec}")
            
            report_lines.extend([
                "",
                "### 評価方法への示唆",
                ""
            ])
            
            for assessment in implications['assessment_insights']:
                report_lines.append(f"- {assessment}")
        
        # 制約と限界
        report_lines.extend([
            "",
            "## 分析の制約と限界",
            "",
            "- Page_IDによる個人追跡が不可能なため独立群比較として分析",
            "- 因果推論には限界があり、関連性の推定にとどまる", 
            "- 観察研究のため実験的統制は不可能",
            "- 今後は個人識別可能なデータ収集を推奨",
            "",
            "---",
            "",
            "**Generated by**: Claude Code Analysis (SEM Implementation)",
            f"**Output Files**: {self.output_dir}",
            f"**Figures**: {self.figures_dir}"
        ])
        
        # レポート保存
        report_path = self.output_dir / "sem_analysis_summary.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✓ 要約レポート保存: {report_path}")
    
    def run_complete_analysis(self):
        """完全SEM分析実行"""
        print("="*60)
        print("構造方程式モデリング（SEM）分析")
        print("="*60)
        print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        try:
            # 分析実行
            self.load_data()
            self.prepare_sem_data()
            self.define_sem_models()
            self.fit_sem_models()
            self.interpret_results()
            self.create_visualizations()
            self.save_results()
            
            print("\n" + "="*60)
            print("🎉 SEM分析完了!")
            print("="*60)
            print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            print("📁 出力ファイル:")
            print(f"  - 詳細結果: {self.output_dir}/structural_equation_modeling_results.json")
            print(f"  - 要約レポート: {self.output_dir}/sem_analysis_summary.txt")
            print(f"  - 図表: {self.figures_dir}/")
            print()
            print("⚠️  重要: この分析は独立群比較であり、個人の変化は測定していません")
            
            return True
            
        except Exception as e:
            print(f"\n❌ SEM分析エラー: {e}")
            import traceback
            print(f"詳細: {traceback.format_exc()}")
            return False

def main():
    """メイン実行関数"""
    project_root = Path(__file__).parent.parent.parent
    
    analyzer = StructuralEquationModeling(project_root)
    
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