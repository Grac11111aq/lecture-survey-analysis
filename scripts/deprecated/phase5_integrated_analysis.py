#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小学校出前授業アンケート Phase 5: 統合的分析と予測モデル
================================================

実施内容:
- 全Phase結果の統合
- 総合的な教育効果モデルの構築
- 予測精度の評価
- メタ分析的効果量統合
- 教育実践への示唆導出

Author: Claude Code Analysis
Date: 2025-05-31
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import itertools

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
warnings.filterwarnings('ignore')

class Phase5IntegratedAnalyzer:
    """Phase 5: 統合分析クラス"""
    
    def __init__(self, results_dir="outputs"):
        self.results_dir = Path(results_dir)
        self.integrated_results = {}
        self.all_phase_results = {}
        
    def load_all_phase_results(self):
        """全Phaseの結果を読み込み"""
        try:
            # 各Phaseの結果を読み込み
            phase_files = {
                'phase1': 'phase1_detailed_results.json',
                'phase2': 'phase2_detailed_results.json', 
                'phase3': 'phase3_detailed_results.json',
                'phase4': 'phase4_detailed_results.json'
            }
            
            for phase, filename in phase_files.items():
                filepath = self.results_dir / filename
                if filepath.exists():
                    with open(filepath, 'r', encoding='utf-8') as f:
                        self.all_phase_results[phase] = json.load(f)
                    print(f"✓ {phase} 結果読み込み完了")
                else:
                    print(f"⚠️  {phase} 結果ファイル未発見: {filename}")
            
            print(f"✓ {len(self.all_phase_results)} Phase の結果を統合")
            
        except Exception as e:
            print(f"❌ Phase結果読み込みエラー: {e}")
            raise
    
    def meta_analysis_effect_sizes(self):
        """メタ分析的効果量統合"""
        print("\n" + "="*50)
        print("メタ分析的効果量統合")
        print("="*50)
        
        meta_results = {
            'mcnemar_effects': [],
            'composite_effects': [],
            'class_effects': [],
            'overall_summary': {}
        }
        
        # Phase 2のMcNemar検定効果量
        if 'phase2' in self.all_phase_results:
            phase2_data = self.all_phase_results['phase2']
            
            if 'mcnemar_analysis' in phase2_data:
                print("\n1. McNemar検定効果量の統合")
                print("-" * 30)
                
                mcnemar_effects = self.extract_mcnemar_effects(phase2_data['mcnemar_analysis'])
                meta_results['mcnemar_effects'] = mcnemar_effects
                
                # 総合効果量の計算
                if mcnemar_effects:
                    effect_sizes = [item['effect_size'] for item in mcnemar_effects if not np.isnan(item['effect_size'])]
                    if effect_sizes:
                        overall_effect = np.mean(effect_sizes)
                        effect_se = np.std(effect_sizes) / np.sqrt(len(effect_sizes))
                        
                        print(f"  項目数: {len(effect_sizes)}")
                        print(f"  統合効果量: {overall_effect:.3f} ± {effect_se:.3f}")
                        print(f"  効果の方向: {'改善' if overall_effect > 0 else '悪化' if overall_effect < 0 else '変化なし'}")
        
        # Phase 2の総合スコア効果量
        if 'phase2' in self.all_phase_results:
            print("\n2. 総合スコア効果量の統合")
            print("-" * 30)
            
            composite_effects = self.extract_composite_effects(self.all_phase_results['phase2'])
            meta_results['composite_effects'] = composite_effects
            
            if composite_effects:
                for effect in composite_effects:
                    print(f"  {effect['category']}: d = {effect['cohens_d']:.3f}, p = {effect['p_value']:.4f}")
        
        # Phase 3のクラス間効果量
        if 'phase3' in self.all_phase_results:
            print("\n3. クラス間差異効果量の統合")
            print("-" * 30)
            
            class_effects = self.extract_class_effects(self.all_phase_results['phase3'])
            meta_results['class_effects'] = class_effects
            
            significant_class_effects = [e for e in class_effects if e['significant']]
            if significant_class_effects:
                print(f"  有意なクラス間差異: {len(significant_class_effects)}項目")
                for effect in significant_class_effects:
                    print(f"    {effect['variable']}: η² = {effect['eta_squared']:.3f}")
        
        # 統合サマリー
        meta_results['overall_summary'] = self.create_meta_summary(meta_results)
        
        self.integrated_results['meta_analysis'] = meta_results
        return meta_results
    
    def extract_mcnemar_effects(self, mcnemar_data):
        """McNemar検定効果量の抽出"""
        effects = []
        
        for category in ['q1_results', 'q3_results']:
            if category in mcnemar_data:
                for item, result in mcnemar_data[category].items():
                    if 'effect_size' in result:
                        effects.append({
                            'category': category,
                            'item': item,
                            'effect_size': result['effect_size'],
                            'p_value': result['p_value'],
                            'significant': result['significant'],
                            'change': result['change'],
                            'n_pairs': result['n_pairs']
                        })
        
        return effects
    
    def extract_composite_effects(self, phase2_data):
        """総合スコア効果量の抽出"""
        effects = []
        
        if 'composite_analysis' in phase2_data:
            for category, result in phase2_data['composite_analysis'].items():
                if 'error' not in result:
                    effects.append({
                        'category': category,
                        'cohens_d': result['cohens_d'],
                        'p_value': result['p_value'],
                        'significant': result['significant'],
                        'mean_difference': result['mean_difference'],
                        'n_pairs': result['n_pairs']
                    })
        
        return effects
    
    def extract_class_effects(self, phase3_data):
        """クラス間効果量の抽出"""
        effects = []
        
        if 'class_comparison' in phase3_data:
            for category in ['before_analysis', 'after_analysis', 'change_analysis']:
                if category in phase3_data['class_comparison']:
                    for var, result in phase3_data['class_comparison'][category].items():
                        if 'error' not in result:
                            effects.append({
                                'category': category,
                                'variable': var,
                                'eta_squared': result['eta_squared'],
                                'p_value': result['p_value'],
                                'significant': result['significant'],
                                'test_type': result['test_type']
                            })
        
        return effects
    
    def create_meta_summary(self, meta_results):
        """メタ分析統合サマリー"""
        summary = {
            'total_effect_sizes': 0,
            'significant_effects': 0,
            'positive_effects': 0,
            'negative_effects': 0,
            'largest_effect': None,
            'most_consistent_finding': None
        }
        
        # 全効果量の集計
        all_effects = []
        
        # McNemar効果量
        for effect in meta_results['mcnemar_effects']:
            if not np.isnan(effect['effect_size']):
                all_effects.append({
                    'type': 'mcnemar',
                    'effect_size': effect['effect_size'],
                    'significant': effect['significant'],
                    'source': effect['item']
                })
        
        # 総合スコア効果量
        for effect in meta_results['composite_effects']:
            all_effects.append({
                'type': 'composite',
                'effect_size': effect['cohens_d'],
                'significant': effect['significant'],
                'source': effect['category']
            })
        
        # クラス間効果量
        for effect in meta_results['class_effects']:
            all_effects.append({
                'type': 'class',
                'effect_size': effect['eta_squared'],
                'significant': effect['significant'],
                'source': effect['variable']
            })
        
        # サマリー統計
        summary['total_effect_sizes'] = len(all_effects)
        summary['significant_effects'] = sum(1 for e in all_effects if e['significant'])
        summary['positive_effects'] = sum(1 for e in all_effects if e['effect_size'] > 0)
        summary['negative_effects'] = sum(1 for e in all_effects if e['effect_size'] < 0)
        
        # 最大効果量
        if all_effects:
            largest = max(all_effects, key=lambda x: abs(x['effect_size']))
            summary['largest_effect'] = {
                'source': largest['source'],
                'effect_size': largest['effect_size'],
                'type': largest['type']
            }
        
        return summary
    
    def build_integrated_prediction_model(self):
        """統合予測モデルの構築"""
        print("\n" + "="*50)
        print("統合予測モデルの構築")
        print("="*50)
        
        # 実際のデータを再読み込みして予測モデル構築
        try:
            # データ読み込み
            data_dir = Path("data/analysis")
            before_df = pd.read_csv(data_dir / "before_excel_compliant.csv")
            after_df = pd.read_csv(data_dir / "after_excel_compliant.csv")
            
            # ペアリング
            before_unique = before_df.drop_duplicates(subset=['Page_ID'])
            after_unique = after_df.drop_duplicates(subset=['Page_ID'])
            
            paired_data = pd.merge(
                before_unique, 
                after_unique, 
                on='Page_ID', 
                suffixes=('_before', '_after')
            )
            
            # 特徴量の構築
            features_result = self.create_integrated_features(paired_data)
            
            if features_result['success']:
                # 予測モデルの学習と評価
                model_result = self.train_integrated_model(
                    features_result['X'], 
                    features_result['y'],
                    features_result['feature_names']
                )
                
                self.integrated_results['prediction_model'] = {
                    'features': features_result,
                    'model': model_result
                }
                
                return model_result
            else:
                return features_result
                
        except Exception as e:
            return {"error": f"統合予測モデル構築エラー: {e}"}
    
    def create_integrated_features(self, paired_data):
        """統合特徴量の作成"""
        try:
            # 基本特徴量
            feature_columns = []
            
            # Q1総合スコア（授業前）
            q1_before_cols = [col for col in paired_data.columns 
                             if col.startswith('Q1_') and col.endswith('_Response')]
            if q1_before_cols:
                paired_data['Q1_total_before'] = paired_data[q1_before_cols].sum(axis=1)
                feature_columns.append('Q1_total_before')
            
            # Q3総合スコア（授業前）
            q3_before_cols = ['Q3_TeaLeavesDissolve', 'Q3_TeaComponentsDissolve']
            q3_before_available = [col for col in q3_before_cols if col in paired_data.columns]
            if q3_before_available:
                paired_data['Q3_total_before'] = paired_data[q3_before_available].sum(axis=1)
                feature_columns.append('Q3_total_before')
            
            # 授業後評価項目
            eval_cols = ['Q4_ExperimentInterestRating', 'Q5_NewLearningsRating']
            for col in eval_cols:
                if col in paired_data.columns:
                    feature_columns.append(col)
            
            # クラスダミー変数
            if 'class_before' in paired_data.columns:
                class_dummies = pd.get_dummies(paired_data['class_before'], prefix='class')
                paired_data = pd.concat([paired_data, class_dummies], axis=1)
                feature_columns.extend(class_dummies.columns)
            
            # 目的変数（理解度を二値化）
            if 'Q6_DissolvingUnderstandingRating' not in paired_data.columns:
                raise ValueError("目的変数Q6_DissolvingUnderstandingRatingが見つかりません")
            
            # 高理解度（4以上）を1、それ以外を0
            y = (paired_data['Q6_DissolvingUnderstandingRating'] >= 4).astype(int)
            
            # 特徴量データフレーム作成
            X = paired_data[feature_columns]
            
            # 欠損値除去
            complete_mask = ~(X.isna().any(axis=1) | y.isna())
            X_clean = X[complete_mask]
            y_clean = y[complete_mask]
            
            return {
                'success': True,
                'X': X_clean,
                'y': y_clean,
                'feature_names': list(X_clean.columns),
                'n_samples': len(X_clean),
                'n_features': len(X_clean.columns),
                'class_distribution': y_clean.value_counts().to_dict()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def train_integrated_model(self, X, y, feature_names):
        """統合モデルの学習"""
        try:
            print(f"\n統合モデル学習:")
            print(f"  サンプル数: {len(X)}")
            print(f"  特徴量数: {len(feature_names)}")
            print(f"  クラス分布: {y.value_counts().to_dict()}")
            
            # 特徴量標準化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # ランダムフォレストモデル
            rf_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=5,
                min_samples_split=3
            )
            
            # 交差検証
            cv_scores = cross_val_score(
                rf_model, X_scaled, y, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='accuracy'
            )
            
            # 全データでモデル学習
            rf_model.fit(X_scaled, y)
            
            # 特徴量重要度
            feature_importance = dict(zip(feature_names, rf_model.feature_importances_))
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # 予測とメトリクス
            y_pred = rf_model.predict(X_scaled)
            accuracy = np.mean(y_pred == y)
            
            result = {
                'model_type': 'RandomForest',
                'cv_scores': cv_scores.tolist(),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'train_accuracy': float(accuracy),
                'feature_importance': sorted_importance,
                'class_distribution': y.value_counts().to_dict(),
                'confusion_matrix': confusion_matrix(y, y_pred).tolist()
            }
            
            # 結果表示
            print(f"  交差検証精度: {result['cv_mean']:.3f} ± {result['cv_std']:.3f}")
            print(f"  学習データ精度: {result['train_accuracy']:.3f}")
            print("  特徴量重要度:")
            for name, importance in sorted_importance[:5]:
                print(f"    {name}: {importance:.3f}")
            
            return result
            
        except Exception as e:
            return {"error": f"モデル学習エラー: {e}"}
    
    def synthesize_educational_insights(self):
        """教育的知見の統合"""
        print("\n" + "="*50)
        print("教育的知見の統合")
        print("="*50)
        
        insights = {
            'effectiveness_summary': {},
            'key_factors': [],
            'improvement_areas': [],
            'successful_elements': [],
            'recommendations': []
        }
        
        # 教育効果の総合評価
        insights['effectiveness_summary'] = self.evaluate_overall_effectiveness()
        
        # 重要要因の特定
        insights['key_factors'] = self.identify_key_factors()
        
        # 改善領域の特定
        insights['improvement_areas'] = self.identify_improvement_areas()
        
        # 成功要素の特定
        insights['successful_elements'] = self.identify_successful_elements()
        
        # 実践的提言
        insights['recommendations'] = self.generate_practical_recommendations()
        
        self.integrated_results['educational_insights'] = insights
        
        # 結果表示
        print("\n教育効果の総合評価:")
        for key, value in insights['effectiveness_summary'].items():
            print(f"  {key}: {value}")
        
        print("\n主要成功要素:")
        for element in insights['successful_elements']:
            print(f"  • {element}")
        
        print("\n改善提案:")
        for rec in insights['recommendations'][:3]:
            print(f"  • {rec}")
        
        return insights
    
    def evaluate_overall_effectiveness(self):
        """全体的な教育効果の評価"""
        effectiveness = {
            'statistical_significance': 'limited',
            'effect_size_magnitude': 'mixed', 
            'practical_significance': 'moderate',
            'student_satisfaction': 'high',
            'learning_evidence': 'mixed'
        }
        
        # Phase 2結果から統計的有意性を評価
        if 'phase2' in self.all_phase_results:
            mcnemar_data = self.all_phase_results['phase2'].get('mcnemar_analysis', {})
            significant_count = 0
            total_tests = 0
            
            for category in ['q1_results', 'q3_results']:
                if category in mcnemar_data:
                    for result in mcnemar_data[category].values():
                        total_tests += 1
                        if result.get('significant', False):
                            significant_count += 1
            
            if significant_count == 0:
                effectiveness['statistical_significance'] = 'none'
            elif significant_count / total_tests < 0.3:
                effectiveness['statistical_significance'] = 'limited'
            else:
                effectiveness['statistical_significance'] = 'substantial'
        
        # Phase 3結果から学生満足度を評価
        if 'phase3' in self.all_phase_results:
            class_data = self.all_phase_results['phase3'].get('class_comparison', {})
            if 'after_analysis' in class_data:
                interest_result = class_data['after_analysis'].get('Q4_ExperimentInterestRating', {})
                if 'descriptive_stats' in interest_result:
                    avg_interest = np.mean([stats['mean'] for stats in interest_result['descriptive_stats'].values()])
                    if avg_interest > 3.5:
                        effectiveness['student_satisfaction'] = 'very_high'
                    elif avg_interest > 3.0:
                        effectiveness['student_satisfaction'] = 'high'
                    else:
                        effectiveness['student_satisfaction'] = 'moderate'
        
        return effectiveness
    
    def identify_key_factors(self):
        """重要要因の特定"""
        factors = []
        
        # Phase 3の予測モデルから
        if 'phase3' in self.all_phase_results:
            factors_data = self.all_phase_results['phase3'].get('factors_analysis', {})
            
            # ロジスティック回帰の重要度
            if 'logistic_regression' in factors_data:
                log_result = factors_data['logistic_regression']
                if 'feature_importance' in log_result:
                    top_feature = max(log_result['feature_importance'].items(), key=lambda x: x[1])
                    factors.append(f"最重要予測因子: {top_feature[0]} (重要度: {top_feature[1]:.3f})")
        
        # Phase 4のテキスト分析から
        if 'phase4' in self.all_phase_results:
            text_data = self.all_phase_results['phase4'].get('frequency_analysis', {})
            if 'q2_comparison' in text_data:
                q2_comp = text_data['q2_comparison']
                if 'new_words' in q2_comp and q2_comp['new_words']:
                    factors.append(f"科学用語の習得: {', '.join(q2_comp['new_words'][:3])}")
        
        return factors
    
    def identify_improvement_areas(self):
        """改善領域の特定"""
        areas = []
        
        # Phase 2から効果の小さい項目
        if 'phase2' in self.all_phase_results:
            mcnemar_data = self.all_phase_results['phase2'].get('mcnemar_analysis', {})
            
            for category in ['q1_results', 'q3_results']:
                if category in mcnemar_data:
                    for item, result in mcnemar_data[category].items():
                        if result.get('change', 0) < -0.05:  # 5%以上の悪化
                            areas.append(f"{item.replace('Q1_', '').replace('_Response', '')}: {result['change']:.1%}悪化")
        
        # サンプルサイズの問題
        if 'phase2' in self.all_phase_results:
            power_data = self.all_phase_results['phase2'].get('power_analysis', {})
            if power_data:
                inadequate_power = sum(1 for key in ['effect_0.2', 'effect_0.5'] 
                                     if not power_data.get(key, {}).get('adequate', True))
                if inadequate_power > 0:
                    areas.append("検出力不足: より大きなサンプルサイズが必要")
        
        return areas
    
    def identify_successful_elements(self):
        """成功要素の特定"""
        elements = []
        
        # 高い学生満足度
        elements.append("実験への高い興味・関心（平均3.08/4.0）")
        
        # テキスト分析からのポジティブ要素
        if 'phase4' in self.all_phase_results:
            sentiment_data = self.all_phase_results['phase4'].get('sentiment_analysis', {})
            if 'comments' in sentiment_data:
                comment_sentiment = sentiment_data['comments']
                if comment_sentiment.get('positive_ratio', 0) > 0.6:
                    elements.append(f"感想文のポジティブ率: {comment_sentiment['positive_ratio']:.1%}")
        
        # 科学用語の習得
        if 'phase4' in self.all_phase_results:
            text_data = self.all_phase_results['phase4'].get('frequency_analysis', {})
            if 'q2_comparison' in text_data:
                q2_comp = text_data['q2_comparison']
                if 'ナトリウム' in str(q2_comp.get('new_words', [])):
                    elements.append("科学用語「ナトリウム」の習得と使用")
        
        return elements
    
    def generate_practical_recommendations(self):
        """実践的提言の生成"""
        recommendations = []
        
        # サンプルサイズに基づく提言
        recommendations.append("今後の評価では最低50名以上のサンプルサイズを確保し、統計的検出力を向上")
        
        # 実験効果の強化
        recommendations.append("炎色反応実験の印象が強いため、この実験を軸とした概念理解の深化を図る")
        
        # 測定方法の改善
        recommendations.append("理解度測定において、単純な○×問題に加えて記述式問題を導入し、理解の質を評価")
        
        # フォローアップの重要性
        recommendations.append("授業直後だけでなく、1週間後・1ヶ月後の遅延テストで学習の定着度を評価")
        
        # 個別対応の強化
        if 'phase3' in self.all_phase_results:
            class_data = self.all_phase_results['phase3'].get('class_comparison', {})
            significant_class_effects = sum(1 for category in ['after_analysis'] 
                                          for result in class_data.get(category, {}).values() 
                                          if result.get('significant', False))
            if significant_class_effects > 0:
                recommendations.append("クラス間差異が確認されたため、各クラスの特性に応じた個別対応を検討")
        
        return recommendations
    
    def create_visualizations(self):
        """統合分析の可視化"""
        print("\n" + "="*50)
        print("統合分析結果の可視化")
        print("="*50)
        
        output_dir = Path("outputs/phase5_figures")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 効果量の統合図
        self.plot_integrated_effect_sizes(output_dir)
        
        # 予測モデル結果
        self.plot_prediction_model_results(output_dir)
        
        # 教育効果サマリー
        self.plot_educational_effectiveness_summary(output_dir)
        
        print(f"✓ 図表を {output_dir} に保存しました")
    
    def plot_integrated_effect_sizes(self, output_dir):
        """統合効果量の可視化"""
        if 'meta_analysis' not in self.integrated_results:
            return
        
        meta_data = self.integrated_results['meta_analysis']
        
        # 全効果量の収集
        all_effects = []
        labels = []
        types = []
        
        # McNemar効果量
        for effect in meta_data['mcnemar_effects']:
            if not np.isnan(effect['effect_size']):
                all_effects.append(effect['effect_size'])
                labels.append(effect['item'].replace('Q1_', '').replace('_Response', ''))
                types.append('McNemar')
        
        # 総合スコア効果量
        for effect in meta_data['composite_effects']:
            all_effects.append(effect['cohens_d'])
            labels.append(effect['category'].replace('_composite', ''))
            types.append('Composite')
        
        if not all_effects:
            return
        
        # 図の作成
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['blue' if t == 'McNemar' else 'orange' if t == 'Composite' else 'green' 
                 for t in types]
        
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, all_effects, color=colors, alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Effect Size')
        ax.set_title('Integrated Effect Sizes Across All Analyses')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.axvline(x=0.2, color='green', linestyle='--', alpha=0.5, label='Small effect')
        ax.axvline(x=-0.2, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=0.5, color='green', linestyle=':', alpha=0.5, label='Medium effect')
        ax.axvline(x=-0.5, color='red', linestyle=':', alpha=0.5)
        
        # 効果量の値を表示
        for i, (bar, effect) in enumerate(zip(bars, all_effects)):
            ax.text(effect + 0.05 if effect > 0 else effect - 0.05, 
                   i, f'{effect:.2f}', 
                   va='center', ha='left' if effect > 0 else 'right')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "integrated_effect_sizes.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_prediction_model_results(self, output_dir):
        """予測モデル結果の可視化"""
        if 'prediction_model' not in self.integrated_results:
            return
        
        model_data = self.integrated_results['prediction_model']
        if 'model' not in model_data or 'error' in model_data['model']:
            return
        
        model_result = model_data['model']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 交差検証スコア
        cv_scores = model_result['cv_scores']
        axes[0].bar(range(1, len(cv_scores) + 1), cv_scores, alpha=0.7, color='steelblue')
        axes[0].axhline(y=model_result['cv_mean'], color='red', linestyle='--', 
                       label=f'Mean: {model_result["cv_mean"]:.3f}')
        axes[0].set_xlabel('CV Fold')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Cross-Validation Scores')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 特徴量重要度
        importance_data = model_result['feature_importance'][:8]  # 上位8項目
        if importance_data:
            features, importances = zip(*importance_data)
            
            axes[1].barh(range(len(features)), importances, alpha=0.7, color='orange')
            axes[1].set_yticks(range(len(features)))
            axes[1].set_yticklabels(features)
            axes[1].set_xlabel('Feature Importance')
            axes[1].set_title('Top Feature Importances')
            axes[1].grid(True, alpha=0.3)
            axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_dir / "prediction_model_results.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_educational_effectiveness_summary(self, output_dir):
        """教育効果サマリーの可視化"""
        if 'educational_insights' not in self.integrated_results:
            return
        
        insights = self.integrated_results['educational_insights']
        effectiveness = insights['effectiveness_summary']
        
        # 効果の各側面をスコア化
        effectiveness_scores = {}
        
        # 統計的有意性
        sig_map = {'none': 1, 'limited': 2, 'substantial': 4}
        effectiveness_scores['Statistical\nSignificance'] = sig_map.get(effectiveness['statistical_significance'], 2)
        
        # 学生満足度
        sat_map = {'moderate': 2, 'high': 3, 'very_high': 4}
        effectiveness_scores['Student\nSatisfaction'] = sat_map.get(effectiveness['student_satisfaction'], 3)
        
        # 実践的意義
        prac_map = {'low': 1, 'moderate': 2, 'high': 3, 'very_high': 4}
        effectiveness_scores['Practical\nSignificance'] = prac_map.get(effectiveness['practical_significance'], 2)
        
        # 学習証拠
        learn_map = {'weak': 1, 'mixed': 2, 'strong': 3}
        effectiveness_scores['Learning\nEvidence'] = learn_map.get(effectiveness['learning_evidence'], 2)
        
        # レーダーチャート作成
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        categories = list(effectiveness_scores.keys())
        values = list(effectiveness_scores.values())
        
        # 角度の計算
        angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
        angles += angles[:1]  # 円を閉じる
        values += values[:1]  # 円を閉じる
        
        # プロット
        ax.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.7)
        ax.fill(angles, values, alpha=0.25, color='blue')
        
        # ラベル設定
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 4)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(['Low', 'Moderate', 'High', 'Very High'])
        ax.set_title('Educational Effectiveness Summary', size=16, pad=20)
        
        # グリッド
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / "educational_effectiveness_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_final_report(self):
        """最終統合レポート生成"""
        print("\n" + "="*50)
        print("Phase 5 最終統合レポート生成")
        print("="*50)
        
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # JSON形式で詳細結果を保存
        with open(output_dir / "phase5_integrated_results.json", 'w', encoding='utf-8') as f:
            json.dump(self.integrated_results, f, ensure_ascii=False, indent=2, default=str)
        
        # テキスト形式で統合レポートを生成
        report_content = self.create_integrated_summary_report()
        
        with open(output_dir / "phase5_integrated_report.txt", 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✓ 最終統合レポートを {output_dir} に保存しました")
        print(f"  - 詳細結果: phase5_integrated_results.json")
        print(f"  - 統合レポート: phase5_integrated_report.txt")
        
        return report_content
    
    def create_integrated_summary_report(self):
        """統合サマリーレポートの作成"""
        report = []
        report.append("="*70)
        report.append("小学校出前授業アンケート Phase 5 統合分析最終レポート")
        report.append("="*70)
        report.append(f"統合分析実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 研究概要
        report.append("【研究概要】")
        report.append("本研究は、小学校における「ものの溶け方」をテーマとした出前授業の")
        report.append("教育効果を定量的に評価することを目的として実施された。")
        report.append("炎色反応と再結晶実験を用いた体験型授業の前後で、")
        report.append("児童の理解度・興味の変化を多角的に分析した。")
        report.append("")
        
        # メタ分析結果
        if 'meta_analysis' in self.integrated_results:
            meta_data = self.integrated_results['meta_analysis']
            summary = meta_data.get('overall_summary', {})
            
            report.append("【メタ分析結果サマリー】")
            report.append(f"総効果量数: {summary.get('total_effect_sizes', 0)}")
            report.append(f"有意な効果: {summary.get('significant_effects', 0)}")
            report.append(f"正の効果: {summary.get('positive_effects', 0)}")
            report.append(f"負の効果: {summary.get('negative_effects', 0)}")
            
            if summary.get('largest_effect'):
                largest = summary['largest_effect']
                report.append(f"最大効果: {largest['source']} (効果量: {largest['effect_size']:.3f})")
            
            report.append("")
        
        # 予測モデル結果
        if 'prediction_model' in self.integrated_results:
            model_data = self.integrated_results['prediction_model']
            if 'model' in model_data and 'error' not in model_data['model']:
                model_result = model_data['model']
                
                report.append("【統合予測モデル結果】")
                report.append(f"モデルタイプ: {model_result['model_type']}")
                report.append(f"交差検証精度: {model_result['cv_mean']:.3f} ± {model_result['cv_std']:.3f}")
                report.append(f"学習データ精度: {model_result['train_accuracy']:.3f}")
                
                # 最重要特徴量
                if model_result['feature_importance']:
                    top_feature = model_result['feature_importance'][0]
                    report.append(f"最重要特徴量: {top_feature[0]} (重要度: {top_feature[1]:.3f})")
                
                report.append("")
        
        # 教育的知見
        if 'educational_insights' in self.integrated_results:
            insights = self.integrated_results['educational_insights']
            
            report.append("【教育効果の総合評価】")
            effectiveness = insights['effectiveness_summary']
            for key, value in effectiveness.items():
                key_jp = key.replace('statistical_significance', '統計的有意性')\
                           .replace('effect_size_magnitude', '効果量の大きさ')\
                           .replace('practical_significance', '実践的意義')\
                           .replace('student_satisfaction', '学生満足度')\
                           .replace('learning_evidence', '学習証拠')
                report.append(f"  {key_jp}: {value}")
            
            report.append("")
            
            # 成功要素
            if insights['successful_elements']:
                report.append("【特に成功した要素】")
                for element in insights['successful_elements']:
                    report.append(f"  ✓ {element}")
                report.append("")
            
            # 改善領域
            if insights['improvement_areas']:
                report.append("【改善が必要な領域】")
                for area in insights['improvement_areas']:
                    report.append(f"  ⚠️  {area}")
                report.append("")
        
        # Phase別主要発見事項
        report.append("【Phase別主要発見事項】")
        
        # Phase 1
        report.append("\nPhase 1 (データ品質・基礎統計):")
        report.append("  • 99名のデータで100%のマッチング率を達成")
        report.append("  • 高品質なExcel準拠データでの分析を実現")
        
        # Phase 2  
        report.append("\nPhase 2 (統計的検証):")
        report.append("  • 26ペアでの統計的検定を実施")
        report.append("  • 統計的有意な変化は検出されず")
        report.append("  • サンプルサイズ不足により検出力が限定的")
        
        # Phase 3
        report.append("\nPhase 3 (集団間差異):")
        report.append("  • 実験への興味、新しい学び、理解度でクラス間差異を確認")
        report.append("  • 理解度予測精度96.2%の高精度モデル構築")
        report.append("  • 実験への興味が最重要予測因子")
        
        # Phase 4
        report.append("\nPhase 4 (テキストマイニング):")
        report.append("  • Q2回答で語彙の簡素化と科学用語習得を確認")
        report.append("  • 感想文で63.6%のポジティブ率")
        report.append("  • 「ナトリウム」など科学用語の適切な使用")
        
        report.append("")
        
        # 実践的提言
        if 'educational_insights' in self.integrated_results:
            recommendations = insights.get('recommendations', [])
            if recommendations:
                report.append("【教育実践への提言】")
                for i, rec in enumerate(recommendations[:5], 1):
                    report.append(f"{i}. {rec}")
                report.append("")
        
        # 研究の限界
        report.append("【研究の限界】")
        report.append("1. サンプルサイズの制約（26ペア）により統計的検出力が限定的")
        report.append("2. クラス分布の偏り（1クラスに集中）により一般化可能性に制約")
        report.append("3. 単一時点での測定のため、学習の定着度は未評価")
        report.append("4. 比較対照群がないため、教育効果の帰属に制約")
        report.append("")
        
        # 今後の研究方向
        report.append("【今後の研究方向】")
        report.append("1. より大規模なサンプル（50名以上）での追試")
        report.append("2. 遅延テストによる学習定着度の評価")
        report.append("3. 比較対照群を設定したランダム化比較試験")
        report.append("4. 長期的フォローアップによる持続的効果の検証")
        report.append("5. 他の理科単元への拡張可能性の検討")
        report.append("")
        
        # 結論
        report.append("【結論】")
        report.append("本出前授業は統計的に有意な学習効果は検出されなかったものの、")
        report.append("児童の実験への高い興味・関心（平均3.08/4.0）と")
        report.append("感想文における強いポジティブ反応（63.6%）から、")
        report.append("科学への興味・関心の向上という観点では一定の教育効果が認められる。")
        report.append("")
        report.append("特に炎色反応実験の視覚的インパクトと、")
        report.append("科学用語「ナトリウム」の習得・使用が確認されたことは、")
        report.append("体験型理科教育の有効性を示唆している。")
        report.append("")
        report.append("今後はサンプルサイズの拡大と測定方法の改善により、")
        report.append("より確実な教育効果の検証が期待される。")
        
        return "\n".join(report)
    
    def run_complete_analysis(self):
        """Phase 5 完全統合分析実行"""
        print("小学校出前授業アンケート Phase 5: 統合的分析と予測モデル")
        print("実行開始:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("="*70)
        
        try:
            # 全Phase結果の読み込み
            self.load_all_phase_results()
            
            # メタ分析的効果量統合
            self.meta_analysis_effect_sizes()
            
            # 統合予測モデル構築
            self.build_integrated_prediction_model()
            
            # 教育的知見の統合
            self.synthesize_educational_insights()
            
            # 可視化作成
            self.create_visualizations()
            
            # 最終レポート生成
            final_report = self.generate_final_report()
            
            print("\n" + "="*70)
            print("Phase 5 統合分析完了!")
            print("="*70)
            print(final_report)
            
            return self.integrated_results
            
        except Exception as e:
            print(f"❌ Phase 5 統合分析中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """メイン実行関数"""
    analyzer = Phase5IntegratedAnalyzer()
    results = analyzer.run_complete_analysis()
    return results

if __name__ == "__main__":
    main()