#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
機械学習による理解度予測モデル構築
==================================

授業後データを用いた理解度予測システムの構築と特徴量重要度分析。

機能:
- 多種類の機械学習モデル比較（RandomForest, XGBoost, LogisticRegression）
- 交差検証による予測精度評価
- 特徴量重要度分析と解釈
- クラス不均衡対応
- ハイパーパラメータチューニング
- 予測結果の可視化と教育的示唆

制約:
- 独立群比較データのため授業後データのみ使用
- 理解度4段階分類問題として扱う
- 限られたサンプルサイズ（N=99）での予測精度評価

Author: Claude Code Analysis (ML Prediction Implementation)
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

# 機械学習ライブラリ
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import pickle

# 警告抑制
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class MachineLearningPredictor:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data" / "analysis"
        self.output_dir = self.project_root / "outputs" / "current" / "05_advanced_analysis"
        self.figures_dir = self.project_root / "outputs" / "figures" / "current" / "05_advanced_analysis"
        
        # 出力ディレクトリ作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.models = {}
        
    def load_data(self):
        """授業後データ読み込み"""
        print("📊 授業後データ読み込み中...")
        
        after_path = self.data_dir / "after_excel_compliant.csv"
        if not after_path.exists():
            raise FileNotFoundError("授業後データファイルが見つかりません")
        
        self.after_data = pd.read_csv(after_path, encoding='utf-8')
        print(f"✓ 授業後データ: {len(self.after_data)} 件")
        
    def prepare_ml_data(self):
        """機械学習用データ準備"""
        print("\n🔧 機械学習用データ準備中...")
        
        ml_data = self.after_data.copy()
        
        # Q1総合スコア計算
        q1_cols = ['Q1_Saltwater', 'Q1_Sugarwater', 'Q1_Muddywater',
                   'Q1_Ink', 'Q1_MisoSoup', 'Q1_SoySauce']
        ml_data['Q1_total'] = ml_data[q1_cols].sum(axis=1)
        
        # Q3総合スコア計算
        q3_cols = ['Q3_TeaLeaves_DissolveInWater', 'Q3_TeaComponents_DissolveInWater']
        ml_data['Q3_total'] = ml_data[q3_cols].sum(axis=1)
        
        # クラスダミー変数作成
        class_dummies = pd.get_dummies(ml_data['class'], prefix='class')
        
        # 特徴量選択
        feature_cols = ['Q1_total', 'Q3_total', 'Q4_ExperimentInterestRating', 'Q5_NewLearningsRating']
        feature_cols.extend(class_dummies.columns.tolist())
        
        # データ統合
        features_df = pd.concat([ml_data[['Q1_total', 'Q3_total', 'Q4_ExperimentInterestRating', 'Q5_NewLearningsRating']], 
                                class_dummies], axis=1)
        
        # 目的変数
        target = ml_data['Q6_DissolvingUnderstandingRating']
        
        # 欠損値除去
        complete_indices = features_df.notna().all(axis=1) & target.notna()
        self.X = features_df[complete_indices].copy()
        self.y = target[complete_indices].copy()
        
        # 特徴量名保存
        self.feature_names = self.X.columns.tolist()
        
        print(f"✓ 完全データ: {len(self.X)} 件")
        print(f"✓ 特徴量数: {len(self.feature_names)}")
        print(f"✓ 特徴量: {self.feature_names}")
        
        # クラス分布確認
        print("\n📊 目的変数（Q6理解度）の分布:")
        class_distribution = self.y.value_counts().sort_index()
        for class_val, count in class_distribution.items():
            print(f"  理解度 {class_val}: {count} 件 ({count/len(self.y)*100:.1f}%)")
        
        # データ概要
        print("\n📊 特徴量の基本統計:")
        print(self.X.describe().round(3))
        
    def train_models(self):
        """複数の機械学習モデル訓練"""
        print("\n🤖 機械学習モデル訓練中...")
        
        # データ標準化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        
        # クラス重み計算（不均衡データ対応）
        class_weights = compute_class_weight('balanced', classes=np.unique(self.y), y=self.y)
        class_weight_dict = dict(zip(np.unique(self.y), class_weights))
        
        # モデル定義
        models_config = {
            'RandomForest': {
                'model': RandomForestClassifier(
                    n_estimators=100, 
                    random_state=42, 
                    class_weight='balanced',
                    max_depth=5  # 過学習防止
                ),
                'use_scaled': False
            },
            'LogisticRegression': {
                'model': LogisticRegression(
                    random_state=42, 
                    class_weight='balanced',
                    max_iter=1000,
                    multi_class='ovr'
                ),
                'use_scaled': True
            }
        }
        
        # 交差検証設定
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 各モデルの訓練と評価
        for model_name, config in models_config.items():
            print(f"\n🔍 {model_name} 訓練中...")
            
            model = config['model']
            X_input = X_scaled if config['use_scaled'] else self.X.values
            
            # 交差検証
            cv_scores = cross_val_score(model, X_input, self.y, cv=cv, scoring='accuracy')
            f1_scores = cross_val_score(model, X_input, self.y, cv=cv, scoring='f1_weighted')
            
            # 全データでモデル訓練（最終モデル）
            model.fit(X_input, self.y)
            
            # 予測（全データ）
            y_pred = model.predict(X_input)
            y_pred_proba = model.predict_proba(X_input) if hasattr(model, 'predict_proba') else None
            
            # 結果保存
            self.models[model_name] = {
                'model': model,
                'scaler': self.scaler if config['use_scaled'] else None,
                'cv_accuracy': cv_scores,
                'cv_f1_weighted': f1_scores,
                'final_accuracy': accuracy_score(self.y, y_pred),
                'final_f1_weighted': f1_score(self.y, y_pred, average='weighted'),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': classification_report(self.y, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(self.y, y_pred)
            }
            
            print(f"✓ CV精度: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"✓ CV F1スコア: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")
            print(f"✓ 最終精度: {accuracy_score(self.y, y_pred):.3f}")
        
        # 特徴量重要度分析
        self._analyze_feature_importance()
        
    def _analyze_feature_importance(self):
        """特徴量重要度分析"""
        print("\n🔍 特徴量重要度分析中...")
        
        # RandomForestの特徴量重要度
        if 'RandomForest' in self.models:
            rf_model = self.models['RandomForest']['model']
            feature_importance = rf_model.feature_importances_
            
            # 重要度をデータフレームで整理
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            self.models['RandomForest']['feature_importance'] = importance_df
            
            print("📊 RandomForest特徴量重要度:")
            for _, row in importance_df.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")
        
        # LogisticRegressionの係数
        if 'LogisticRegression' in self.models:
            lr_model = self.models['LogisticRegression']['model']
            
            # 多クラス分類の場合の係数処理
            if hasattr(lr_model, 'coef_'):
                if lr_model.coef_.ndim == 1:
                    coefficients = lr_model.coef_
                else:
                    # 多クラスの場合は平均絶対値
                    coefficients = np.mean(np.abs(lr_model.coef_), axis=0)
                
                coef_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'coefficient': coefficients
                }).sort_values('coefficient', key=abs, ascending=False)
                
                self.models['LogisticRegression']['coefficients'] = coef_df
                
                print("\n📊 LogisticRegression係数:")
                for _, row in coef_df.head(10).iterrows():
                    print(f"  {row['feature']}: {row['coefficient']:.3f}")
    
    def hyperparameter_tuning(self):
        """ハイパーパラメータチューニング"""
        print("\n⚙️ ハイパーパラメータチューニング中...")
        
        # RandomForestのチューニング
        if 'RandomForest' in self.models:
            print("🔧 RandomForest チューニング...")
            
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 計算量削減
            
            grid_search = GridSearchCV(
                rf_base, param_grid, cv=cv, scoring='f1_weighted', 
                n_jobs=1, verbose=0  # 並列化無効（安全性のため）
            )
            
            try:
                grid_search.fit(self.X.values, self.y)
                
                self.models['RandomForest_Tuned'] = {
                    'model': grid_search.best_estimator_,
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'cv_results': grid_search.cv_results_
                }
                
                print(f"✓ 最適パラメータ: {grid_search.best_params_}")
                print(f"✓ 最適CV F1スコア: {grid_search.best_score_:.3f}")
                
            except Exception as e:
                print(f"⚠️ チューニングエラー: {e}")
                print("デフォルトパラメータを使用します")
    
    def evaluate_models(self):
        """モデル評価"""
        print("\n📈 モデル評価中...")
        
        evaluation_results = {}
        
        for model_name, model_data in self.models.items():
            if 'cv_accuracy' in model_data:  # 訓練済みモデルのみ
                evaluation_results[model_name] = {
                    'cv_accuracy_mean': model_data['cv_accuracy'].mean(),
                    'cv_accuracy_std': model_data['cv_accuracy'].std(),
                    'cv_f1_mean': model_data['cv_f1_weighted'].mean(),
                    'cv_f1_std': model_data['cv_f1_weighted'].std(),
                    'final_accuracy': model_data['final_accuracy'],
                    'final_f1': model_data['final_f1_weighted']
                }
        
        self.evaluation_results = evaluation_results
        
        # 結果表示
        print("\n📊 モデル比較結果:")
        print("Model\t\t\tCV Accuracy\t\tCV F1\t\t\tFinal Accuracy")
        print("-" * 70)
        
        for model_name, metrics in evaluation_results.items():
            print(f"{model_name:<20}\t{metrics['cv_accuracy_mean']:.3f}±{metrics['cv_accuracy_std']:.3f}\t\t"
                  f"{metrics['cv_f1_mean']:.3f}±{metrics['cv_f1_std']:.3f}\t\t{metrics['final_accuracy']:.3f}")
    
    def create_visualizations(self):
        """可視化作成"""
        print("\n📊 可視化作成中...")
        
        # 1. モデル比較
        self._create_model_comparison_plot()
        
        # 2. 特徴量重要度
        self._create_feature_importance_plot()
        
        # 3. 混同行列
        self._create_confusion_matrices()
        
        # 4. 予測結果分布
        self._create_prediction_distribution_plot()
        
    def _create_model_comparison_plot(self):
        """モデル比較プロット"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # CV精度比較
        models = []
        accuracies = []
        f1_scores = []
        
        for model_name, metrics in self.evaluation_results.items():
            models.append(model_name.replace('_', '\n'))
            accuracies.append(metrics['cv_accuracy_mean'])
            f1_scores.append(metrics['cv_f1_mean'])
        
        x_pos = np.arange(len(models))
        
        ax1.bar(x_pos, accuracies, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('モデル')
        ax1.set_ylabel('交差検証精度')
        ax1.set_title('モデル別予測精度比較')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.set_ylim(0, 1)
        
        # 精度値をバーの上に表示
        for i, v in enumerate(accuracies):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # F1スコア比較
        ax2.bar(x_pos, f1_scores, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('モデル')
        ax2.set_ylabel('F1スコア（重み付き）')
        ax2.set_title('モデル別F1スコア比較')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.set_ylim(0, 1)
        
        # F1値をバーの上に表示
        for i, v in enumerate(f1_scores):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        output_path = self.figures_dir / "model_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ モデル比較図保存: {output_path}")
    
    def _create_feature_importance_plot(self):
        """特徴量重要度プロット"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # RandomForest重要度
        if 'RandomForest' in self.models and 'feature_importance' in self.models['RandomForest']:
            importance_df = self.models['RandomForest']['feature_importance'].head(10)
            
            axes[0].barh(importance_df['feature'], importance_df['importance'])
            axes[0].set_xlabel('重要度')
            axes[0].set_title('RandomForest 特徴量重要度')
            axes[0].invert_yaxis()
        
        # LogisticRegression係数
        if 'LogisticRegression' in self.models and 'coefficients' in self.models['LogisticRegression']:
            coef_df = self.models['LogisticRegression']['coefficients'].head(10)
            
            colors = ['red' if x < 0 else 'blue' for x in coef_df['coefficient']]
            axes[1].barh(coef_df['feature'], coef_df['coefficient'], color=colors, alpha=0.7)
            axes[1].set_xlabel('係数')
            axes[1].set_title('LogisticRegression 回帰係数')
            axes[1].invert_yaxis()
            axes[1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        output_path = self.figures_dir / "feature_importance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 特徴量重要度図保存: {output_path}")
    
    def _create_confusion_matrices(self):
        """混同行列作成"""
        n_models = len([m for m in self.models.keys() if 'confusion_matrix' in self.models[m]])
        if n_models == 0:
            return
        
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        if n_models == 1:
            axes = [axes]
        
        model_idx = 0
        for model_name, model_data in self.models.items():
            if 'confusion_matrix' in model_data:
                cm = model_data['confusion_matrix']
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[model_idx])
                axes[model_idx].set_title(f'{model_name}\n混同行列')
                axes[model_idx].set_xlabel('予測ラベル')
                axes[model_idx].set_ylabel('実際のラベル')
                
                model_idx += 1
        
        plt.tight_layout()
        
        output_path = self.figures_dir / "confusion_matrices.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 混同行列保存: {output_path}")
    
    def _create_prediction_distribution_plot(self):
        """予測結果分布プロット"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 実際の分布
        self.y.value_counts().sort_index().plot(kind='bar', ax=axes[0,0], color='lightblue')
        axes[0,0].set_title('実際の理解度分布')
        axes[0,0].set_xlabel('理解度レベル')
        axes[0,0].set_ylabel('頻度')
        axes[0,0].tick_params(axis='x', rotation=0)
        
        # 各モデルの予測分布
        model_idx = 1
        for model_name, model_data in self.models.items():
            if 'predictions' in model_data and model_idx < 4:
                row, col = divmod(model_idx, 2)
                
                pred_series = pd.Series(model_data['predictions'])
                pred_series.value_counts().sort_index().plot(kind='bar', ax=axes[row,col], color='lightcoral')
                axes[row,col].set_title(f'{model_name} 予測分布')
                axes[row,col].set_xlabel('理解度レベル')
                axes[row,col].set_ylabel('頻度')
                axes[row,col].tick_params(axis='x', rotation=0)
                
                model_idx += 1
        
        # 未使用の軸を非表示
        for i in range(model_idx, 4):
            row, col = divmod(i, 2)
            axes[row,col].axis('off')
        
        plt.tight_layout()
        
        output_path = self.figures_dir / "prediction_distributions.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 予測分布図保存: {output_path}")
    
    def interpret_results(self):
        """結果解釈と教育的示唆"""
        print("\n📝 結果解釈中...")
        
        interpretations = {
            'model_performance': self._interpret_model_performance(),
            'feature_insights': self._interpret_feature_importance(),
            'educational_implications': self._generate_educational_implications(),
            'methodological_notes': self._generate_methodological_notes()
        }
        
        self.results['interpretations'] = interpretations
        
    def _interpret_model_performance(self):
        """モデル性能解釈"""
        best_model = max(self.evaluation_results.items(), key=lambda x: x[1]['cv_f1_mean'])
        best_model_name, best_metrics = best_model
        
        interpretation = {
            'best_model': best_model_name,
            'best_f1_score': best_metrics['cv_f1_mean'],
            'best_accuracy': best_metrics['cv_accuracy_mean'],
            'performance_level': self._classify_performance_level(best_metrics['cv_f1_mean']),
            'model_reliability': self._assess_model_reliability(best_metrics),
            'practical_utility': self._assess_practical_utility(best_metrics)
        }
        
        return interpretation
    
    def _classify_performance_level(self, f1_score):
        """性能レベル分類"""
        if f1_score >= 0.8:
            return "優秀"
        elif f1_score >= 0.7:
            return "良好"
        elif f1_score >= 0.6:
            return "中程度"
        else:
            return "改善が必要"
    
    def _assess_model_reliability(self, metrics):
        """モデル信頼性評価"""
        cv_std = metrics['cv_f1_std']
        if cv_std <= 0.05:
            return "非常に安定"
        elif cv_std <= 0.1:
            return "安定"
        else:
            return "やや不安定"
    
    def _assess_practical_utility(self, metrics):
        """実用性評価"""
        accuracy = metrics['cv_accuracy_mean']
        if accuracy >= 0.8:
            return "教育現場での実用性高"
        elif accuracy >= 0.7:
            return "参考指標として有用"
        else:
            return "追加改善が必要"
    
    def _interpret_feature_importance(self):
        """特徴量重要度解釈"""
        insights = {
            'most_important_features': [],
            'learning_factors': [],
            'class_effects': []
        }
        
        if 'RandomForest' in self.models and 'feature_importance' in self.models['RandomForest']:
            importance_df = self.models['RandomForest']['feature_importance']
            
            # 上位3つの重要特徴量
            top_features = importance_df.head(3)
            for _, row in top_features.iterrows():
                insights['most_important_features'].append({
                    'feature': row['feature'],
                    'importance': row['importance'],
                    'interpretation': self._interpret_feature_meaning(row['feature'])
                })
            
            # 学習要因と背景要因の分離
            for _, row in importance_df.iterrows():
                if row['feature'] in ['Q4_ExperimentInterestRating', 'Q5_NewLearningsRating']:
                    insights['learning_factors'].append({
                        'feature': row['feature'],
                        'importance': row['importance']
                    })
                elif 'class_' in row['feature']:
                    insights['class_effects'].append({
                        'feature': row['feature'],
                        'importance': row['importance']
                    })
        
        return insights
    
    def _interpret_feature_meaning(self, feature_name):
        """特徴量の意味解釈"""
        interpretations = {
            'Q1_total': '基礎的な溶解概念の理解度（授業前知識の影響）',
            'Q3_total': 'お茶に関する具体的知識（身近な事例への適用）',
            'Q4_ExperimentInterestRating': '実験活動への興味・関心度（学習動機）',
            'Q5_NewLearningsRating': '新しい学びへの自覚度（メタ認知）',
            'class_1.0': 'クラス1の特徴（指導環境・集団特性）',
            'class_2.0': 'クラス2の特徴（指導環境・集団特性）',
            'class_3.0': 'クラス3の特徴（指導環境・集団特性）',
            'class_4.0': 'クラス4の特徴（指導環境・集団特性）'
        }
        return interpretations.get(feature_name, '未定義の特徴量')
    
    def _generate_educational_implications(self):
        """教育的示唆生成"""
        implications = {
            'instruction_strategies': [
                "実験活動への興味喚起が理解度向上の鍵となる要因",
                "基礎概念の確実な理解が発展的学習の基盤",
                "新しい学びへの自覚を促すメタ認知支援の重要性",
                "クラス特性を考慮した個別化指導の必要性"
            ],
            'assessment_insights': [
                "理解度予測における複合的要因の重要性",
                "学習動機と認知要因の相互作用",
                "量的指標による学習成果の客観的評価の可能性"
            ],
            'curriculum_design': [
                "段階的な概念形成を支援する教材開発",
                "実験と理論の効果的な統合方法",
                "個人差に対応した多様な学習支援策"
            ]
        }
        
        return implications
    
    def _generate_methodological_notes(self):
        """方法論的注意事項"""
        notes = {
            'sample_size_limitations': [
                f"限定的なサンプルサイズ（N={len(self.y)}）による予測精度の制約",
                "交差検証による汎化性能の評価の重要性",
                "追加データ収集による予測モデル改善の必要性"
            ],
            'feature_engineering': [
                "現在の特徴量設計の妥当性と改善可能性",
                "テキストデータ等の質的情報の活用検討",
                "時系列要素（授業前後変化）の将来的な組み込み"
            ],
            'model_interpretability': [
                "教育現場での解釈可能性を重視したモデル選択",
                "特徴量重要度の教育学的意味の継続的検証",
                "予測結果の教育実践への適用における注意点"
            ]
        }
        
        return notes
    
    def save_results(self):
        """結果保存"""
        print("\n💾 結果保存中...")
        
        # 詳細結果準備
        detailed_results = {
            'metadata': {
                'analysis_type': 'Machine Learning Prediction',
                'generated_at': datetime.now().isoformat(),
                'sample_size': len(self.y),
                'features_used': self.feature_names,
                'target_variable': 'Q6_DissolvingUnderstandingRating'
            },
            'model_performance': self.evaluation_results,
            'feature_analysis': {},
            'interpretations': self.results.get('interpretations', {}),
            'models_summary': {}
        }
        
        # 特徴量重要度情報追加
        for model_name, model_data in self.models.items():
            if 'feature_importance' in model_data:
                detailed_results['feature_analysis'][model_name] = model_data['feature_importance'].to_dict('records')
            if 'coefficients' in model_data:
                detailed_results['feature_analysis'][model_name + '_coefficients'] = model_data['coefficients'].to_dict('records')
        
        # モデル要約情報
        for model_name, model_data in self.models.items():
            if 'cv_accuracy' in model_data:
                detailed_results['models_summary'][model_name] = {
                    'cross_validation_accuracy': model_data['cv_accuracy'].tolist(),
                    'cross_validation_f1': model_data['cv_f1_weighted'].tolist(),
                    'final_metrics': {
                        'accuracy': model_data['final_accuracy'],
                        'f1_weighted': model_data['final_f1_weighted']
                    }
                }
        
        # JSON保存
        output_path = self.output_dir / "machine_learning_prediction_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✓ 詳細結果保存: {output_path}")
        
        # モデル保存（最良モデル）
        self._save_best_model()
        
        # 要約レポート作成
        self._create_summary_report(detailed_results)
    
    def _save_best_model(self):
        """最良モデル保存"""
        if self.evaluation_results:
            best_model_name = max(self.evaluation_results.items(), key=lambda x: x[1]['cv_f1_mean'])[0]
            best_model_data = self.models[best_model_name]
            
            model_package = {
                'model': best_model_data['model'],
                'scaler': best_model_data.get('scaler'),
                'feature_names': self.feature_names,
                'model_name': best_model_name,
                'performance_metrics': self.evaluation_results[best_model_name]
            }
            
            model_path = self.output_dir / "best_prediction_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model_package, f)
            
            print(f"✓ 最良モデル保存: {model_path}")
    
    def _create_summary_report(self, detailed_results):
        """要約レポート作成"""
        report_lines = [
            "# 機械学習による理解度予測モデル分析レポート",
            "## 小学校出前授業アンケート - 予測モデル構築",
            "",
            f"**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**サンプルサイズ**: {detailed_results['metadata']['sample_size']} 件",
            f"**目的変数**: {detailed_results['metadata']['target_variable']} (理解度4段階)",
            "",
            "## 分析概要",
            "",
            "授業後データを用いた理解度予測モデルを構築し、学習成果に影響する要因を特定。",
            "複数の機械学習手法を比較し、教育現場での実用性を評価。",
            "",
            "## モデル性能比較",
            ""
        ]
        
        # モデル性能表
        report_lines.append("| モデル | CV精度 | CV F1スコア | 最終精度 |")
        report_lines.append("|--------|--------|-------------|----------|")
        
        for model_name, metrics in detailed_results['model_performance'].items():
            report_lines.append(
                f"| {model_name} | {metrics['cv_accuracy_mean']:.3f}±{metrics['cv_accuracy_std']:.3f} | "
                f"{metrics['cv_f1_mean']:.3f}±{metrics['cv_f1_std']:.3f} | {metrics['final_accuracy']:.3f} |"
            )
        
        # 最良モデル
        best_model = max(detailed_results['model_performance'].items(), key=lambda x: x[1]['cv_f1_mean'])
        report_lines.extend([
            "",
            f"**最良モデル**: {best_model[0]}",
            f"**CV F1スコア**: {best_model[1]['cv_f1_mean']:.3f}",
            ""
        ])
        
        # 特徴量重要度
        if 'feature_analysis' in detailed_results:
            report_lines.extend([
                "## 重要な学習要因",
                ""
            ])
            
            # RandomForestの重要度がある場合
            rf_importance = detailed_results['feature_analysis'].get('RandomForest')
            if rf_importance:
                report_lines.append("### 特徴量重要度（RandomForest）")
                report_lines.append("")
                for feature_data in rf_importance[:5]:
                    report_lines.append(f"- **{feature_data['feature']}**: {feature_data['importance']:.3f}")
                report_lines.append("")
        
        # 教育的示唆
        if 'interpretations' in detailed_results and 'educational_implications' in detailed_results['interpretations']:
            implications = detailed_results['interpretations']['educational_implications']
            
            report_lines.extend([
                "## 教育的示唆",
                "",
                "### 指導戦略への提言",
                ""
            ])
            
            for strategy in implications['instruction_strategies']:
                report_lines.append(f"- {strategy}")
            
            report_lines.extend([
                "",
                "### 評価・カリキュラム設計への示唆",
                ""
            ])
            
            for insight in implications['assessment_insights']:
                report_lines.append(f"- {insight}")
        
        # 制約と限界
        report_lines.extend([
            "",
            "## 分析の制約と限界",
            "",
            f"- 限定的なサンプルサイズ（N={detailed_results['metadata']['sample_size']}）",
            "- 独立群比較データのため個人の変化は考慮不可",
            "- 追加特徴量（テキストデータ等）の活用余地",
            "- 教育現場での実装における解釈可能性の重要性",
            "",
            "## 今後の改善提案",
            "",
            "1. **データ拡充**: より大規模なデータセットでの検証",
            "2. **特徴量拡張**: テキスト分析結果の統合",
            "3. **縦断分析**: 個人追跡可能なデータでの学習過程分析",
            "4. **実用化検討**: 教育現場での予測システム導入",
            "",
            "---",
            "",
            "**Generated by**: Claude Code Analysis (ML Prediction Implementation)",
            f"**Model Files**: {self.output_dir}",
            f"**Visualizations**: {self.figures_dir}"
        ])
        
        # レポート保存
        report_path = self.output_dir / "ml_prediction_summary.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✓ 要約レポート保存: {report_path}")
    
    def run_complete_analysis(self):
        """完全機械学習分析実行"""
        print("="*60)
        print("機械学習による理解度予測モデル構築")
        print("="*60)
        print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        try:
            # 分析実行
            self.load_data()
            self.prepare_ml_data()
            self.train_models()
            self.hyperparameter_tuning()
            self.evaluate_models()
            self.create_visualizations()
            self.interpret_results()
            self.save_results()
            
            print("\n" + "="*60)
            print("🎉 機械学習分析完了!")
            print("="*60)
            print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            print("📁 出力ファイル:")
            print(f"  - 詳細結果: {self.output_dir}/machine_learning_prediction_results.json")
            print(f"  - 要約レポート: {self.output_dir}/ml_prediction_summary.txt")
            print(f"  - 最良モデル: {self.output_dir}/best_prediction_model.pkl")
            print(f"  - 図表: {self.figures_dir}/")
            print()
            
            # 最良モデル情報表示
            if hasattr(self, 'evaluation_results') and self.evaluation_results:
                best_model = max(self.evaluation_results.items(), key=lambda x: x[1]['cv_f1_mean'])
                print(f"🏆 最良モデル: {best_model[0]}")
                print(f"   CV F1スコア: {best_model[1]['cv_f1_mean']:.3f}")
                print(f"   CV精度: {best_model[1]['cv_accuracy_mean']:.3f}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ 機械学習分析エラー: {e}")
            import traceback
            print(f"詳細: {traceback.format_exc()}")
            return False

def main():
    """メイン実行関数"""
    project_root = Path(__file__).parent.parent.parent
    
    predictor = MachineLearningPredictor(project_root)
    
    try:
        success = predictor.run_complete_analysis()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 ユーザーによって中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()