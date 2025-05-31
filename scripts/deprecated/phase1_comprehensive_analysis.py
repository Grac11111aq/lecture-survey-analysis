#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小学校出前授業アンケート Phase 1: データ品質確認と基礎統計
=======================================================

実施内容:
- データ品質チェック (欠損値、一貫性、外れ値)
- 記述統計量の算出
- 基礎的な可視化
- 前後アンケートのマッチング確認

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

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
warnings.filterwarnings('ignore')

class Phase1DataQualityAnalyzer:
    """Phase 1: データ品質分析クラス"""
    
    def __init__(self, data_dir="data/analysis"):
        self.data_dir = Path(data_dir)
        self.results = {}
        self.before_df = None
        self.after_df = None
        self.comment_df = None
        
    def load_data(self):
        """データの読み込み"""
        try:
            self.before_df = pd.read_csv(self.data_dir / "before_excel_compliant.csv")
            self.after_df = pd.read_csv(self.data_dir / "after_excel_compliant.csv")
            self.comment_df = pd.read_csv(self.data_dir / "comment.csv")
            
            print("✓ データ読み込み完了")
            print(f"  - 授業前: {len(self.before_df)} 行, {len(self.before_df.columns)} 列")
            print(f"  - 授業後: {len(self.after_df)} 行, {len(self.after_df.columns)} 列")
            print(f"  - 感想文: {len(self.comment_df)} 行, {len(self.comment_df.columns)} 列")
            
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            raise
    
    def check_data_quality(self):
        """データ品質チェック"""
        print("\n" + "="*50)
        print("データ品質チェック開始")
        print("="*50)
        
        quality_results = {}
        
        # 1. 欠損値パターンの確認
        print("\n1. 欠損値分析")
        print("-" * 30)
        
        for name, df in [("授業前", self.before_df), ("授業後", self.after_df), ("感想文", self.comment_df)]:
            missing_info = self.analyze_missing_values(df, name)
            quality_results[f"{name}_missing"] = missing_info
        
        # 2. Page_IDの重複・一意性確認
        print("\n2. Page_ID整合性チェック")
        print("-" * 30)
        
        page_id_analysis = self.check_page_id_consistency()
        quality_results["page_id_analysis"] = page_id_analysis
        
        # 3. 論理的一貫性チェック
        print("\n3. 論理的一貫性チェック")
        print("-" * 30)
        
        consistency_analysis = self.check_logical_consistency()
        quality_results["consistency_analysis"] = consistency_analysis
        
        # 4. 外れ値・異常値の検出
        print("\n4. 外れ値・異常値検出")
        print("-" * 30)
        
        outlier_analysis = self.detect_outliers()
        quality_results["outlier_analysis"] = outlier_analysis
        
        self.results["quality_check"] = quality_results
        return quality_results
    
    def analyze_missing_values(self, df, name):
        """欠損値分析"""
        missing_count = df.isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
        
        missing_info = {
            "total_rows": len(df),
            "missing_by_column": {},
            "high_missing_columns": [],
            "missing_pattern": {}
        }
        
        for col in df.columns:
            missing_info["missing_by_column"][col] = {
                "count": int(missing_count[col]),
                "percent": float(missing_percent[col])
            }
            
            if missing_percent[col] > 20:
                missing_info["high_missing_columns"].append(col)
        
        print(f"【{name}】")
        print(f"  総行数: {len(df)}")
        print(f"  欠損値あり列数: {(missing_count > 0).sum()}/{len(df.columns)}")
        
        if missing_info["high_missing_columns"]:
            print(f"  ⚠️  高欠損率列 (>20%): {missing_info['high_missing_columns']}")
        else:
            print("  ✓ 高欠損率列なし")
            
        # 欠損パターン分析
        missing_pattern = df.isnull().sum(axis=1).value_counts()
        missing_info["missing_pattern"] = missing_pattern.to_dict()
        
        return missing_info
    
    def check_page_id_consistency(self):
        """Page_ID整合性チェック"""
        before_ids = set(self.before_df['Page_ID'].dropna())
        after_ids = set(self.after_df['Page_ID'].dropna())
        
        # Page_ID分析
        analysis = {
            "before_unique_ids": len(before_ids),
            "after_unique_ids": len(after_ids),
            "matched_ids": len(before_ids & after_ids),
            "before_only": list(before_ids - after_ids),
            "after_only": list(after_ids - before_ids),
            "duplicate_check": {}
        }
        
        # 重複チェック
        for name, df in [("授業前", self.before_df), ("授業後", self.after_df)]:
            duplicates = df['Page_ID'].duplicated().sum()
            analysis["duplicate_check"][name] = int(duplicates)
        
        print(f"授業前 固有ID数: {analysis['before_unique_ids']}")
        print(f"授業後 固有ID数: {analysis['after_unique_ids']}")
        print(f"マッチした ID数: {analysis['matched_ids']}")
        
        if analysis["before_only"]:
            print(f"⚠️  授業前のみ: {len(analysis['before_only'])} 件")
        if analysis["after_only"]:
            print(f"⚠️  授業後のみ: {len(analysis['after_only'])} 件")
            
        if sum(analysis["duplicate_check"].values()) > 0:
            print(f"❌ Page_ID重複あり: {analysis['duplicate_check']}")
        else:
            print("✓ Page_ID重複なし")
        
        return analysis
    
    def check_logical_consistency(self):
        """論理的一貫性チェック"""
        consistency_issues = []
        
        # Q1の回答パターンチェック (授業前後)
        for name, df in [("授業前", self.before_df), ("授業後", self.after_df)]:
            q1_cols = [col for col in df.columns if col.startswith('Q1_')]
            
            if q1_cols:
                # すべてFalseの行をチェック
                all_false_count = (df[q1_cols] == False).all(axis=1).sum()
                all_true_count = (df[q1_cols] == True).all(axis=1).sum()
                
                consistency_issues.append({
                    "dataset": name,
                    "issue": "Q1_all_false",
                    "count": int(all_false_count),
                    "description": "Q1すべてFalse回答"
                })
                
                consistency_issues.append({
                    "dataset": name,
                    "issue": "Q1_all_true", 
                    "count": int(all_true_count),
                    "description": "Q1すべてTrue回答"
                })
        
        # 授業後の評価スコアチェック
        if 'Q4_ExperimentInterestRating' in self.after_df.columns:
            invalid_ratings = (~self.after_df['Q4_ExperimentInterestRating'].isin([1,2,3,4])).sum()
            consistency_issues.append({
                "dataset": "授業後",
                "issue": "invalid_Q4_rating",
                "count": int(invalid_ratings),
                "description": "Q4評価値異常 (1-4以外)"
            })
        
        print("論理的一貫性チェック結果:")
        for issue in consistency_issues:
            if issue["count"] > 0:
                print(f"  ⚠️  {issue['dataset']}: {issue['description']} - {issue['count']}件")
            else:
                print(f"  ✓ {issue['dataset']}: {issue['description']} - 問題なし")
        
        return consistency_issues
    
    def detect_outliers(self):
        """外れ値・異常値検出"""
        outlier_results = {}
        
        # Page_IDの範囲チェック
        for name, df in [("授業前", self.before_df), ("授業後", self.after_df)]:
            page_ids = df['Page_ID'].dropna()
            outlier_results[f"{name}_page_id"] = {
                "min": int(page_ids.min()),
                "max": int(page_ids.max()),
                "range": int(page_ids.max() - page_ids.min()),
                "gaps": self.find_id_gaps(page_ids)
            }
        
        # クラス番号の妥当性
        for name, df in [("授業前", self.before_df), ("授業後", self.after_df)]:
            classes = df['class'].dropna().unique()
            outlier_results[f"{name}_classes"] = {
                "unique_classes": sorted([int(c) for c in classes]),
                "class_counts": df['class'].value_counts().to_dict()
            }
        
        print("外れ値・異常値検出結果:")
        for key, result in outlier_results.items():
            if "page_id" in key:
                print(f"  {key}: ID範囲 {result['min']}-{result['max']}")
                if result["gaps"]:
                    print(f"    ⚠️  ID欠番: {result['gaps'][:5]}{'...' if len(result['gaps']) > 5 else ''}")
            elif "classes" in key:
                print(f"  {key}: クラス {result['unique_classes']}")
        
        return outlier_results
    
    def find_id_gaps(self, page_ids):
        """Page_IDの欠番を検出"""
        sorted_ids = sorted(page_ids.dropna())
        gaps = []
        for i in range(1, len(sorted_ids)):
            if sorted_ids[i] - sorted_ids[i-1] > 1:
                for missing_id in range(sorted_ids[i-1] + 1, sorted_ids[i]):
                    gaps.append(missing_id)
        return gaps
    
    def calculate_basic_statistics(self):
        """基礎統計量の算出"""
        print("\n" + "="*50)
        print("基礎統計量の算出")
        print("="*50)
        
        stats_results = {}
        
        # Q1項目の正答率分析
        print("\n1. Q1: 水溶液認識の統計")
        print("-" * 30)
        
        q1_stats = self.analyze_q1_statistics()
        stats_results["q1_analysis"] = q1_stats
        
        # Q3項目の分析
        print("\n2. Q3: お茶の理解度統計")
        print("-" * 30)
        
        q3_stats = self.analyze_q3_statistics()
        stats_results["q3_analysis"] = q3_stats
        
        # 授業後の評価項目分析
        print("\n3. 授業後評価項目の統計")
        print("-" * 30)
        
        evaluation_stats = self.analyze_evaluation_statistics()
        stats_results["evaluation_analysis"] = evaluation_stats
        
        # クラス別の分布
        print("\n4. クラス別分布")
        print("-" * 30)
        
        class_stats = self.analyze_class_distribution()
        stats_results["class_analysis"] = class_stats
        
        self.results["basic_statistics"] = stats_results
        return stats_results
    
    def analyze_q1_statistics(self):
        """Q1項目の統計分析"""
        q1_items = ['Saltwater', 'Sugarwater', 'Muddywater', 'Ink', 'MisoSoup', 'SoySauce']
        
        q1_analysis = {
            "before": {},
            "after": {},
            "comparison": {}
        }
        
        # 授業前の統計
        before_q1_cols = [f"Q1_{item}_Response" if f"Q1_{item}_Response" in self.before_df.columns 
                          else f"Q1_{item}" for item in q1_items]
        before_q1_cols = [col for col in before_q1_cols if col in self.before_df.columns]
        
        for col in before_q1_cols:
            item_name = col.replace("Q1_", "").replace("_Response", "")
            true_rate = self.before_df[col].mean() * 100
            q1_analysis["before"][item_name] = {
                "true_rate": float(true_rate),
                "count": int(self.before_df[col].sum()),
                "total": int(self.before_df[col].count())
            }
        
        # 授業後の統計
        after_q1_cols = [f"Q1_{item}" for item in q1_items if f"Q1_{item}" in self.after_df.columns]
        
        for col in after_q1_cols:
            item_name = col.replace("Q1_", "")
            true_rate = self.after_df[col].mean() * 100
            q1_analysis["after"][item_name] = {
                "true_rate": float(true_rate),
                "count": int(self.after_df[col].sum()),
                "total": int(self.after_df[col].count())
            }
        
        # 比較分析
        for item in q1_items:
            item_key = item
            if item_key in q1_analysis["before"] and item_key in q1_analysis["after"]:
                before_rate = q1_analysis["before"][item_key]["true_rate"]
                after_rate = q1_analysis["after"][item_key]["true_rate"]
                change = after_rate - before_rate
                
                q1_analysis["comparison"][item_key] = {
                    "before_rate": before_rate,
                    "after_rate": after_rate,
                    "change": float(change),
                    "change_direction": "increase" if change > 0 else "decrease" if change < 0 else "no_change"
                }
        
        # 結果表示
        print("Q1項目別正答率:")
        for item, comp in q1_analysis["comparison"].items():
            direction_symbol = "↗️" if comp["change"] > 5 else "↘️" if comp["change"] < -5 else "→"
            print(f"  {item:12}: {comp['before_rate']:5.1f}% → {comp['after_rate']:5.1f}% {direction_symbol} ({comp['change']:+5.1f}%)")
        
        return q1_analysis
    
    def analyze_q3_statistics(self):
        """Q3項目の統計分析"""
        q3_analysis = {
            "before": {},
            "after": {},
            "comparison": {}
        }
        
        q3_items = ['TeaLeavesDissolve', 'TeaComponentsDissolve']
        
        # 授業前分析
        for item in q3_items:
            col_name = f"Q3_{item}"
            if col_name in self.before_df.columns:
                true_rate = self.before_df[col_name].mean() * 100
                q3_analysis["before"][item] = {
                    "true_rate": float(true_rate),
                    "count": int(self.before_df[col_name].sum()),
                    "total": int(self.before_df[col_name].count())
                }
        
        # 授業後分析 (列名が異なる場合を考慮)
        q3_after_mapping = {
            'TeaLeavesDissolve': 'Q3_TeaLeaves_DissolveInWater',
            'TeaComponentsDissolve': 'Q3_TeaComponents_DissolveInWater'
        }
        
        for item, col_name in q3_after_mapping.items():
            if col_name in self.after_df.columns:
                true_rate = self.after_df[col_name].mean() * 100
                q3_analysis["after"][item] = {
                    "true_rate": float(true_rate),
                    "count": int(self.after_df[col_name].sum()),
                    "total": int(self.after_df[col_name].count())
                }
        
        # 比較分析
        for item in q3_items:
            if item in q3_analysis["before"] and item in q3_analysis["after"]:
                before_rate = q3_analysis["before"][item]["true_rate"]
                after_rate = q3_analysis["after"][item]["true_rate"]
                change = after_rate - before_rate
                
                q3_analysis["comparison"][item] = {
                    "before_rate": before_rate,
                    "after_rate": after_rate,
                    "change": float(change),
                    "change_direction": "increase" if change > 0 else "decrease" if change < 0 else "no_change"
                }
        
        # 結果表示
        print("Q3項目別正答率:")
        for item, comp in q3_analysis["comparison"].items():
            direction_symbol = "↗️" if comp["change"] > 5 else "↘️" if comp["change"] < -5 else "→"
            item_label = "茶葉溶解" if "Leaves" in item else "茶成分溶解"
            print(f"  {item_label:8}: {comp['before_rate']:5.1f}% → {comp['after_rate']:5.1f}% {direction_symbol} ({comp['change']:+5.1f}%)")
        
        return q3_analysis
    
    def analyze_evaluation_statistics(self):
        """授業後評価項目の統計分析"""
        evaluation_cols = [
            'Q4_ExperimentInterestRating',
            'Q5_NewLearningsRating', 
            'Q6_DissolvingUnderstandingRating'
        ]
        
        evaluation_stats = {}
        
        for col in evaluation_cols:
            if col in self.after_df.columns:
                data = self.after_df[col].dropna()
                
                evaluation_stats[col] = {
                    "count": int(len(data)),
                    "mean": float(data.mean()),
                    "std": float(data.std()),
                    "median": float(data.median()),
                    "mode": int(data.mode().iloc[0]) if len(data.mode()) > 0 else None,
                    "distribution": data.value_counts().sort_index().to_dict()
                }
        
        # 結果表示
        print("授業後評価項目統計:")
        label_mapping = {
            'Q4_ExperimentInterestRating': '実験への興味',
            'Q5_NewLearningsRating': '新しい学び',
            'Q6_DissolvingUnderstandingRating': '溶解理解度'
        }
        
        for col, stats in evaluation_stats.items():
            label = label_mapping.get(col, col)
            print(f"  {label:12}: 平均 {stats['mean']:.2f} (SD={stats['std']:.2f}), 中央値 {stats['median']:.1f}")
            dist_str = ", ".join([f"{k}:{v}件" for k, v in stats['distribution'].items()])
            print(f"    分布: {dist_str}")
        
        return evaluation_stats
    
    def analyze_class_distribution(self):
        """クラス別分布分析"""
        class_stats = {}
        
        for name, df in [("授業前", self.before_df), ("授業後", self.after_df)]:
            class_dist = df['class'].value_counts().sort_index()
            class_stats[name] = {
                "distribution": class_dist.to_dict(),
                "total": int(class_dist.sum()),
                "classes": sorted([int(c) for c in class_dist.index])
            }
        
        # 結果表示
        print("クラス別分布:")
        for name, stats in class_stats.items():
            print(f"  {name}: {stats['total']}名")
            for class_num, count in stats['distribution'].items():
                percentage = (count / stats['total']) * 100
                print(f"    {class_num}組: {count}名 ({percentage:.1f}%)")
        
        return class_stats
    
    def create_visualizations(self):
        """基礎的な可視化の作成"""
        print("\n" + "="*50)
        print("基礎的な可視化の作成")
        print("="*50)
        
        # 図の保存ディレクトリ作成
        output_dir = Path("outputs/phase1_figures")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 欠損値パターンの可視化
        self.plot_missing_patterns(output_dir)
        
        # 2. Q1項目の前後比較
        self.plot_q1_comparison(output_dir)
        
        # 3. クラス別分布
        self.plot_class_distribution(output_dir)
        
        # 4. 評価項目の分布
        self.plot_evaluation_distribution(output_dir)
        
        print(f"✓ 図表を {output_dir} に保存しました")
    
    def plot_missing_patterns(self, output_dir):
        """欠損値パターンの可視化"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (name, df) in enumerate([("Before", self.before_df), ("After", self.after_df), ("Comment", self.comment_df)]):
            missing_percent = (df.isnull().sum() / len(df)) * 100
            
            axes[i].bar(range(len(missing_percent)), missing_percent.values)
            axes[i].set_title(f'Missing Values - {name}')
            axes[i].set_xlabel('Columns')
            axes[i].set_ylabel('Missing %')
            axes[i].set_xticks(range(len(missing_percent)))
            axes[i].set_xticklabels(missing_percent.index, rotation=45, ha='right')
            
            # 20%ラインを追加
            axes[i].axhline(y=20, color='red', linestyle='--', alpha=0.7, label='20% threshold')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "missing_patterns.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_q1_comparison(self, output_dir):
        """Q1項目の前後比較"""
        if "basic_statistics" not in self.results or "q1_analysis" not in self.results["basic_statistics"]:
            return
            
        q1_data = self.results["basic_statistics"]["q1_analysis"]["comparison"]
        
        if not q1_data:
            return
        
        items = list(q1_data.keys())
        before_rates = [q1_data[item]["before_rate"] for item in items]
        after_rates = [q1_data[item]["after_rate"] for item in items]
        
        x = np.arange(len(items))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, before_rates, width, label='Before Class', alpha=0.8)
        bars2 = ax.bar(x + width/2, after_rates, width, label='After Class', alpha=0.8)
        
        ax.set_xlabel('Q1 Items')
        ax.set_ylabel('True Response Rate (%)')
        ax.set_title('Q1: Water Solution Recognition - Before vs After')
        ax.set_xticks(x)
        ax.set_xticklabels(items, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 値をバーの上に表示
        for i, (before, after) in enumerate(zip(before_rates, after_rates)):
            ax.text(i - width/2, before + 1, f'{before:.1f}%', ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, after + 1, f'{after:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / "q1_before_after_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_class_distribution(self, output_dir):
        """クラス別分布の可視化"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, (name, df) in enumerate([("Before", self.before_df), ("After", self.after_df)]):
            class_counts = df['class'].value_counts().sort_index()
            
            axes[i].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
            axes[i].set_title(f'Class Distribution - {name}')
        
        plt.tight_layout()
        plt.savefig(output_dir / "class_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_evaluation_distribution(self, output_dir):
        """評価項目分布の可視化"""
        evaluation_cols = [
            'Q4_ExperimentInterestRating',
            'Q5_NewLearningsRating', 
            'Q6_DissolvingUnderstandingRating'
        ]
        
        available_cols = [col for col in evaluation_cols if col in self.after_df.columns]
        
        if not available_cols:
            return
        
        fig, axes = plt.subplots(1, len(available_cols), figsize=(4*len(available_cols), 4))
        if len(available_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(available_cols):
            data = self.after_df[col].dropna()
            counts = data.value_counts().sort_index()
            
            axes[i].bar(counts.index, counts.values)
            axes[i].set_title(col.replace('_', ' '))
            axes[i].set_xlabel('Rating')
            axes[i].set_ylabel('Count')
            axes[i].set_xticks(range(1, 5))
            
            # 平均値を表示
            mean_val = data.mean()
            axes[i].axvline(x=mean_val, color='red', linestyle='--', alpha=0.7)
            axes[i].text(mean_val + 0.1, max(counts.values) * 0.8, f'Mean: {mean_val:.2f}', 
                        verticalalignment='center', color='red')
        
        plt.tight_layout()
        plt.savefig(output_dir / "evaluation_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self):
        """Phase 1 レポート生成"""
        print("\n" + "="*50)
        print("Phase 1 レポート生成")
        print("="*50)
        
        # レポート保存ディレクトリ作成
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # JSON形式で詳細結果を保存
        with open(output_dir / "phase1_detailed_results.json", 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # テキスト形式でサマリーレポートを生成
        report_content = self.create_summary_report()
        
        with open(output_dir / "phase1_summary_report.txt", 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✓ レポートを {output_dir} に保存しました")
        print(f"  - 詳細結果: phase1_detailed_results.json")
        print(f"  - サマリー: phase1_summary_report.txt")
        
        return report_content
    
    def create_summary_report(self):
        """サマリーレポートの作成"""
        report = []
        report.append("="*60)
        report.append("小学校出前授業アンケート Phase 1 分析結果サマリー")
        report.append("="*60)
        report.append(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # データ概要
        report.append("【データ概要】")
        report.append(f"授業前アンケート: {len(self.before_df)} 名")
        report.append(f"授業後アンケート: {len(self.after_df)} 名")
        report.append(f"感想文: {len(self.comment_df)} 件")
        report.append("")
        
        # データ品質評価
        if "quality_check" in self.results:
            report.append("【データ品質評価】")
            quality = self.results["quality_check"]
            
            # Page_ID整合性
            page_analysis = quality.get("page_id_analysis", {})
            matched_rate = (page_analysis.get("matched_ids", 0) / 
                          max(page_analysis.get("before_unique_ids", 1), 1)) * 100
            report.append(f"Page_IDマッチ率: {matched_rate:.1f}%")
            
            # 欠損値
            before_missing = quality.get("授業前_missing", {})
            after_missing = quality.get("授業後_missing", {})
            
            if before_missing.get("high_missing_columns"):
                report.append(f"⚠️  高欠損率列 (授業前): {before_missing['high_missing_columns']}")
            if after_missing.get("high_missing_columns"):
                report.append(f"⚠️  高欠損率列 (授業後): {after_missing['high_missing_columns']}")
            
            if not (before_missing.get("high_missing_columns") or after_missing.get("high_missing_columns")):
                report.append("✓ 高欠損率列なし")
            
            report.append("")
        
        # 主要発見事項
        if "basic_statistics" in self.results:
            report.append("【主要発見事項】")
            
            # Q1分析結果
            q1_analysis = self.results["basic_statistics"].get("q1_analysis", {})
            if "comparison" in q1_analysis:
                report.append("Q1: 水溶液認識の変化")
                for item, comp in q1_analysis["comparison"].items():
                    change = comp["change"]
                    direction = "改善" if change > 5 else "悪化" if change < -5 else "変化なし"
                    report.append(f"  {item}: {comp['before_rate']:.1f}% → {comp['after_rate']:.1f}% ({direction})")
                report.append("")
            
            # Q3分析結果
            q3_analysis = self.results["basic_statistics"].get("q3_analysis", {})
            if "comparison" in q3_analysis:
                report.append("Q3: お茶の理解度変化")
                for item, comp in q3_analysis["comparison"].items():
                    change = comp["change"]
                    direction = "改善" if change > 5 else "悪化" if change < -5 else "変化なし"
                    item_label = "茶葉溶解" if "Leaves" in item else "茶成分溶解"
                    report.append(f"  {item_label}: {comp['before_rate']:.1f}% → {comp['after_rate']:.1f}% ({direction})")
                report.append("")
            
            # 評価項目
            eval_analysis = self.results["basic_statistics"].get("evaluation_analysis", {})
            if eval_analysis:
                report.append("授業後評価項目")
                for col, stats in eval_analysis.items():
                    col_label = col.replace('Q4_ExperimentInterestRating', '実験への興味')\
                                  .replace('Q5_NewLearningsRating', '新しい学び')\
                                  .replace('Q6_DissolvingUnderstandingRating', '溶解理解度')
                    report.append(f"  {col_label}: 平均 {stats['mean']:.2f} (SD={stats['std']:.2f})")
                report.append("")
        
        # Phase 2への推奨事項
        report.append("【Phase 2への推奨事項】")
        
        # マッチ率に基づく推奨
        if "quality_check" in self.results:
            page_analysis = self.results["quality_check"].get("page_id_analysis", {})
            matched_ids = page_analysis.get("matched_ids", 0)
            
            if matched_ids >= 70:  # 十分なサンプルサイズ
                report.append("✓ 統計的検定の実施を推奨 (十分なサンプルサイズ)")
            elif matched_ids >= 30:
                report.append("⚠️  ノンパラメトリック検定の検討を推奨")
            else:
                report.append("❌ サンプルサイズ不足の可能性")
        
        # 効果量の予測
        if "basic_statistics" in self.results:
            q1_comparison = self.results["basic_statistics"].get("q1_analysis", {}).get("comparison", {})
            large_changes = [item for item, comp in q1_comparison.items() if abs(comp.get("change", 0)) > 10]
            
            if large_changes:
                report.append(f"✓ 大きな変化が期待される項目: {', '.join(large_changes)}")
            else:
                report.append("⚠️  変化が小さい可能性 - 効果量の慎重な評価が必要")
        
        report.append("")
        report.append("【次のステップ】")
        report.append("1. Phase 2: 統計的検定の実施")
        report.append("2. McNemar検定による前後比較")
        report.append("3. 効果量の算出")
        report.append("4. 多重比較補正の適用")
        
        return "\n".join(report)
    
    def run_complete_analysis(self):
        """Phase 1 完全分析実行"""
        print("小学校出前授業アンケート Phase 1: データ品質確認と基礎統計")
        print("実行開始:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("="*60)
        
        try:
            # データ読み込み
            self.load_data()
            
            # データ品質チェック
            self.check_data_quality()
            
            # 基礎統計量算出
            self.calculate_basic_statistics()
            
            # 可視化作成
            self.create_visualizations()
            
            # レポート生成
            summary_report = self.generate_report()
            
            print("\n" + "="*60)
            print("Phase 1 分析完了!")
            print("="*60)
            print(summary_report)
            
            return self.results
            
        except Exception as e:
            print(f"❌ Phase 1 分析中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """メイン実行関数"""
    analyzer = Phase1DataQualityAnalyzer()
    results = analyzer.run_complete_analysis()
    return results

if __name__ == "__main__":
    main()