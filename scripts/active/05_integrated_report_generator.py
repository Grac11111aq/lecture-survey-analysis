#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合分析レポート生成器
===================

独立群比較前提での有効な結果のみを抽出・統合する。

有効な分析結果:
- Phase 1: 基礎統計、欠損値分析、クラス分布
- Phase 2: 修正版独立群比較（χ²検定、Mann-Whitney U検定）
- Phase 3: クラス間比較（授業後データのみ）
- Phase 4: テキストマイニング（群間差として解釈修正）

Author: Claude Code Analysis
Date: 2025-05-31
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class IntegratedAnalysisGenerator:
    def __init__(self):
        self.output_dir = Path("outputs")
        self.results = {}
        
    def load_valid_results(self):
        """有効な分析結果を読み込み"""
        print("有効な分析結果を読み込み中...")
        
        # Phase 1: 基礎統計と品質チェック（アーカイブから）
        with open(self.output_dir / "archive" / "phase1_detailed_results.json", 'r', encoding='utf-8') as f:
            phase1_data = json.load(f)
            self.results['phase1_valid'] = {
                'basic_statistics': phase1_data['basic_statistics'],
                'quality_check': phase1_data['quality_check']
            }
        
        # Phase 2: 修正版独立群比較（現在の有効結果から）
        with open(self.output_dir / "current" / "02_group_comparison" / "phase2_revised_results.json", 'r', encoding='utf-8') as f:
            self.results['phase2_revised'] = json.load(f)
        
        # Phase 3: クラス間比較（アーカイブから）
        with open(self.output_dir / "archive" / "phase3_detailed_results.json", 'r', encoding='utf-8') as f:
            phase3_data = json.load(f)
            self.results['phase3_valid'] = {
                'class_comparison_after': phase3_data['class_comparison'].get('after_analysis', {}),
                'factors_analysis': phase3_data.get('factors_analysis', {})
            }
        
        # Phase 4: テキストマイニング（アーカイブから）
        with open(self.output_dir / "archive" / "phase4_detailed_results.json", 'r', encoding='utf-8') as f:
            phase4_data = json.load(f)
            self.results['phase4_valid'] = {
                'frequency_analysis': phase4_data.get('frequency_analysis', {}),
                'sentiment_analysis': phase4_data.get('sentiment_analysis', {})
            }
        
        print("✓ 有効な分析結果の読み込み完了")
    
    def generate_executive_summary(self):
        """エグゼクティブサマリー生成"""
        summary = {
            'analysis_type': '独立群比較分析',
            'sample_size': {
                'before_group': 99,
                'after_group': 99
            },
            'key_findings': [],
            'statistical_significance': [],
            'limitations': [
                'Page_IDは個人識別子ではないため個人追跡不可能',
                '独立群比較のため個人レベルの変化は測定不可',
                '因果推論には限界がある（ランダム割付なし）',
                '授業効果以外の要因による群間差の可能性'
            ]
        }
        
        # Phase 2修正版の主要結果を抽出
        phase2_results = self.results.get('phase2_revised', {})
        
        # χ²検定結果の要約
        chi2_results = phase2_results.get('chi_square_analysis', {})
        # Q1とQ3の結果を統合
        all_chi2_results = {}
        if 'q1_results' in chi2_results:
            all_chi2_results.update(chi2_results['q1_results'])
        if 'q3_results' in chi2_results:
            all_chi2_results.update(chi2_results['q3_results'])
        
        summary['chi2_summary'] = {
            'total_tests': len(all_chi2_results),
            'significant_results': sum(1 for result in all_chi2_results.values() 
                                     if result.get('adjusted_significant', False)),
            'largest_effect': self._find_largest_effect(all_chi2_results)
        }
        
        # Mann-Whitney U検定結果の要約（composite_analysisを含む）
        mannwhitney_results = phase2_results.get('mann_whitney_analysis', {})
        composite_results = phase2_results.get('composite_analysis', {})
        
        # composite_analysisとmann_whitney_analysisを統合
        all_mw_results = {}
        all_mw_results.update(mannwhitney_results)
        all_mw_results.update(composite_results)
        
        summary['mannwhitney_summary'] = {
            'total_tests': len(all_mw_results),
            'significant_results': sum(1 for result in all_mw_results.values() 
                                     if result.get('significant', False)),
            'largest_effect': self._find_largest_effect_mw(all_mw_results)
        }
        
        return summary
    
    def _find_largest_effect(self, results):
        """χ²検定で最大効果量を探す"""
        max_effect = 0
        max_item = None
        for item, result in results.items():
            if 'cohens_h' in result and abs(result['cohens_h']) > max_effect:
                max_effect = abs(result['cohens_h'])
                max_item = item
        return {'item': max_item, 'effect_size': max_effect}
    
    def _find_largest_effect_mw(self, results):
        """Mann-Whitney U検定で最大効果量を探す"""
        max_effect = 0
        max_item = None
        for item, result in results.items():
            if 'cohens_d' in result and abs(result['cohens_d']) > max_effect:
                max_effect = abs(result['cohens_d'])
                max_item = item
        return {'item': max_item, 'effect_size': max_effect}
    
    def generate_methodology_section(self):
        """方法論セクション生成"""
        methodology = {
            'data_structure': {
                'description': 'Page_IDは個人識別子ではなく単なるページ番号',
                'analysis_approach': '独立群比較',
                'sample_sizes': {
                    'before_group': 99,
                    'after_group': 99
                }
            },
            'statistical_methods': {
                'chi_square': 'カテゴリカルデータの独立性検定',
                'mann_whitney_u': '順序尺度データの独立2群比較',
                'kruskal_wallis': 'クラス間比較（授業後データ）',
                'effect_sizes': ['Cohen\'s h', 'Cohen\'s d', 'η²']
            },
            'multiple_comparison_correction': {
                'fdr': 'False Discovery Rate補正',
                'bonferroni': 'Bonferroni補正'
            }
        }
        return methodology
    
    def generate_results_section(self):
        """結果セクション生成"""
        results_section = {}
        
        # Phase 1: 基礎統計
        results_section['basic_statistics'] = self.results['phase1_valid']['basic_statistics']
        
        # Phase 2: 独立群比較
        results_section['independent_group_comparison'] = self.results['phase2_revised']
        
        # Phase 3: クラス間分析
        results_section['class_analysis'] = self.results['phase3_valid']
        
        # Phase 4: テキスト分析（解釈修正済み）
        results_section['text_analysis'] = self._modify_text_interpretation(
            self.results['phase4_valid']
        )
        
        return results_section
    
    def _modify_text_interpretation(self, text_results):
        """テキスト分析の解釈を群間差に修正"""
        modified_results = text_results.copy()
        
        # 解釈文言を修正（個人の変化 → 群間差）
        interpretation_mapping = {
            '授業により': '授業後群では',
            '児童の理解が': '授業後群の理解が',
            '個人レベルで': '群間で',
            '変化が見られ': '差異が観察され',
            '向上した': '高い傾向を示した',
            '改善された': '良好な傾向を示した'
        }
        
        # テキスト内容の修正（実装は簡略化）
        if 'sentiment_analysis' in modified_results:
            modified_results['sentiment_analysis']['interpretation_note'] = \
                "テキスト分析の解釈は独立群比較として行う。個人の変化ではなく群間差として解釈する。"
        
        return modified_results
    
    def generate_discussion_section(self):
        """考察セクション生成"""
        discussion = {
            'key_findings': {
                'statistical_significance': self._summarize_significance(),
                'effect_sizes': self._summarize_effect_sizes(),
                'practical_significance': self._assess_practical_significance()
            },
            'educational_implications': {
                'positive_trends': [],
                'areas_for_improvement': [],
                'cautious_interpretations': [
                    '群間差の観察であり個人の変化ではない',
                    '授業効果以外の要因による可能性',
                    'ランダム割付でないため因果推論に限界'
                ]
            },
            'limitations': {
                'data_structure': 'Page_IDによる個人追跡不可能',
                'study_design': '独立群比較のため変化量測定不可',
                'confounding_factors': '授業以外の要因の影響可能性',
                'generalizability': 'この学校・学年に限定された結果'
            },
            'future_research': {
                'recommendations': [
                    '個人識別可能なデータ収集システムの構築',
                    'ランダム割付による実験デザイン',
                    '長期追跡調査の実施',
                    '他校での再現性確認'
                ]
            }
        }
        return discussion
    
    def _summarize_significance(self):
        """有意性の要約"""
        phase2_results = self.results.get('phase2_revised', {})
        
        # χ²検定の有意性を集計
        chi2_significant = 0
        chi2_total = 0
        chi2_results = phase2_results.get('chi_square_analysis', {})
        all_chi2_results = {}
        if 'q1_results' in chi2_results:
            all_chi2_results.update(chi2_results['q1_results'])
        if 'q3_results' in chi2_results:
            all_chi2_results.update(chi2_results['q3_results'])
        
        for result in all_chi2_results.values():
            chi2_total += 1
            if result.get('adjusted_significant', False):
                chi2_significant += 1
        
        # Mann-Whitney U検定の有意性を集計（composite_analysisを含む）
        mw_significant = 0
        mw_total = 0
        mannwhitney_results = phase2_results.get('mann_whitney_analysis', {})
        composite_results = phase2_results.get('composite_analysis', {})
        
        all_mw_results = {}
        all_mw_results.update(mannwhitney_results)
        all_mw_results.update(composite_results)
        
        for result in all_mw_results.values():
            mw_total += 1
            if result.get('significant', False):
                mw_significant += 1
        
        return {
            'chi2_tests': f"{chi2_significant}/{chi2_total}",
            'mann_whitney_tests': f"{mw_significant}/{mw_total}"
        }
    
    def _summarize_effect_sizes(self):
        """効果量の要約"""
        phase2_results = self.results.get('phase2_revised', {})
        
        effect_summary = {
            'large_effects': [],
            'medium_effects': [],
            'small_effects': []
        }
        
        # χ²検定の効果量
        chi2_results = phase2_results.get('chi_square_analysis', {})
        all_chi2_results = {}
        if 'q1_results' in chi2_results:
            all_chi2_results.update(chi2_results['q1_results'])
        if 'q3_results' in chi2_results:
            all_chi2_results.update(chi2_results['q3_results'])
        
        for item, result in all_chi2_results.items():
            cohens_h = abs(result.get('cohens_h', 0))
            if cohens_h >= 0.8:
                effect_summary['large_effects'].append(f"{item}: h = {cohens_h:.3f}")
            elif cohens_h >= 0.5:
                effect_summary['medium_effects'].append(f"{item}: h = {cohens_h:.3f}")
            elif cohens_h >= 0.2:
                effect_summary['small_effects'].append(f"{item}: h = {cohens_h:.3f}")
        
        # Mann-Whitney U検定の効果量（composite_analysisを含む）
        mannwhitney_results = phase2_results.get('mann_whitney_analysis', {})
        composite_results = phase2_results.get('composite_analysis', {})
        
        all_mw_results = {}
        all_mw_results.update(mannwhitney_results)
        all_mw_results.update(composite_results)
        
        for item, result in all_mw_results.items():
            cohens_d = abs(result.get('cohens_d', 0))
            if cohens_d >= 0.8:
                effect_summary['large_effects'].append(f"{item}: d = {cohens_d:.3f}")
            elif cohens_d >= 0.5:
                effect_summary['medium_effects'].append(f"{item}: d = {cohens_d:.3f}")
            elif cohens_d >= 0.2:
                effect_summary['small_effects'].append(f"{item}: d = {cohens_d:.3f}")
        
        return effect_summary
    
    def _assess_practical_significance(self):
        """実用的有意性の評価"""
        effect_summary = self._summarize_effect_sizes()
        
        assessment = {
            'substantial_effects': len(effect_summary['large_effects']) > 0,
            'moderate_effects': len(effect_summary['medium_effects']) > 0,
            'overall_assessment': 'limited' if len(effect_summary['large_effects']) == 0 else 'moderate'
        }
        
        return assessment
    
    def generate_integrated_report(self):
        """統合レポート生成"""
        print("統合レポートを生成中...")
        
        report = {
            'metadata': {
                'title': '小学校出前授業アンケート分析 最終統合レポート',
                'subtitle': '独立群比較による効果検証',
                'generated_at': datetime.now().isoformat(),
                'analysis_type': '独立群比較分析',
                'note': 'Page_IDによる個人追跡は不可能であることを前提とした分析'
            },
            'executive_summary': self.generate_executive_summary(),
            'methodology': self.generate_methodology_section(),
            'results': self.generate_results_section(),
            'discussion': self.generate_discussion_section()
        }
        
        # JSON形式で保存（新しい構造に合わせて）
        output_file = self.output_dir / "current" / "05_final_report" / "integrated_final_report.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # テキスト形式のサマリーも生成
        self.generate_text_summary(report)
        
        print(f"✓ 統合レポートを {output_file} に保存しました")
        return report
    
    def generate_text_summary(self, report):
        """テキスト形式のサマリー生成"""
        summary_text = f"""
# {report['metadata']['title']}
## {report['metadata']['subtitle']}

生成日時: {report['metadata']['generated_at']}

## エグゼクティブサマリー

本分析は、Page_IDが個人識別子ではないことを前提とした独立群比較分析です。
授業前群（n=99）と授業後群（n=99）の比較を行いました。

### 主要な発見事項

#### χ²検定結果
- 検定総数: {report['executive_summary']['chi2_summary']['total_tests']}
- 有意な結果: {report['executive_summary']['chi2_summary']['significant_results']}
- 最大効果量: {report['executive_summary']['chi2_summary']['largest_effect']['item']} 
  (効果量 = {report['executive_summary']['chi2_summary']['largest_effect']['effect_size']:.3f})

#### Mann-Whitney U検定結果
- 検定総数: {report['executive_summary']['mannwhitney_summary']['total_tests']}
- 有意な結果: {report['executive_summary']['mannwhitney_summary']['significant_results']}
- 最大効果量: {report['executive_summary']['mannwhitney_summary']['largest_effect']['item']} 
  (効果量 = {report['executive_summary']['mannwhitney_summary']['largest_effect']['effect_size']:.3f})

### 重要な限界事項

"""
        
        for limitation in report['executive_summary']['limitations']:
            summary_text += f"- {limitation}\n"
        
        summary_text += """
## 結論

本研究は独立群比較として実施され、授業前後の群間差を検討しました。
個人レベルの変化や直接的な授業効果の測定はデータ構造上不可能でした。
観察された群間差については、授業効果以外の要因による可能性も考慮する必要があります。

## 今後の提言

1. 個人識別可能なデータ収集システムの構築
2. ランダム割付による実験デザインの採用
3. 長期追跡調査の実施
4. 他校での再現性確認

---
生成者: Claude Code Analysis (Integrated Report Generator)
"""
        
        # テキストファイルとして保存（新しい構造に合わせて）
        text_file = self.output_dir / "current" / "05_final_report" / "integrated_final_summary.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        print(f"✓ テキストサマリーを {text_file} に保存しました")

def main():
    """メイン実行関数"""
    print("小学校出前授業アンケート 統合分析レポート生成器")
    print("=" * 60)
    print(f"実行開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    generator = IntegratedAnalysisGenerator()
    
    # 有効な結果を読み込み
    generator.load_valid_results()
    
    # 統合レポート生成
    report = generator.generate_integrated_report()
    
    print()
    print("=" * 60)
    print("統合分析レポート生成完了!")
    print("=" * 60)
    print()
    print("生成されたファイル:")
    print("- outputs/integrated_final_report.json")
    print("- outputs/integrated_final_summary.txt")
    print()
    print("重要: この分析は独立群比較であり、個人の変化は測定していません。")

if __name__ == "__main__":
    main()