# 有効な分析結果の抽出ガイド

**目的**: 初回分析から有効な部分を抽出し、修正版と統合するための技術ガイド

## 📂 結果ファイルの構造

```
outputs/
├── phase1_detailed_results.json    # ✅ 部分的に有効
├── phase2_detailed_results.json    # ❌ 無効（ペアデータ前提）
├── phase2_revised_results.json     # ✅ 新規作成（独立群比較）
├── phase3_detailed_results.json    # ✅ 部分的に有効
├── phase4_detailed_results.json    # ✅ ほぼ有効
└── phase5_integrated_results.json  # ❌ 再構築必要
```

## 🔍 Phase別有効データ抽出コード

### Phase 1: 基礎統計の抽出

```python
import json
from pathlib import Path

def extract_valid_phase1_results():
    """Phase 1から有効な結果を抽出"""
    
    with open('outputs/phase1_detailed_results.json', 'r', encoding='utf-8') as f:
        phase1_data = json.load(f)
    
    valid_results = {
        'data_quality': {
            # データ品質チェック結果（すべて有効）
            'missing_analysis': phase1_data['quality_check'],
            'sample_sizes': {
                'before': phase1_data['basic_statistics']['class_analysis']['授業前']['total'],
                'after': phase1_data['basic_statistics']['class_analysis']['授業後']['total']
            }
        },
        'descriptive_statistics': {
            # 授業前後別々の記述統計（有効）
            'before': {
                'q1_rates': phase1_data['basic_statistics']['q1_analysis']['before'],
                'q3_rates': phase1_data['basic_statistics']['q3_analysis']['before']
            },
            'after': {
                'q1_rates': phase1_data['basic_statistics']['q1_analysis']['after'],
                'q3_rates': phase1_data['basic_statistics']['q3_analysis']['after'],
                'evaluation_stats': phase1_data['basic_statistics']['evaluation_analysis']
            }
        },
        'class_distribution': phase1_data['basic_statistics']['class_analysis']
    }
    
    # ❌ 除外すべき内容
    # - q1_analysis['comparison'] （個人の変化を前提）
    # - q3_analysis['comparison'] （個人の変化を前提）
    
    return valid_results
```

### Phase 3: クラス間分析の抽出

```python
def extract_valid_phase3_results():
    """Phase 3から有効な結果を抽出"""
    
    with open('outputs/phase3_detailed_results.json', 'r', encoding='utf-8') as f:
        phase3_data = json.load(f)
    
    valid_results = {
        'class_comparisons': {
            # 各時点でのクラス間比較（有効）
            'before_analysis': phase3_data['class_comparison']['before_analysis'],
            'after_analysis': phase3_data['class_comparison']['after_analysis']
            # ❌ 'change_analysis'は除外（変化量分析のため）
        },
        'prediction_model': {
            # 授業後データのみの予測モデル（有効）
            'logistic_regression': phase3_data['factors_analysis']['logistic_regression']
            # 注: multiple_regressionは要確認（変化量を使用していなければ有効）
        }
    }
    
    return valid_results
```

### Phase 4: テキスト分析の抽出

```python
def extract_valid_phase4_results():
    """Phase 4から有効な結果を抽出（解釈修正付き）"""
    
    with open('outputs/phase4_detailed_results.json', 'r', encoding='utf-8') as f:
        phase4_data = json.load(f)
    
    valid_results = phase4_data.copy()  # ほぼすべて有効
    
    # Q2比較の解釈を修正
    if 'frequency_analysis' in valid_results and 'q2_comparison' in valid_results['frequency_analysis']:
        q2_comp = valid_results['frequency_analysis']['q2_comparison']
        
        # 解釈の修正を追記
        q2_comp['interpretation_note'] = (
            "注意: これは授業前群と授業後群の語彙使用の差異を示すものであり、"
            "個人の語彙変化ではない。"
        )
        
        # 用語の修正
        if 'new_words' in q2_comp:
            q2_comp['words_unique_to_after_group'] = q2_comp.pop('new_words')
        if 'disappeared_words' in q2_comp:
            q2_comp['words_unique_to_before_group'] = q2_comp.pop('disappeared_words')
    
    return valid_results
```

## 🔗 統合スクリプトのテンプレート

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最終統合分析スクリプト
独立群比較に基づく修正版
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

class FinalIntegratedAnalyzer:
    def __init__(self):
        self.results_dir = Path("outputs")
        self.integrated_results = {}
        
    def load_all_valid_results(self):
        """すべての有効な結果を読み込み"""
        
        # Phase 1（有効部分）
        self.integrated_results['phase1'] = extract_valid_phase1_results()
        
        # Phase 2（修正版）
        with open(self.results_dir / "phase2_revised_results.json", 'r', encoding='utf-8') as f:
            self.integrated_results['phase2'] = json.load(f)
        
        # Phase 3（有効部分）
        self.integrated_results['phase3'] = extract_valid_phase3_results()
        
        # Phase 4（解釈修正済み）
        self.integrated_results['phase4'] = extract_valid_phase4_results()
        
    def create_final_report(self):
        """最終統合レポートの作成"""
        
        report = {
            'metadata': {
                'analysis_type': '独立群比較分析',
                'limitation': 'Page_IDは個人識別子ではないため、個人追跡は不可能',
                'interpretation_note': '群間差を示すものであり、個人の変化ではない'
            },
            'results': self.integrated_results,
            'conclusions': self.synthesize_conclusions()
        }
        
        return report
        
    def synthesize_conclusions(self):
        """結論の統合（慎重な解釈）"""
        conclusions = {
            'group_differences': [],
            'educational_implications': [],
            'limitations': [],
            'future_research': []
        }
        
        # 具体的な結論の構築...
        
        return conclusions
```

## ⚠️ 統合時の注意事項

### 1. 用語の統一
| 旧用語 | 新用語 |
|--------|--------|
| 変化 | 群間差 |
| 向上/改善 | 授業後群で高い |
| 悪化/低下 | 授業後群で低い |
| 効果 | 差異 |

### 2. 図表の修正
- タイトルに「独立群比較」を明記
- 軸ラベルを「Before Group」「After Group」に
- キャプションに解釈の限界を記載

### 3. 統計量の表記
- 対応のあるt検定 → Mann-Whitney U検定
- McNemar検定 → χ²検定
- 個人レベルの相関 → 群レベルの関連

## 📊 最終チェックリスト

- [ ] すべての「ペア」「対応」という用語を削除
- [ ] 「変化」を「差異」に置換
- [ ] 統計手法が独立群用になっているか確認
- [ ] 解釈に因果関係を示唆する表現がないか確認
- [ ] 限界事項が明記されているか確認

---

このガイドに従って、次のセッションで効率的に有効な結果を抽出・統合すること。