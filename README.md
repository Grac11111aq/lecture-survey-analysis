# 小学校出前授業アンケート分析

**独立群比較による効果検証分析**

---

## 🎯 概要

小学校出前授業の教育効果をアンケートデータから統計的に分析するプロジェクトです。

### ⚠️ 重要な前提
- **Page_IDは個人識別子ではない**: 単なるページ番号
- **分析手法**: 独立群比較（授業前群 vs 授業後群）
- **個人追跡は不可能**: ペアデータ分析は実施不可

---

## 🚀 クイックスタート

### 完全分析実行
```bash
# メインスクリプト実行（推奨）
python run_complete_analysis.py
```

### 主要結果の確認
- **分析結果**: `outputs/current/02_group_comparison/`
- **統合レポート**: `outputs/current/05_final_report/`
- **最終報告書**: `docs/reports/comprehensive_final_report.md`

---

## 📁 プロジェクト構造

```
lecture-survey-analysis/
├── 📋 README.md                 # このファイル
├── 🚀 run_complete_analysis.py  # メインエントリーポイント
├── ⚙️ config/                   # プロジェクト設定
├── 🗄️ data/                     # アンケートデータ
├── 📊 outputs/                  # 分析結果
│   ├── current/                # 有効な結果
│   └── archive/                # 過去版・無効結果
├── 🔧 scripts/                  # 分析スクリプト
│   ├── active/                 # 現在有効なスクリプト
│   └── deprecated/             # 廃止済みスクリプト
├── 📚 docs/                     # ドキュメント
├── 💡 examples/                 # サンプルコード
└── 🐍 venv/                     # Python仮想環境
```

---

## 📊 主要な分析結果

| 項目 | 結果 | 統計的有意性 | 効果量 |
|------|------|-------------|--------|
| Q1総合スコア | 授業後群が低い | p=0.0125* | Cohen's d=-0.329 |
| χ²検定（8項目） | 有意差なし | p>0.05 | 最大h=0.268 |
| Mann-Whitney U | クラス間差顕著 | p<0.001 | - |

**解釈**: 予期しない結果として授業後群のスコアが低下。これは学習の質的変化（より慎重で正確な判断）を示唆する可能性。

---

## 📚 詳細ドキュメント

### 🔧 技術ドキュメント
- **[プロジェクト構造](docs/project_management/project_structure.md)**: 詳細なディレクトリ構造
- **[データ前提条件](docs/analysis_guides/data_assumptions.md)**: 重要な分析前提
- **[分析手法ガイド](docs/analysis_guides/)**: 統計手法の詳細

### 📋 プロジェクト管理
- **[クイックスタートガイド](docs/project_management/quickstart_guide.md)**: 詳細な実行手順
- **[セッション引き継ぎ](docs/project_management/session_handover.md)**: 開発引き継ぎ情報
- **[分析計画](docs/project_management/analysis_plan.md)**: 分析戦略と計画

### 📊 分析結果
- **[包括的最終レポート](docs/reports/comprehensive_final_report.md)**: 詳細な分析結果と考察

---

## ⚡ よく使うコマンド

```bash
# 仮想環境有効化
source venv/bin/activate

# 完全分析実行
python run_complete_analysis.py

# 個別スクリプト実行
python scripts/active/02_independent_groups_analysis.py
python scripts/active/05_integrated_report_generator.py

# 設定確認
cat config/analysis_metadata.yaml
```

---

## ⚠️ 重要な注意事項

### ❌ 避けるべきこと
- `scripts/deprecated/` のスクリプト使用（ペアデータ前提で無効）
- McNemar検定、対応のあるt検定の使用
- 「個人の変化」としての結果解釈

### ✅ 推奨される使用方法
- `run_complete_analysis.py` での一括実行
- `outputs/current/` 以下の結果のみ使用
- 「群間差」としての結果解釈

---

## 🛠️ トラブルシューティング

### よくある問題
1. **環境エラー**: `source venv/bin/activate` で仮想環境を有効化
2. **ファイル不見つからない**: `outputs/current/` 構造を確認
3. **結果の解釈**: 個人変化ではなく群間差として解釈

### サポート情報
- **設定ファイル**: `config/analysis_metadata.yaml`
- **実行ログ**: `outputs/current/execution_log.json`
- **詳細ガイド**: `docs/project_management/`

---

## 📝 ライセンス・引用

本プロジェクトは教育研究目的で開発されています。

**生成者**: Claude Code Analysis System  
**更新日**: 2025-05-31  
**バージョン**: 2.0.0（整理済み構造）

---

💡 **重要**: この分析は独立群比較であり、個人レベルの変化は測定していません。すべての解釈は群間差として行ってください。