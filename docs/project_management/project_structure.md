# 小学校出前授業アンケート分析プロジェクト

**独立群比較による効果検証分析（整理済み構造）**

---

## 🎯 プロジェクト概要

本プロジェクトは小学校出前授業の教育効果をアンケートデータから分析するものです。

### ⚠️ 重要な前提条件
- **Page_IDは個人識別子ではない**: 単なるページ番号
- **個人追跡は不可能**: 独立群比較のみ実施可能
- **分析タイプ**: 授業前群（n=99） vs 授業後群（n=99）の比較

---

## 📁 整理済みディレクトリ構造

```
lecture-survey-analysis/
├── 🔧 scripts/                    # 分析スクリプト
│   ├── active/                    # 現在有効なスクリプト ⭐
│   │   ├── 02_independent_groups_analysis.py
│   │   └── 05_integrated_report_generator.py
│   ├── deprecated/                # 廃止済み（ペアデータ前提）
│   │   ├── phase1_comprehensive_analysis.py
│   │   ├── phase2_statistical_testing.py
│   │   ├── phase3_group_analysis.py
│   │   ├── phase4_text_mining.py
│   │   └── phase5_integrated_analysis.py
│   └── utilities/                 # 共通ユーティリティ（空）
├── 📊 outputs/                    # 分析結果
│   ├── current/                   # 最新の有効結果 ⭐
│   │   ├── 02_group_comparison/   # 独立群比較結果
│   │   └── 05_final_report/       # 統合レポート
│   ├── archive/                   # 過去版・無効結果
│   └── figures/                   # 可視化結果
├── 📚 docs/                       # ドキュメント体系
│   ├── analysis_guides/           # 分析手法ガイド
│   ├── project_management/        # プロジェクト管理
│   └── reports/                   # 最終報告書
├── 🗄️ data/                       # データファイル（既存維持）
├── ⚙️ config/                     # 設定ファイル
│   └── analysis_metadata.yaml    # プロジェクト設定
└── 🚀 run_complete_analysis.py    # マスタースクリプト
```

---

## 🚀 クイックスタート

### 1. 完全分析実行（推奨）
```bash
# 仮想環境有効化
source venv/bin/activate

# 完全分析実行（環境チェック + 順次実行 + 検証）
python run_complete_analysis.py
```

### 2. 個別スクリプト実行
```bash
# Phase 2: 独立群比較分析
python scripts/active/02_independent_groups_analysis.py

# Phase 5: 統合レポート生成
python scripts/active/05_integrated_report_generator.py
```

### 3. 結果確認
- **主要結果**: `outputs/current/02_group_comparison/phase2_revised_results.json`
- **統合レポート**: `outputs/current/05_final_report/integrated_final_report.json`
- **最終レポート**: `docs/reports/comprehensive_final_report.md`

---

## 📋 重要ファイル一覧

### 🔧 有効なスクリプト（scripts/active/）
| ファイル | 目的 | 入力 | 出力 | 実行時間 |
|----------|------|------|------|----------|
| `02_independent_groups_analysis.py` | χ²検定・Mann-Whitney U検定 | CSVファイル | JSON・TXT | ~3秒 |
| `05_integrated_report_generator.py` | 有効結果の統合 | JSON群 | 統合JSON・TXT | ~1秒 |

### 📊 主要な分析結果
| 項目 | 結果 | p値 | 効果量 | 解釈 |
|------|------|-----|--------|------|
| Q1総合スコア | 授業後群が低い | p=0.0125 | Cohen's d=-0.329 | 有意差・中程度効果 |
| χ²検定 | 8項目中0項目有意 | - | 最大 h=0.268 | 有意差なし |
| Mann-Whitney U | 5項目中5項目有意 | - | - | クラス間差顕著 |

### 📚 重要ドキュメント
- `config/analysis_metadata.yaml`: プロジェクト設定・依存関係
- `docs/analysis_guides/data_assumptions.md`: データ構造と前提条件
- `docs/project_management/quickstart_guide.md`: クイックスタートガイド

---

## ⚠️ 重要な注意事項

### ❌ 絶対に避けるべきこと
1. **deprecated/ディレクトリのスクリプト使用**
   - ペアデータ前提のため結果が無効
   
2. **無効な統計手法の使用**
   - McNemar検定、対応のあるt検定は禁止
   
3. **間違った解釈**
   - 「個人の変化」ではなく「群間差」として解釈
   - 因果推論には限界あり

### ✅ 推奨される作業方法
1. **マスタースクリプト使用**: `run_complete_analysis.py` で一括実行
2. **設定ファイル確認**: `config/analysis_metadata.yaml` を常に参照
3. **有効結果のみ使用**: `outputs/current/` 以下のみ利用

---

## 🛡️ 散らかり防止システム

### 1. 自動チェック機能
- **廃止パターン検出**: 問題のある手法使用をチェック
- **依存関係確認**: スクリプトの実行順序を自動管理
- **出力検証**: 期待される結果ファイルの存在を確認

### 2. 命名規則
```
スクリプト: {順序番号}_{機能名}_{説明}.py
出力: {順序番号}_{分析名}_{結果種別}.{拡張子}
ドキュメント: {カテゴリ}_{目的}_{詳細}.md
```

### 3. ライフサイクル管理
```
新規作成 → 開発中 → テスト済み → アクティブ → 廃止予定 → アーカイブ
```

---

## 🔧 トラブルシューティング

### エラー対応
```bash
# 設定ファイル確認
cat config/analysis_metadata.yaml

# 環境チェック
python run_complete_analysis.py  # 自動で環境チェック実行

# 個別実行（デバッグ用）
python scripts/active/02_independent_groups_analysis.py
```

### よくある問題
1. **ファイルが見つからない**: `outputs/current/` 構造を確認
2. **スクリプト実行失敗**: deprecated/ のファイルを使用していないか確認
3. **結果の解釈**: 個人変化ではなく群間差として解釈

---

## 📈 分析結果の解釈指針

### 統計的有意性
- **Q1総合**: 授業後群が有意に低い（予期しない結果）
- **χ²検定**: 個別項目では有意差なし
- **効果量**: 小〜中程度、実用的意義は限定的

### 教育的示唆（慎重に）
1. **学習の質的変化**: より正確で慎重な判断への変化の可能性
2. **概念理解の深化**: 表面的回答から熟考した回答への移行
3. **研究限界**: 因果推論には限界、他要因の影響可能性

---

## 🔄 更新履歴

### v2.0.0 (2025-05-31)
- **ディレクトリ構造の完全整理**
- **独立群比較前提への統一**
- **ペアデータ前提分析の廃止**
- **マスタースクリプト導入**
- **メタデータ管理システム導入**

### v1.x (Previous)
- 初期分析実装（ペアデータ前提・無効）

---

## 🤝 開発者向け情報

### 新機能追加時の手順
1. `config/analysis_metadata.yaml` に登録
2. `scripts/active/` に適切な番号で配置
3. 依存関係を明記
4. マスタースクリプトでテスト
5. ドキュメント更新

### 品質保証
- 独立群前提の統計手法のみ使用
- 「個人変化」表現の禁止
- 因果推論の限界明記

---

**💡 重要**: このプロジェクトは独立群比較であり、個人レベルの変化は測定していません。すべての解釈は群間差として行ってください。