# ファイル管理・作業効率化ガイド
## 重複防止・品質保証システム

**作成日**: 2025-05-31  
**目的**: 効率的な作業進行とファイル散乱防止

## 📁 ファイル構造・命名規則

### 基本ディレクトリ構造
```
lecture-survey-analysis/
├── data/                    # データファイル（変更禁止）
│   ├── raw/                # 原始データ
│   ├── intermediate/       # 中間処理データ  
│   ├── analysis/          # 分析用データ（主要）
│   └── validation/        # 検証用データ
├── scripts/               # 分析スクリプト
│   ├── active/           # 現在有効なスクリプト（★重要★）
│   ├── deprecated/       # 廃止スクリプト（参照禁止）
│   └── utilities/        # ユーティリティ
├── outputs/              # 分析結果出力
│   ├── current/         # 有効な結果（★使用する★）
│   └── archive/         # 廃止された結果（参照禁止）
├── docs/                # ドキュメント
│   ├── project_management/  # プロジェクト管理
│   ├── analysis_guides/     # 分析ガイド
│   └── reports/            # 最終報告書
└── notebooks/           # Jupyter notebook（補助的）
```

### ファイル命名規則

#### 分析スクリプト（scripts/active/）
```
{番号}_{機能名}_{詳細}.py
例: 06_structural_equation_modeling.py
   07_machine_learning_prediction.py
   08_power_analysis.py
```

#### 分析結果（outputs/current/）
```
{フェーズ}_{分析名}/
  ├── {分析名}_results.json      # 詳細結果
  ├── {分析名}_summary.txt       # 要約レポート
  └── {分析名}_figures/          # 図表
  
例: 05_advanced_analysis/
    ├── structural_equation_modeling_results.json
    ├── sem_analysis_summary.txt
    ├── machine_learning_prediction_results.json
    └── ml_prediction_summary.txt
```

#### 報告書（docs/reports/）
```
{対象}_{種類}_{日付}.{拡張子}
例: project_members_technical_report_20251231.md
   school_teachers_practical_report_20251231.md
```

## ⚠️ 使用禁止・注意ファイル

### 🚫 絶対に使用禁止
```
scripts/deprecated/          # 廃止されたスクリプト
outputs/archive/            # 無効な分析結果
reports/INVALIDATED_2025-05-30/  # 無効化された報告書
```

### ⚠️ 参照時注意が必要
```
backup_*/                   # バックアップファイル
*_old.*                    # 旧バージョンファイル
temp_*                     # 一時ファイル
```

## ✅ 現在有効なファイル一覧

### データファイル（確定版）
- `data/analysis/before_excel_compliant.csv` - 授業前アンケート
- `data/analysis/after_excel_compliant.csv` - 授業後アンケート  
- `data/analysis/comment.csv` - 感想・コメント

### 分析スクリプト（最新版）
- `scripts/active/02_independent_groups_analysis.py` - 独立群比較分析
- `scripts/active/05_integrated_report_generator.py` - 統合レポート生成
- `scripts/active/06_structural_equation_modeling.py` - SEM分析
- `scripts/active/07_machine_learning_prediction.py` - 機械学習予測
- `scripts/active/08_power_analysis.py` - 検出力分析

### 分析結果（有効版）
- `outputs/current/02_group_comparison/` - 独立群比較結果
- `outputs/current/05_final_report/` - 統合最終報告
- `outputs/current/05_advanced_analysis/` - 高度分析結果

## 🔄 作業フロー・重複防止

### 新規分析開始前のチェックリスト
```
□ config/analysis_metadata.yaml で有効スクリプト確認
□ outputs/current/ で既存結果確認
□ docs/project_management/ で要件・方針確認
□ 類似分析の実施済み確認
```

### 分析実行時の注意事項
1. **必ずscripts/active/のスクリプトのみ使用**
2. **outputs/current/に結果を出力**
3. **実行前に既存結果との重複確認**
4. **実行後は結果の妥当性確認**

### 結果保存時のチェック
```
□ JSON形式での詳細結果保存
□ TXT形式での要約レポート作成
□ PNG形式での図表保存
□ ファイル名の命名規則遵守
```

## 📋 品質保証システム

### 分析品質チェックポイント
1. **データ整合性**: 正しいデータファイルの使用確認
2. **方法論妥当性**: 独立群比較前提の遵守
3. **統計的妥当性**: 適切な検定手法・効果量計算
4. **解釈妥当性**: 過度な因果推論の回避

### 結果検証プロセス
```
1. 自動チェック: スクリプト内での基本検証
2. 手動チェック: 結果の妥当性・整合性確認  
3. 相互チェック: 複数手法での結果比較
4. 文書化: 分析過程・結果の明確な記録
```

## 🎯 効率化のための運用ルール

### 日常作業での注意点
- **作業開始時**: 必ずTodoリストで現在のタスク確認
- **ファイル参照時**: パスを相対パスでなく絶対パスで指定
- **コード実行時**: 仮想環境の確実な有効化
- **結果確認時**: outputs/current/のみを参照

### コミュニケーション効率化
- **進捗報告**: TodoWriteツールでの進捗更新
- **課題報告**: 具体的なファイルパス・エラー内容の記載
- **成果共有**: 標準化された形式での結果提示

### バックアップ・バージョン管理
- **定期コミット**: 作業完了時の確実なGitコミット
- **明確なコミットメッセージ**: 変更内容の具体的記述
- **ブランチ戦略**: main ブランチでの安定版管理

## 🚨 緊急時対応手順

### ファイル破損・消失時
1. Git履歴からの復旧
2. バックアップディレクトリからの復元
3. 最新コミットからの再実行

### 分析結果の不整合発見時
1. 使用したファイル・スクリプトの確認
2. 分析手法の妥当性再検討
3. 必要に応じて再分析実行

### 重複作業の発見時
1. 既存結果との差異確認
2. より適切な手法の採用判断
3. 不要な結果の適切な整理

## 📊 作業進捗管理

### 完了済みタスク（参考用）
- [x] 基礎データ品質確認
- [x] 独立群比較分析
- [x] 統合レポート生成
- [x] 高度分析（SEM・機械学習・検出力）
- [x] プロジェクト方針・要件文書化

### 進行中・予定タスク
- [ ] 対象別報告書作成
- [ ] ステークホルダー向け資料準備
- [ ] 最終品質確認・検証

---

**管理責任**: Claude Code Analysis  
**更新頻度**: 作業フェーズ変更時  
**参照義務**: 全分析作業開始前に必読