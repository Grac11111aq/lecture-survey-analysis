# 解析結果管理のベストプラクティス

## 推奨ディレクトリ構造

```
data/
├── analysis/          # 解析用の最終データセット（既存）
│   ├── before_excel_compliant.csv
│   ├── after_excel_compliant.csv
│   ├── comment.csv
│   └── DATA_CONTEXT.md
│
├── results/          # 解析結果の保存先（新規作成）
│   ├── exploratory/  # 探索的データ分析
│   │   ├── basic_statistics/
│   │   │   ├── summary_stats_20250530.csv
│   │   │   └── missing_values_report.csv
│   │   ├── distributions/
│   │   │   ├── grade_gender_distribution.png
│   │   │   └── response_patterns.html
│   │   └── correlations/
│   │       ├── correlation_matrix.csv
│   │       └── heatmap.png
│   │
│   ├── hypothesis_testing/  # 仮説検証
│   │   ├── pre_post_comparison/
│   │   │   ├── paired_ttest_results.csv
│   │   │   ├── effect_sizes.csv
│   │   │   └── significance_summary.md
│   │   ├── class_comparisons/
│   │   │   └── anova_results.csv
│   │   └── understanding_factors/
│   │       ├── regression_analysis.csv
│   │       └── feature_importance.png
│   │
│   ├── text_analysis/  # テキスト分析（感想文）
│   │   ├── word_frequency/
│   │   │   ├── word_counts.csv
│   │   │   └── wordcloud.png
│   │   ├── sentiment_analysis/
│   │   │   └── sentiment_scores.csv
│   │   └── topic_modeling/
│   │       ├── topics_summary.csv
│   │       └── topic_visualization.html
│   │
│   ├── visualizations/  # 可視化結果
│   │   ├── static/      # 静的画像（PNG, PDF）
│   │   ├── interactive/ # インタラクティブ（HTML）
│   │   └── dashboards/  # ダッシュボード
│   │
│   ├── reports/  # 最終レポート
│   │   ├── summary_report_20250530.md
│   │   ├── executive_summary.pdf
│   │   └── detailed_findings.html
│   │
│   └── metadata/  # メタデータ
│       ├── analysis_log.json
│       ├── version_info.txt
│       └── reproducibility.md
│
├── notebooks/  # 解析用Jupyterノートブック
│   ├── 01_data_exploration.ipynb
│   ├── 02_hypothesis_testing.ipynb
│   ├── 03_text_analysis.ipynb
│   ├── 04_visualization.ipynb
│   └── 05_final_report.ipynb
│
└── scripts/  # 再利用可能な解析スクリプト
    ├── analysis/
    │   ├── statistical_tests.py
    │   ├── effect_size_calculator.py
    │   └── report_generator.py
    └── visualization/
        ├── plot_helpers.py
        └── dashboard_builder.py
```

## 命名規則

### ファイル名
- 日付形式: `YYYYMMDD` (例: 20250530)
- バージョン: `v1.0`, `v2.0` など
- 記述的な名前: `pre_post_understanding_comparison_20250530_v1.csv`

### ディレクトリ名
- 小文字とアンダースコア使用
- 目的が明確な名前
- 階層は3レベルまで

## 解析ワークフロー

### 1. 探索的データ分析（EDA）
```
data/analysis/ → notebooks/01_data_exploration.ipynb → data/results/exploratory/
```

### 2. 仮説検証
```
期待される仮説（DATA_CONTEXT.mdより）:
- 授業前後で水溶液の理解度が向上
- 実験（炎色反応、再結晶）への高い興味
- クラス間での理解度の差異

notebooks/02_hypothesis_testing.ipynb → data/results/hypothesis_testing/
```

### 3. テキスト分析
```
感想文の分析:
- キーワード抽出（「面白い」「分かった」「結晶」など）
- 感情分析
- トピックモデリング

notebooks/03_text_analysis.ipynb → data/results/text_analysis/
```

### 4. 可視化・レポート作成
```
notebooks/04_visualization.ipynb → data/results/visualizations/
notebooks/05_final_report.ipynb → data/results/reports/
```

## データ管理のベストプラクティス

### 1. バージョン管理
- 重要な解析結果は日付とバージョンを付けて保存
- `metadata/analysis_log.json`に解析履歴を記録

### 2. 再現性の確保
- 使用したライブラリのバージョンを記録
- ランダムシードを固定
- 解析手順をドキュメント化

### 3. データ品質管理
- 外れ値の処理方法を記録
- 欠損値の扱いを明記
- データ変換の過程を追跡可能に

### 4. プライバシー保護
- Page_IDのみ使用（個人特定不可）
- クラス単位での集計を基本とする
- 個別の感想文は要約・集計後に公開

## 期待される主要な解析

### 基本統計
- 回答者の属性分布（クラス別）
- 各質問の回答分布
- 欠損値パターン

### 教育効果の測定
- 授業前後の正答率変化（Q1, Q3）
- 理解度の自己評価（Q6）
- 実験への興味度（Q4）

### 相関分析
- 実験への興味と理解度の関係
- クラス別の教育効果の差異
- 自由記述の内容と理解度の関連

### テキストマイニング
- 頻出キーワード（「結晶」「色」「溶ける」など）
- ポジティブ/ネガティブな感想の比率
- 学習内容の理解を示す表現の抽出

## 自動化とワークフロー効率化

### Makefile / スクリプト化
```bash
# 例: make analyze
analyze:
    python scripts/analysis/run_all_analyses.py
    python scripts/visualization/generate_plots.py
    python scripts/analysis/report_generator.py
```

### 定期実行
- 新しいデータが追加された際の自動解析
- 週次/月次レポートの自動生成

## チェックリスト

解析開始前:
- [ ] DATA_CONTEXT.mdを読んで調査背景を理解
- [ ] データの品質チェック（欠損値、異常値）
- [ ] 解析目的と仮説の明確化

解析実施中:
- [ ] 結果を適切なディレクトリに保存
- [ ] メタデータ（日時、パラメータ）を記録
- [ ] 図表に適切なタイトルとラベルを付与

解析完了後:
- [ ] 結果の妥当性を確認
- [ ] 再現可能性を確保（コード、データ、環境）
- [ ] レポートに主要な発見をまとめる