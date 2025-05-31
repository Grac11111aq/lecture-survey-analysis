# 🚀 次のセッション クイックスタートガイド

## 1️⃣ 最初に読むべきドキュメント（必須）

```bash
# 作業ディレクトリに移動
cd /home/grace/projects/social-implement/lecture-survey-analysis/lecture-survey-analysis

# 重要ドキュメントを確認
cat CRITICAL_DATA_ASSUMPTIONS.md    # データ構造の理解（最重要）
cat SESSION_HANDOVER_DOCUMENT.md     # 引き継ぎ内容
```

## 2️⃣ 環境セットアップ

```bash
# Python環境の有効化
source venv/bin/activate

# 必要なパッケージの確認
pip list | grep -E "pandas|numpy|scipy|matplotlib|statsmodels|scikit-learn"
```

## 3️⃣ Phase 2修正版の実行

```bash
# 独立群比較による再分析
python phase2_independent_groups_analysis.py

# 結果の確認
ls -la outputs/phase2_revised_*
```

## 4️⃣ 有効な結果の抽出と統合

```python
# Pythonインタラクティブシェルで実行
python

# 有効な結果の抽出関数を読み込み
exec(open('VALID_RESULTS_EXTRACTION_GUIDE.md').read())  # 注: 実際はPythonファイルとして保存して実行

# または新しい統合スクリプトを作成
# vim phase5_independent_integrated_analysis.py
```

## 📋 作業チェックリスト

### 必須タスク
- [ ] Phase 2修正版の実行と確認
- [ ] 有効な分析結果の抽出（Phase 1, 3, 4）
- [ ] 最終統合レポートの作成

### オプションタスク
- [ ] 可視化の更新（独立群比較として）
- [ ] プレゼンテーション資料の作成
- [ ] 追加の統計分析（必要に応じて）

## ⚠️ 絶対に避けるべきこと

1. **Page_IDでデータをマージしない**
2. **「ペア」「対応」という用語を使わない**
3. **個人の変化として解釈しない**
4. **McNemar検定や対応のあるt検定を使わない**

## 💡 重要な認識

**このデータは匿名化により個人追跡ができない独立群データである。**

すべての分析と解釈はこの前提に基づいて行う。

---

成功を祈ります！ 🎯