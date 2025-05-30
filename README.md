# 小学校出前授業アンケート分析環境

このプロジェクトは、社会実装で実施した小学校での出前授業のアンケート分析用環境です。

## セットアップ済みツール

### 1. Python環境（Miniconda）
- **場所**: `./miniconda3`
- **仮想環境**: `survey-analysis` (Python 3.11)

### 2. データ分析ツール
- **pandas**: データ操作・分析
- **numpy**: 数値計算
- **matplotlib**: グラフ描画
- **seaborn**: 統計的可視化
- **plotly**: インタラクティブな可視化
- **scikit-learn**: 機械学習・統計分析
- **nltk**: 自然言語処理
- **wordcloud**: ワードクラウド生成
- **japanize-matplotlib**: 日本語表示対応

### 3. データ閲覧ツール
- **JupyterLab**: ノートブック環境
- **Visidata**: ターミナルでのCSV高速閲覧
- **gnuplotpy**: グラフ作成

## 使い方

### 環境の有効化
```bash
cd /home/grace/projects/social-implementation/lecture-survey-analysis
eval "$(./miniconda3/bin/conda shell.bash hook)"
conda activate survey-analysis
```

### JupyterLabの起動
```bash
jupyter lab
```
ブラウザが自動的に開き、`survey_analysis_sample.ipynb`を開いて分析を開始できます。

### Visidataでのデータ確認
```bash
visidata your_data.csv
```
主なキーバインド：
- `h`,`j`,`k`,`l`: 移動
- `q`: 終了
- `Shift+F`: 周波数表
- `Shift+I`: 統計情報

### サンプルノートブックの構成
1. **データ読み込み**: CSVファイルの読み込みとサンプルデータ生成
2. **基本統計**: データ型、欠損値、分布の確認
3. **可視化分析**: 
   - 学年・性別分布
   - 授業評価の分析
   - インタラクティブな可視化（Sankey図）
4. **テキスト分析**: 感想のワードクラウド生成
5. **詳細分析**: 相関関係の可視化
6. **レポート生成**: 集計結果のサマリー作成
7. **データエクスポート**: Excel形式での結果保存

## CSVファイルの準備

アンケートデータは以下の形式のCSVファイルとして準備してください：

```csv
学年,性別,授業の面白さ,理解度,興味の変化,感想
3年,男,とても面白かった,よく理解できた,とても興味を持った,プログラミングが楽しかった
4年,女,面白かった,理解できた,興味を持った,ロボットが動くのがすごかった
...
```

## 注意事項

- 日本語フォントが正しく表示されない場合は、システムにIPAフォントやNotoフォントをインストールしてください
- 大量のデータを扱う場合は、メモリ使用量に注意してください
- 個人情報を含むデータを扱う場合は、適切に匿名化してください

## トラブルシューティング

### JupyterLabが起動しない場合
```bash
# ポートを指定して起動
jupyter lab --port=8889
```

### 日本語が文字化けする場合
ノートブック内で以下を実行：
```python
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
```

## 更新履歴
- 2025-05-30: 初期環境構築完了