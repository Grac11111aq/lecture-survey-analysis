#!/bin/bash
# アンケート分析環境の有効化スクリプト

echo "アンケート分析環境を有効化しています..."
cd "$(dirname "$0")"
eval "$(./miniconda3/bin/conda shell.bash hook)"
conda activate survey-analysis

echo "環境が有効化されました。"
echo "JupyterLabを起動する場合: jupyter lab"
echo "Visidataを使う場合: visidata your_data.csv"
echo ""
echo "Python環境: $(which python)"
echo "インストール済みパッケージを確認: pip list"

# シェルを起動（環境を維持）
exec bash