# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a data analysis environment for elementary school outreach program surveys (小学校出前授業アンケート分析). The project uses Python with JupyterLab for analyzing survey responses collected from students.

## Environment Setup and Common Commands

### Activate Python Environment
```bash
cd /home/grace/projects/social-implementation/lecture-survey-analysis
eval "$(./miniconda3/bin/conda shell.bash hook)"
conda activate survey-analysis
```

Alternatively, use the convenience script:
```bash
./activate_env.sh
```

### Common Development Commands
- **Start JupyterLab**: `jupyter lab`
- **View CSV data quickly**: `visidata data.csv` (press `q` to quit, `Shift+F` for frequency table)
- **Install new packages**: `pip install package_name` or `conda install -c conda-forge package_name`
- **Export environment**: `conda env export > environment.yml`

### Git Operations
- **Check status**: `git status`
- **Commit changes**: `git add . && git commit -m "message"`
- **Push to GitHub**: `git push`
- **GitHub Repository**: https://github.com/Grac11111aq/lecture-survey-analysis
- **Default Branch**: `main` (masterブランチは削除済み)

## Data Architecture

### Expected CSV Data Format
The analysis expects survey data in the following format:
- `学年` (Grade): 3年, 4年, 5年, 6年
- `性別` (Gender): 男, 女
- `授業の面白さ` (Class Interest): とても面白かった, 面白かった, 普通, あまり面白くなかった
- `理解度` (Understanding): よく理解できた, 理解できた, 少し理解できた, 理解できなかった
- `興味の変化` (Interest Change): とても興味を持った, 興味を持った, 変わらない, 興味が減った
- `感想` (Comments): Free text

### Available Data Files
- `before.csv`: Pre-class survey data (匿名化済み、GitHubで公開)
- `after.csv`: Post-class survey data (匿名化済み、GitHubで公開)
- `comment.csv`: Additional comments (匿名化済み、GitHubで公開)

**Note**: すべてのCSVデータは個人情報を含まない匿名化済みデータです。Page_IDを使用して個人を特定できないようになっています。

## Analysis Pipeline Architecture

The sample notebook (`survey_analysis_sample.ipynb`) follows this structure:

1. **Data Loading**: Reads CSV files or generates sample data
2. **Basic Statistics**: Data types, missing values, distributions
3. **Visualization Analysis**:
   - Demographic distributions (grade/gender)
   - Class evaluation metrics
   - Interactive visualizations using Plotly (Sankey diagrams)
4. **Text Analysis**: Word cloud generation from comments
5. **Correlation Analysis**: Heatmap visualization
6. **Report Generation**: Summary statistics and Excel export

## Key Technical Considerations

### Japanese Language Support
- Matplotlib requires Japanese font configuration
- Default uses `IPAexGothic` font
- Word clouds need proper font path (`/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`)

### Installed Analysis Stack
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn
- **NLP**: nltk, wordcloud
- **Japanese Support**: japanize-matplotlib

### Environment Management
The project uses Miniconda installed locally in `./miniconda3`. This avoids system-level dependencies and ensures reproducibility.

## Project Updates (2025-05-30)

### Recent Changes
1. **GitHub Integration**: 
   - リポジトリをGitHubに公開: https://github.com/Grac11111aq/lecture-survey-analysis
   - デフォルトブランチを`main`に統一（`master`ブランチは削除）
   
2. **Data Sharing**:
   - 匿名化済みCSVファイル（before.csv, after.csv, comment.csv）をGitHubで公開
   - .gitignoreを更新してCSVファイルの除外を解除
   
3. **Security Considerations**:
   - Miniconda環境（miniconda3/）は.gitignoreで除外
   - 個人情報を含まないデータのみを公開
   - Page_IDによる匿名化で個人を特定できない仕組み