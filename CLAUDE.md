# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**東京高専社会実装プロジェクト - 小学校出前授業成果評価**  
Elementary School Outreach Lesson Analysis for Tokyo KOSEN Social Implementation Project

**プロジェクト目標**: 東京高専が実施した小学校理科出前授業の教育効果をデータに基づいて評価し、プロジェクト完了報告書を作成する。

**報告対象**: プロジェクトメンバー、担当教員、小学校教員、東京高専学生・教員

**Critical Data Constraint**: Page_ID is NOT a personal identifier but simply a page number, making individual tracking impossible. All analyses must be conducted using independent groups comparison methods.

## Core Architecture

### Analysis Pipeline
```
Data Flow: raw/ → intermediate/ → analysis/ → validation/ → outputs/
Analysis Type: Independent Groups Comparison (独立群比較)
Sample Sizes: Before group (n=99) vs After group (n=99)
```

### Active Scripts (Execution Order)
1. **`scripts/active/02_independent_groups_analysis.py`** - Core statistical analysis using χ² tests, Mann-Whitney U tests
2. **`scripts/active/05_integrated_report_generator.py`** - Consolidates results and generates final reports

## Key Commands

### Environment Setup
```bash
# Activate virtual environment
./activate_env.sh

# Verify Python version (requires 3.12+)
python --version
```

### Main Analysis Execution
```bash
# Run complete analysis pipeline
python run_complete_analysis.py

# Individual phase execution
python scripts/active/02_independent_groups_analysis.py
python scripts/active/05_integrated_report_generator.py
```

### Configuration Management
```bash
# View analysis configuration
cat config/analysis_metadata.yaml

# Check execution log
cat outputs/current/execution_log.json
```

## Configuration System

**Central Configuration**: `config/analysis_metadata.yaml`
- Defines active vs deprecated scripts
- Specifies data assumptions and limitations  
- Controls quality validation parameters
- Manages output file specifications

## Output Structure

### Valid Results (Use Only These)
- `outputs/current/02_group_comparison/` - Independent groups analysis results
- `outputs/current/05_final_report/` - Integrated final reports

### Reference Only (Invalid for Analysis)
- `outputs/archive/` - Legacy paired-analysis results (keep for reference, do not use)

## Critical Analysis Constraints

### What This Analysis CAN Do
- Compare before/after groups as independent populations
- Detect group-level differences using appropriate statistical tests
- Calculate effect sizes for group comparisons
- Evaluate educational effectiveness for project completion report
- Provide practical insights for education stakeholders

### What This Analysis CANNOT Do
- Track individual changes (Page_ID ≠ Person_ID)
- Measure causal effects of the intervention
- Use paired statistical methods (McNemar, paired t-tests, etc.)
- Generalize beyond this specific project context
- Conduct theoretical research or theory building

## Quality Safeguards

### Automatic Validations
- Deprecated pattern detection in active scripts
- Output file structure validation
- JSON format integrity checks
- Environment requirement verification

### Prohibited Patterns
- McNemar検定, 対応のあるt検定, paired_ttest
- Individual change language in outputs
- Paired data analysis methods

## Key Findings (Latest Analysis)

### Statistical Results
- **χ² Tests**: 0/8 items showed significance
- **Mann-Whitney U Tests**: 5/5 items showed significance  
- **Q1 Total Score**: Significant difference (p=0.0125, Cohen's d=-0.329)

### Important Limitations
- No individual tracking capability
- No causal inference possible
- Observational design without randomization

## Emergency Procedures

### If Incorrect Analysis Detected
1. Stop execution immediately
2. Check `config/analysis_metadata.yaml` for deprecated files
3. Use only scripts listed in `active_scripts`
4. Re-verify `data_assumptions`

### File Confusion Resolution
1. Consult `config/analysis_metadata.yaml` as source of truth
2. Use only `outputs/current/` for valid results
3. Treat `outputs/archive/` as reference only

## Dependencies

### Required Python Packages
- pandas >= 2.2.3
- numpy >= 2.2.6  
- scipy >= 1.15.3
- matplotlib >= 3.10.3
- statsmodels >= 0.14.4
- scikit-learn >= 1.6.1

### File Structure Dependencies
- Data files must exist in `data/analysis/`
- Output directories must be pre-created
- Virtual environment must be activated before execution

## 🎯 Project-Specific Guidelines

### Analysis Focus Areas
1. **Learning Effectiveness**: Before/after comparison of understanding levels
2. **Interest & Engagement**: Changes in experimental interest and learning motivation  
3. **Teaching Method Evaluation**: Effectiveness of specific lesson components
4. **Stakeholder-Specific Insights**: Tailored findings for each reporting target

### Reporting Requirements
- **Project Members**: Detailed technical analysis with improvement suggestions
- **Supervising Teachers**: Academic rigor with methodological validation
- **Elementary Teachers**: Practical insights with student learning outcomes
- **Tokyo KOSEN Community**: Social contribution achievements and impact

### Key Success Metrics
- Statistically significant learning effects (p < 0.05)
- Practically meaningful effect sizes (Cohen's d > 0.2)
- High prediction accuracy in ML models (>70%)
- Positive qualitative feedback in comments analysis

### Important Constraints
- **Local Focus**: Specific to Tokyo KOSEN outreach project
- **Completion Report**: Primary goal is project completion documentation
- **Resource Conscious**: Efficient analysis without over-theorizing
- **Practical Value**: Emphasis on actionable insights over academic contribution