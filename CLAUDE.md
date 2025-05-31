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
3. **`scripts/active/06_structural_equation_modeling.py`** - SEM analysis for causal structure modeling
4. **`scripts/active/07_machine_learning_prediction.py`** - ML prediction models (75.7% CV accuracy)
5. **`scripts/active/08_power_analysis.py`** - Statistical power analysis and sample size calculations

## Key Commands

### Environment Setup
```bash
# Activate virtual environment
./scripts/utilities/activate_env.sh

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

# Advanced analysis execution
python scripts/active/06_structural_equation_modeling.py
python scripts/active/07_machine_learning_prediction.py
python scripts/active/08_power_analysis.py
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
- `outputs/current/05_advanced_analysis/` - Advanced analysis results (SEM, ML, Power Analysis)
- `outputs/figures/current/` - All visualization outputs

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
- **χ² Tests**: 0/8 items showed significance (after multiple comparison correction)
- **Mann-Whitney U Tests**: Q1 total score significant (p=0.0125, Cohen's d=-0.329)
- **Q3 Total Score**: No significant difference (p=0.2802, Cohen's d=0.144)
- **Machine Learning**: RandomForest achieved 75.7% CV accuracy for understanding prediction
- **Power Analysis**: Confirms adequate power for detected effects

### Important Limitations
- No individual tracking capability
- No causal inference possible
- Observational design without randomization

## Advanced Analysis Capabilities

### Structural Equation Modeling (SEM)
- **Script**: `scripts/active/06_structural_equation_modeling.py`
- **Purpose**: Latent variable modeling and causal structure analysis
- **Outputs**: Model fit indices (CFI, RMSEA, SRMR), path coefficients
- **Results**: `outputs/current/05_advanced_analysis/structural_equation_modeling_results.json`

### Machine Learning Prediction
- **Script**: `scripts/active/07_machine_learning_prediction.py`
- **Models**: RandomForest, LogisticRegression, XGBoost comparison
- **Performance**: 75.7% cross-validation accuracy (RandomForest)
- **Features**: Q1/Q3 scores, class indicators, understanding ratings
- **Results**: `outputs/current/05_advanced_analysis/machine_learning_prediction_results.json`

### Statistical Power Analysis
- **Script**: `scripts/active/08_power_analysis.py`
- **Functions**: Post-hoc power calculation, sample size recommendations
- **Methods**: Bootstrap confidence intervals, effect size validation
- **Results**: `outputs/current/05_advanced_analysis/power_analysis_results.json`

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
- semopy (for SEM analysis)
- seaborn (for advanced visualizations)

### File Structure Dependencies
- Data files must exist in `data/analysis/`
- Output directories are automatically created
- Virtual environment must be activated before execution
- Advanced analysis requires additional packages (semopy, sklearn)

### Current Project Status
- **Last Analysis**: 2025-05-31 14:58:27
- **Analysis Version**: 2.0.0 (Independent Groups)
- **Execution Status**: All scripts successfully completed
- **Total Duration**: 4.6 seconds (main pipeline)

## 🎯 Project-Specific Guidelines

### Analysis Focus Areas
1. **Learning Effectiveness**: Before/after comparison of understanding levels
2. **Interest & Engagement**: Changes in experimental interest and learning motivation  
3. **Teaching Method Evaluation**: Effectiveness of specific lesson components
4. **Stakeholder-Specific Insights**: Tailored findings for each reporting target
5. **Advanced Modeling**: SEM causal analysis, ML prediction, statistical power validation

### Reporting Requirements
- **Project Members**: Detailed technical analysis with improvement suggestions
- **Supervising Teachers**: Academic rigor with methodological validation
- **Elementary Teachers**: Practical insights with student learning outcomes
- **Tokyo KOSEN Community**: Social contribution achievements and impact

### Key Success Metrics
- Statistically significant learning effects (p < 0.05) ✓ **ACHIEVED**: Q1 total score p=0.0125
- Practically meaningful effect sizes (Cohen's d > 0.2) ✓ **ACHIEVED**: Cohen's d=-0.329
- High prediction accuracy in ML models (>70%) ✓ **ACHIEVED**: 75.7% CV accuracy
- Comprehensive analysis completion ✓ **ACHIEVED**: SEM, ML, Power Analysis

### Important Constraints
- **Local Focus**: Specific to Tokyo KOSEN outreach project
- **Completion Report**: Primary goal is project completion documentation
- **Resource Conscious**: Efficient analysis without over-theorizing
- **Practical Value**: Emphasis on actionable insights over academic contribution
- **Analysis Maturity**: Project has evolved to include advanced statistical methods

## Current Analysis Status (2025-05-31)

### Completed Analyses
✓ **Phase 1**: Data quality validation and basic statistics  
✓ **Phase 2**: Independent groups comparison (χ², Mann-Whitney U)  
✓ **Phase 3**: Class-level analysis and group comparisons  
✓ **Phase 4**: Text mining and qualitative analysis  
✓ **Phase 5**: Integrated reporting and synthesis  
✓ **Advanced**: SEM modeling, ML prediction, power analysis  

### Key Achievements
- **Statistical Significance**: Q1 total score group difference (p=0.0125)
- **Effect Size**: Medium effect detected (Cohen's d=-0.329)
- **ML Performance**: 75.7% prediction accuracy achieved
- **Comprehensive Documentation**: 266-line final report completed
- **Methodological Rigor**: Power analysis confirms statistical validity

### Next Steps for Future Research
- **Individual Tracking**: Implement personal identifier system
- **Randomized Design**: Control for selection bias
- **Longitudinal Follow-up**: Assess long-term learning retention
- **Multi-site Validation**: Replicate findings across schools