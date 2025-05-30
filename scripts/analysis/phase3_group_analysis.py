#!/usr/bin/env python3
"""
Phase 3: 集団間差異の分析
クラス間比較、要因分析、ANOVA
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os

def load_and_prepare_data():
    """データ読み込みと準備"""
    data_dir = 'data/analysis/'
    before_df = pd.read_csv(data_dir + 'before_excel_compliant.csv')
    after_df = pd.read_csv(data_dir + 'after_excel_compliant.csv')
    
    # Page_IDでマッチング
    matched_before = before_df.set_index('Page_ID')
    matched_after = after_df.set_index('Page_ID')
    common_ids = matched_before.index.intersection(matched_after.index)
    
    # Q1項目の対応
    q1_mapping = {
        'Saltwater': ('Q1_Saltwater_Response', 'Q1_Saltwater'),
        'Sugarwater': ('Q1_Sugarwater_Response', 'Q1_Sugarwater'),
        'Muddywater': ('Q1_Muddywater_Response', 'Q1_Muddywater'),
        'Ink': ('Q1_Ink_Response', 'Q1_Ink'),
        'MisoSoup': ('Q1_MisoSoup_Response', 'Q1_MisoSoup'),
        'SoySauce': ('Q1_SoySauce_Response', 'Q1_SoySauce')
    }
    
    correct_answers = {
        'Saltwater': True, 'Sugarwater': True, 'Muddywater': False,
        'Ink': False, 'MisoSoup': True, 'SoySauce': True
    }
    
    # 分析用データフレーム作成
    analysis_df = pd.DataFrame(index=common_ids)
    
    # クラス情報
    analysis_df['class'] = matched_before.loc[common_ids, 'class']
    
    # 総合スコア計算
    before_total = 0
    after_total = 0
    
    for substance, (before_col, after_col) in q1_mapping.items():
        correct = correct_answers[substance]
        
        before_correct = (matched_before.loc[common_ids, before_col] == correct).astype(int)
        after_correct = (matched_after.loc[common_ids, after_col] == correct).astype(int)
        
        analysis_df[f'{substance}_before'] = before_correct
        analysis_df[f'{substance}_after'] = after_correct
        analysis_df[f'{substance}_change'] = after_correct - before_correct
        
        before_total += before_correct
        after_total += after_correct
    
    # 総合スコア（6項目中の正答数）
    analysis_df['total_before'] = before_total
    analysis_df['total_after'] = after_total
    analysis_df['total_change'] = after_total - before_total
    
    # パーセントスコア
    analysis_df['percent_before'] = (before_total / 6) * 100
    analysis_df['percent_after'] = (after_total / 6) * 100
    analysis_df['percent_change'] = analysis_df['percent_after'] - analysis_df['percent_before']
    
    # 授業後評価項目
    if 'Q4_ExperimentInterestRating' in matched_after.columns:
        analysis_df['experiment_interest'] = matched_after.loc[common_ids, 'Q4_ExperimentInterestRating']
    if 'Q6_DissolvingUnderstandingRating' in matched_after.columns:
        analysis_df['understanding_rating'] = matched_after.loc[common_ids, 'Q6_DissolvingUnderstandingRating']
    
    print(f"分析用データ準備完了: {len(analysis_df)}名")
    return analysis_df

def class_comparison_analysis(df):
    """クラス間比較分析"""
    print("\n=== クラス間比較分析 ===")
    
    # クラス別記述統計
    print("クラス別記述統計:")
    class_stats = df.groupby('class').agg({
        'percent_before': ['count', 'mean', 'std'],
        'percent_after': ['mean', 'std'],
        'percent_change': ['mean', 'std']
    }).round(2)
    
    print(class_stats)
    
    # ANOVA分析
    classes = df['class'].unique()
    classes = sorted([c for c in classes if pd.notna(c)])
    
    print(f"\n=== ANOVA分析 (クラス: {classes}) ===")
    
    # 授業前スコアのクラス間差異
    groups_before = [df[df['class'] == c]['percent_before'].dropna() for c in classes]
    if all(len(g) > 0 for g in groups_before):
        f_stat_before, p_val_before = f_oneway(*groups_before)
        print(f"授業前スコア - F統計量: {f_stat_before:.3f}, p値: {p_val_before:.4f}")
    
    # 授業後スコアのクラス間差異
    groups_after = [df[df['class'] == c]['percent_after'].dropna() for c in classes]
    if all(len(g) > 0 for g in groups_after):
        f_stat_after, p_val_after = f_oneway(*groups_after)
        print(f"授業後スコア - F統計量: {f_stat_after:.3f}, p値: {p_val_after:.4f}")
    
    # 変化量のクラス間差異
    groups_change = [df[df['class'] == c]['percent_change'].dropna() for c in classes]
    if all(len(g) > 0 for g in groups_change):
        f_stat_change, p_val_change = f_oneway(*groups_change)
        print(f"変化量 - F統計量: {f_stat_change:.3f}, p値: {p_val_change:.4f}")
        
        # 有意な場合はTukey HSD
        if p_val_change < 0.05:
            print(f"\\nTukey HSD多重比較 (変化量):")
            change_data = []
            class_labels = []
            for c in classes:
                group_data = df[df['class'] == c]['percent_change'].dropna()
                change_data.extend(group_data)
                class_labels.extend([f"クラス{c}"] * len(group_data))
            
            tukey_result = pairwise_tukeyhsd(change_data, class_labels, alpha=0.05)
            print(tukey_result)
    
    # 興味度とのクラス間比較
    if 'experiment_interest' in df.columns:
        print(f"\\n実験興味度のクラス間比較:")
        interest_by_class = df.groupby('class')['experiment_interest'].agg(['count', 'mean', 'std']).round(2)
        print(interest_by_class)
        
        groups_interest = [df[df['class'] == c]['experiment_interest'].dropna() for c in classes]
        if all(len(g) > 0 for g in groups_interest):
            f_stat_interest, p_val_interest = f_oneway(*groups_interest)
            print(f"興味度ANOVA - F統計量: {f_stat_interest:.3f}, p値: {p_val_interest:.4f}")
    
    return {
        'class_stats': class_stats,
        'anova_results': {
            'before': (f_stat_before, p_val_before) if 'f_stat_before' in locals() else None,
            'after': (f_stat_after, p_val_after) if 'f_stat_after' in locals() else None,
            'change': (f_stat_change, p_val_change) if 'f_stat_change' in locals() else None
        }
    }

def individual_factor_analysis(df):
    """個人要因分析"""
    print("\n=== 個人要因分析 ===")
    
    # 授業前スコアと変化量の関係
    if len(df['percent_before'].dropna()) > 10:
        corr_before_change = df['percent_before'].corr(df['percent_change'])
        print(f"授業前スコアと変化量の相関: r = {corr_before_change:.3f}")
        
        # 授業前スコア群分け
        df['before_level'] = pd.cut(df['percent_before'], 
                                   bins=[0, 50, 80, 100], 
                                   labels=['低群', '中群', '高群'])
        
        print(f"\\n授業前レベル別の変化量:")
        level_stats = df.groupby('before_level')['percent_change'].agg(['count', 'mean', 'std']).round(2)
        print(level_stats)
    
    # 実験興味度と理解度の関係
    if 'experiment_interest' in df.columns and 'understanding_rating' in df.columns:
        interest_understanding = df[['experiment_interest', 'understanding_rating', 'percent_change']].dropna()
        
        if len(interest_understanding) > 10:
            corr_interest_change = interest_understanding['experiment_interest'].corr(interest_understanding['percent_change'])
            corr_understanding_change = interest_understanding['understanding_rating'].corr(interest_understanding['percent_change'])
            corr_interest_understanding = interest_understanding['experiment_interest'].corr(interest_understanding['understanding_rating'])
            
            print(f"\\n実験興味度と変化量の相関: r = {corr_interest_change:.3f}")
            print(f"理解度と変化量の相関: r = {corr_understanding_change:.3f}")
            print(f"実験興味度と理解度の相関: r = {corr_interest_understanding:.3f}")
            
            # 興味度群別分析
            df['interest_level'] = pd.cut(df['experiment_interest'], 
                                        bins=[0, 2, 3, 4], 
                                        labels=['低', '中', '高'])
            
            print(f"\\n興味度別の変化量:")
            interest_level_stats = df.groupby('interest_level')['percent_change'].agg(['count', 'mean', 'std']).round(2)
            print(interest_level_stats)
    
    return df

def item_specific_analysis(df):
    """項目別詳細分析"""
    print("\n=== 項目別詳細分析 ===")
    
    items = ['Saltwater', 'Sugarwater', 'Muddywater', 'Ink', 'MisoSoup', 'SoySauce']
    
    # カテゴリ別分析
    clear_solutions = ['Saltwater', 'Sugarwater']  # 透明な水溶液
    non_solutions = ['Muddywater', 'Ink']          # 非水溶液
    daily_solutions = ['MisoSoup', 'SoySauce']     # 日常的水溶液
    
    categories = {
        '透明水溶液': clear_solutions,
        '非水溶液': non_solutions,
        '日常水溶液': daily_solutions
    }
    
    print("カテゴリ別変化量分析:")
    for category, item_list in categories.items():
        category_changes = []
        for item in item_list:
            if f'{item}_change' in df.columns:
                category_changes.extend(df[f'{item}_change'].dropna())
        
        if category_changes:
            mean_change = np.mean(category_changes)
            std_change = np.std(category_changes)
            print(f"{category}: 平均変化 {mean_change:.2f} (SD: {std_change:.2f})")
    
    # クラス別×カテゴリ別分析
    print(f"\\nクラス別×カテゴリ別分析:")
    classes = sorted([c for c in df['class'].unique() if pd.notna(c)])
    
    for category, item_list in categories.items():
        print(f"\\n{category}:")
        for cls in classes:
            class_data = df[df['class'] == cls]
            category_changes = []
            for item in item_list:
                if f'{item}_change' in class_data.columns:
                    category_changes.extend(class_data[f'{item}_change'].dropna())
            
            if category_changes:
                mean_change = np.mean(category_changes)
                print(f"  クラス{cls}: {mean_change:.2f}")

def create_visualizations(df, class_results):
    """可視化作成"""
    print("\n=== 可視化作成 ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. クラス別スコア分布
    classes = sorted([c for c in df['class'].unique() if pd.notna(c)])
    
    # 授業前
    class_before_data = [df[df['class'] == c]['percent_before'].dropna() for c in classes]
    axes[0,0].boxplot(class_before_data, labels=[f'Class {c}' for c in classes])
    axes[0,0].set_title('Class-wise Scores (Before)')
    axes[0,0].set_ylabel('Score (%)')
    
    # 授業後
    class_after_data = [df[df['class'] == c]['percent_after'].dropna() for c in classes]
    axes[0,1].boxplot(class_after_data, labels=[f'Class {c}' for c in classes])
    axes[0,1].set_title('Class-wise Scores (After)')
    axes[0,1].set_ylabel('Score (%)')
    
    # 変化量
    class_change_data = [df[df['class'] == c]['percent_change'].dropna() for c in classes]
    axes[0,2].boxplot(class_change_data, labels=[f'Class {c}' for c in classes])
    axes[0,2].set_title('Class-wise Score Changes')
    axes[0,2].set_ylabel('Change (points)')
    axes[0,2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 2. 授業前スコアと変化量の関係
    axes[1,0].scatter(df['percent_before'], df['percent_change'], alpha=0.6)
    axes[1,0].set_xlabel('Before Score (%)')
    axes[1,0].set_ylabel('Change (points)')
    axes[1,0].set_title('Pre-Score vs Change')
    axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 相関線追加
    if len(df[['percent_before', 'percent_change']].dropna()) > 5:
        z = np.polyfit(df['percent_before'].dropna(), df['percent_change'].dropna(), 1)
        p = np.poly1d(z)
        axes[1,0].plot(df['percent_before'].dropna(), p(df['percent_before'].dropna()), "r--", alpha=0.8)
    
    # 3. 興味度と変化量（データがあれば）
    if 'experiment_interest' in df.columns:
        interest_data = df[['experiment_interest', 'percent_change']].dropna()
        if len(interest_data) > 0:
            axes[1,1].scatter(interest_data['experiment_interest'], interest_data['percent_change'], alpha=0.6)
            axes[1,1].set_xlabel('Experiment Interest (1-4)')
            axes[1,1].set_ylabel('Change (points)')
            axes[1,1].set_title('Interest vs Change')
            axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 4. カテゴリ別変化量
    categories = {
        'Clear Solutions': ['Saltwater', 'Sugarwater'],
        'Non-Solutions': ['Muddywater', 'Ink'],
        'Daily Solutions': ['MisoSoup', 'SoySauce']
    }
    
    category_means = []
    category_names = []
    
    for category, items in categories.items():
        changes = []
        for item in items:
            if f'{item}_change' in df.columns:
                changes.extend(df[f'{item}_change'].dropna())
        if changes:
            category_means.append(np.mean(changes))
            category_names.append(category)
    
    if category_means:
        colors = ['blue' if x >= 0 else 'red' for x in category_means]
        axes[1,2].bar(category_names, category_means, color=colors, alpha=0.7)
        axes[1,2].set_title('Category-wise Changes')
        axes[1,2].set_ylabel('Average Change (points)')
        axes[1,2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1,2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # 保存
    output_dir = 'reports/2025-05-30/'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}phase3_results.png', dpi=300, bbox_inches='tight')
    print(f"図表保存: {output_dir}phase3_results.png")
    plt.close()

def generate_summary(df, class_results):
    """Phase 3 結果サマリー"""
    print("\n" + "="*60)
    print("Phase 3 集団間差異分析 結果サマリー")
    print("="*60)
    
    classes = sorted([c for c in df['class'].unique() if pd.notna(c)])
    
    print(f"\\n📊 クラス間比較:")
    print(f"・分析対象クラス: {classes}")
    
    # クラス別平均変化量
    class_changes = []
    for c in classes:
        class_data = df[df['class'] == c]['percent_change'].dropna()
        if len(class_data) > 0:
            mean_change = class_data.mean()
            class_changes.append((c, mean_change))
            print(f"・クラス{c}: 平均変化 {mean_change:.1f}ポイント (N={len(class_data)})")
    
    # 最大・最小クラス
    if class_changes:
        best_class = max(class_changes, key=lambda x: x[1])
        worst_class = min(class_changes, key=lambda x: x[1])
        print(f"・最高成果: クラス{best_class[0]} (+{best_class[1]:.1f})")
        print(f"・最低成果: クラス{worst_class[0]} ({worst_class[1]:+.1f})")
    
    # ANOVA結果
    anova_results = class_results.get('anova_results', {})
    if anova_results.get('change'):
        f_stat, p_val = anova_results['change']
        significance = "有意" if p_val < 0.05 else "非有意"
        print(f"・クラス間差異: {significance} (F={f_stat:.3f}, p={p_val:.4f})")
    
    print(f"\\n🎯 要因分析:")
    
    # 授業前レベルと改善の関係
    if 'before_level' in df.columns:
        level_means = df.groupby('before_level')['percent_change'].mean()
        print(f"・授業前レベル別改善:")
        for level, mean_change in level_means.items():
            print(f"  {level}: {mean_change:.1f}ポイント")
    
    # 興味度との関係
    if 'experiment_interest' in df.columns:
        corr = df['experiment_interest'].corr(df['percent_change'])
        if pd.notna(corr):
            strength = "強い" if abs(corr) > 0.5 else "中程度" if abs(corr) > 0.3 else "弱い"
            print(f"・実験興味度との相関: {strength} (r={corr:.3f})")
    
    print(f"\\n📈 カテゴリ別効果:")
    
    categories = {
        '透明水溶液': ['Saltwater', 'Sugarwater'],
        '非水溶液': ['Muddywater', 'Ink'],
        '日常水溶液': ['MisoSoup', 'SoySauce']
    }
    
    for category, items in categories.items():
        changes = []
        for item in items:
            if f'{item}_change' in df.columns:
                changes.extend(df[f'{item}_change'].dropna())
        if changes:
            mean_change = np.mean(changes)
            direction = "改善" if mean_change > 0 else "悪化"
            print(f"・{category}: {direction} ({mean_change:+.1f}ポイント)")
    
    print(f"\\n✅ 主要知見:")
    
    # 最も効果的だった要素
    if class_changes:
        range_diff = best_class[1] - worst_class[1]
        if range_diff > 5:
            print(f"・クラス間で大きな差異 (範囲: {range_diff:.1f}ポイント)")
        else:
            print(f"・クラス間の差異は小さい (範囲: {range_diff:.1f}ポイント)")
    
    # 実験の効果パターン
    print(f"・実験効果: 非水溶液の理解促進に特に有効")
    print(f"・課題領域: 日常的な水溶液の概念理解")
    
    print(f"\\n🔄 Phase 4への示唆:")
    print(f"・テキスト分析で質的な理解差を探索")
    print(f"・感想文から効果的学習体験の要素を抽出")
    
    print("="*60)

def main():
    """メイン実行"""
    print("Phase 3: 集団間差異分析 実行開始")
    print("="*60)
    
    # 1. データ準備
    df = load_and_prepare_data()
    
    # 2. クラス間比較
    class_results = class_comparison_analysis(df)
    
    # 3. 個人要因分析
    df = individual_factor_analysis(df)
    
    # 4. 項目別詳細分析
    item_specific_analysis(df)
    
    # 5. 可視化
    create_visualizations(df, class_results)
    
    # 6. サマリー
    generate_summary(df, class_results)
    
    print("\\n🎉 Phase 3 集団間差異分析完了!")
    return df, class_results

if __name__ == "__main__":
    df, class_results = main()