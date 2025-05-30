#!/usr/bin/env python3
"""
Phase 2: 教育効果の統計的検証（簡易版）
McNemar検定、対応のあるt検定、効果量算出
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI不要のバックエンド
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
import os

def load_data():
    """データ読み込み"""
    data_dir = 'data/analysis/'
    before_df = pd.read_csv(data_dir + 'before_excel_compliant.csv')
    after_df = pd.read_csv(data_dir + 'after_excel_compliant.csv')
    return before_df, after_df

def prepare_matched_data(before_df, after_df):
    """前後マッチングデータの準備"""
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
    
    # 正答基準
    correct_answers = {
        'Saltwater': True, 'Sugarwater': True, 'Muddywater': False,
        'Ink': False, 'MisoSoup': True, 'SoySauce': True
    }
    
    # マッチングデータ作成
    matched_data = {}
    for substance, (before_col, after_col) in q1_mapping.items():
        before_responses = matched_before.loc[common_ids, before_col]
        after_responses = matched_after.loc[common_ids, after_col]
        
        correct = correct_answers[substance]
        before_correct = (before_responses == correct)
        after_correct = (after_responses == correct)
        
        matched_data[substance] = {
            'before': before_correct,
            'after': after_correct,
            'correct_answer': correct
        }
    
    print(f"マッチング完了: {len(common_ids)}名のデータ")
    return matched_data, common_ids

def mcnemar_analysis(matched_data):
    """McNemar検定による前後比較"""
    print("\n=== McNemar検定による前後比較 ===")
    
    results = []
    for substance, data in matched_data.items():
        before = data['before']
        after = data['after']
        
        # 欠損値除去
        valid_mask = before.notna() & after.notna()
        before_valid = before[valid_mask]
        after_valid = after[valid_mask]
        
        if len(before_valid) == 0:
            continue
        
        # 2x2分割表
        tt = ((before_valid == True) & (after_valid == True)).sum()
        tf = ((before_valid == True) & (after_valid == False)).sum()
        ft = ((before_valid == False) & (after_valid == True)).sum()
        ff = ((before_valid == False) & (after_valid == False)).sum()
        
        contingency = np.array([[tt, tf], [ft, ff]])
        
        # McNemar検定
        if tf + ft > 0:
            try:
                result = mcnemar(contingency, exact=True)
                p_value = result.pvalue
            except:
                result = mcnemar(contingency, exact=False)
                p_value = result.pvalue
        else:
            p_value = 1.0
        
        # 正答率計算
        before_rate = before_valid.mean() * 100
        after_rate = after_valid.mean() * 100
        change = after_rate - before_rate
        
        results.append({
            '物質': substance,
            'N': len(before_valid),
            '授業前正答率': round(before_rate, 1),
            '授業後正答率': round(after_rate, 1),
            '変化': round(change, 1),
            'p値': round(p_value, 4)
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # 多重比較補正
    p_values = results_df['p値'].values
    rejected, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
    
    results_df['p値_補正'] = np.round(p_corrected, 4)
    results_df['有意'] = rejected
    
    print(f"\n多重比較補正後 (Bonferroni法):")
    significant = results_df[results_df['有意']]
    if len(significant) > 0:
        print("有意な変化:")
        for _, row in significant.iterrows():
            direction = "改善" if row['変化'] > 0 else "悪化"
            print(f"・{row['物質']}: {row['変化']:+.1f}ポイント ({direction}, p={row['p値_補正']:.4f})")
    else:
        print("有意な変化なし")
    
    return results_df

def composite_score_analysis(matched_data):
    """総合スコア分析"""
    print("\n=== 総合スコア分析 ===")
    
    before_scores = []
    after_scores = []
    
    # 全ての個人のスコアを計算
    substances = list(matched_data.keys())
    n_individuals = len(next(iter(matched_data.values()))['before'])
    
    for i in range(n_individuals):
        before_total = 0
        after_total = 0
        valid_count = 0
        
        for substance in substances:
            before_val = matched_data[substance]['before'].iloc[i]
            after_val = matched_data[substance]['after'].iloc[i]
            
            if pd.notna(before_val) and pd.notna(after_val):
                before_total += int(before_val)
                after_total += int(after_val)
                valid_count += 1
        
        if valid_count >= 4:  # 最低4項目有効
            before_scores.append(before_total / valid_count * 100)
            after_scores.append(after_total / valid_count * 100)
    
    before_scores = np.array(before_scores)
    after_scores = np.array(after_scores)
    
    print(f"分析対象: {len(before_scores)}名")
    print(f"授業前平均: {before_scores.mean():.1f}% (SD: {before_scores.std():.1f})")
    print(f"授業後平均: {after_scores.mean():.1f}% (SD: {after_scores.std():.1f})")
    
    # 対応のあるt検定
    t_stat, p_value = stats.ttest_rel(after_scores, before_scores)
    
    # 効果量 (Cohen's d)
    diff = after_scores - before_scores
    cohens_d = diff.mean() / diff.std()
    
    print(f"\n対応のあるt検定:")
    print(f"t統計量: {t_stat:.3f}")
    print(f"p値: {p_value:.4f}")
    print(f"Cohen's d: {cohens_d:.3f}")
    
    # 効果量解釈
    if abs(cohens_d) < 0.2:
        interpretation = "小"
    elif abs(cohens_d) < 0.5:
        interpretation = "中"
    elif abs(cohens_d) < 0.8:
        interpretation = "大"
    else:
        interpretation = "非常に大"
    
    print(f"効果量: {interpretation}")
    
    # 95%信頼区間
    se = diff.std() / np.sqrt(len(diff))
    ci_lower = diff.mean() - 1.96 * se
    ci_upper = diff.mean() + 1.96 * se
    print(f"95%信頼区間: [{ci_lower:.1f}, {ci_upper:.1f}]")
    
    return {
        'before_scores': before_scores,
        'after_scores': after_scores,
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'is_significant': p_value < 0.05
    }

def create_visualizations(mcnemar_results, composite_results):
    """結果の可視化"""
    print("\n=== 結果可視化 ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 正答率変化
    substances = mcnemar_results['物質']
    changes = mcnemar_results['変化']
    colors = ['red' if x < 0 else 'blue' for x in changes]
    
    axes[0,0].bar(substances, changes, color=colors, alpha=0.7)
    axes[0,0].set_title('Q1項目 正答率変化')
    axes[0,0].set_ylabel('変化 (ポイント)')
    axes[0,0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 有意性マーク
    for i, (_, row) in enumerate(mcnemar_results.iterrows()):
        if row['有意']:
            axes[0,0].text(i, row['変化'] + (1 if row['変化'] >= 0 else -1), 
                          '*', ha='center', fontsize=16, color='red')
    
    # 2. 前後比較
    x = np.arange(len(substances))
    width = 0.35
    
    axes[0,1].bar(x - width/2, mcnemar_results['授業前正答率'], width, 
                 label='授業前', alpha=0.7)
    axes[0,1].bar(x + width/2, mcnemar_results['授業後正答率'], width, 
                 label='授業後', alpha=0.7)
    axes[0,1].set_title('授業前後正答率比較')
    axes[0,1].set_ylabel('正答率 (%)')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(substances, rotation=45)
    axes[0,1].legend()
    
    # 3. 個人別変化
    before_scores = composite_results['before_scores']
    after_scores = composite_results['after_scores']
    
    axes[1,0].scatter(before_scores, after_scores, alpha=0.6)
    min_score = min(before_scores.min(), after_scores.min())
    max_score = max(before_scores.max(), after_scores.max())
    axes[1,0].plot([min_score, max_score], [min_score, max_score], 'r--', alpha=0.5)
    axes[1,0].set_xlabel('授業前スコア (%)')
    axes[1,0].set_ylabel('授業後スコア (%)')
    axes[1,0].set_title('個人別総合スコア変化')
    
    # 4. 変化量分布
    changes_individual = after_scores - before_scores
    axes[1,1].hist(changes_individual, bins=15, alpha=0.7, edgecolor='black')
    axes[1,1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[1,1].axvline(x=changes_individual.mean(), color='blue', linestyle='-',
                     label=f'平均: {changes_individual.mean():.1f}')
    axes[1,1].set_xlabel('スコア変化 (ポイント)')
    axes[1,1].set_ylabel('人数')
    axes[1,1].set_title('個人別変化量分布')
    axes[1,1].legend()
    
    plt.tight_layout()
    
    # 保存
    output_dir = 'reports/2025-05-30/'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}phase2_results.png', dpi=300, bbox_inches='tight')
    print(f"図表保存: {output_dir}phase2_results.png")
    plt.close()

def generate_summary(mcnemar_results, composite_results):
    """Phase 2 結果サマリー"""
    print("\n" + "="*60)
    print("Phase 2 教育効果の統計的検証 結果サマリー")
    print("="*60)
    
    significant_items = mcnemar_results[mcnemar_results['有意']]
    improved_items = mcnemar_results[mcnemar_results['変化'] > 0]
    
    print(f"\n📊 分析結果:")
    print(f"・分析項目: {len(mcnemar_results)}項目")
    print(f"・有意な変化: {len(significant_items)}項目")
    print(f"・改善傾向: {len(improved_items)}項目")
    
    print(f"\n📈 総合スコア:")
    avg_change = (composite_results['after_scores'] - composite_results['before_scores']).mean()
    print(f"・平均変化: {avg_change:.1f}ポイント")
    print(f"・効果量: {composite_results['cohens_d']:.3f}")
    print(f"・統計的有意性: {'有意' if composite_results['is_significant'] else '非有意'}")
    
    print(f"\n🎯 主要知見:")
    # 非水溶液項目の改善
    non_solution = mcnemar_results[mcnemar_results['物質'].isin(['Muddywater', 'Ink'])]
    if len(non_solution) > 0:
        avg_improvement = non_solution['変化'].mean()
        print(f"・非水溶液の理解大幅改善: +{avg_improvement:.1f}ポイント")
    
    # 水溶液項目の課題
    solution = mcnemar_results[mcnemar_results['物質'].isin(['MisoSoup', 'SoySauce'])]
    if len(solution) > 0:
        avg_change_solution = solution['変化'].mean()
        if avg_change_solution < 0:
            print(f"・日常的水溶液で課題: {avg_change_solution:.1f}ポイント")
    
    print(f"\n✅ 結論:")
    if len(significant_items) > 0 or composite_results['is_significant']:
        print("🟢 教育効果を統計的に確認")
        print("🟢 Phase 3 (集団間差異分析) に進行可能")
    else:
        print("🟡 限定的な効果 - Phase 3で詳細分析必要")
    
    print("="*60)

def main():
    """メイン実行"""
    print("Phase 2: 教育効果の統計的検証 実行開始")
    print("="*60)
    
    # 1. データ読み込み
    before_df, after_df = load_data()
    
    # 2. マッチングデータ準備
    matched_data, common_ids = prepare_matched_data(before_df, after_df)
    
    # 3. McNemar検定
    mcnemar_results = mcnemar_analysis(matched_data)
    
    # 4. 総合スコア分析
    composite_results = composite_score_analysis(matched_data)
    
    # 5. 可視化
    create_visualizations(mcnemar_results, composite_results)
    
    # 6. サマリー
    generate_summary(mcnemar_results, composite_results)
    
    print("\n🎉 Phase 2 統計的検証完了!")
    return mcnemar_results, composite_results

if __name__ == "__main__":
    mcnemar_results, composite_results = main()