"""
詳細境界値分析スクリプト
F1スコア定義修正版 + 3特徴量の境界緩和効果 + 最適2条件組み合わせ
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from pathlib import Path
from itertools import combinations, product
import warnings
warnings.filterwarnings('ignore')

def detailed_boundary_analysis(data_file=None, output_dir=None):
    """詳細境界値分析とF1スコア修正版"""
    
    # デフォルト設定
    if data_file is None:
        data_file = "data/processed/preprocessed_100data.csv"
    if output_dir is None:
        output_dir = "output/detailed_boundary_analysis"
    
    # データの読み込み
    data = pd.read_csv(data_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    target_column = '利用意向'
    overall_intent_rate = data[target_column].mean()
    
    print("=== 詳細境界値分析（F1スコア修正版） ===")
    print(f"データ: {data_file}")
    print(f"サンプル数: {len(data)}件")
    print(f"全体利用意向率: {overall_intent_rate:.1%}")
    
    def calculate_metrics(mask, data, target_column):
        """正しいF1スコア等の計算"""
        if mask.sum() == 0:
            return {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
        
        tp = (mask & (data[target_column] == 1)).sum()  # True Positive
        fp = (mask & (data[target_column] == 0)).sum()  # False Positive  
        fn = (~mask & (data[target_column] == 1)).sum() # False Negative
        tn = (~mask & (data[target_column] == 0)).sum() # True Negative
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': precision, 'recall': recall, 'f1_score': f1_score
        }
    
    # 1. 3特徴量それぞれの境界値緩和効果
    print("\n=== 1. 各特徴量の境界値緩和効果分析 ===")
    
    # 最高月収の詳細境界分析
    print("\n【最高月収の境界分析】")
    income_boundaries = []
    income_ranges = [
        (30, 60), (35, 65), (40, 70), (45, 75), (50, 80),
        (40, 60), (45, 70), (50, 75), (35, 70), (40, 80),
        (30, 70), (35, 75), (45, 65), (50, 70), (55, 85)
    ]
    
    for min_val, max_val in income_ranges:
        mask = (data['最高月収'] >= min_val) & (data['最高月収'] <= max_val)
        segment_size = mask.sum()
        segment_intent_rate = data[mask][target_column].mean() if segment_size > 0 else 0
        lift = segment_intent_rate / overall_intent_rate if segment_intent_rate > 0 else 0
        
        metrics = calculate_metrics(mask, data, target_column)
        
        income_boundaries.append({
            'range': f"{min_val}-{max_val}",
            'min': min_val, 'max': max_val,
            'segment_size': segment_size,
            'intent_rate': segment_intent_rate,
            'lift': lift,
            'f1_score': metrics['f1_score'],
            'precision': metrics['precision'],
            'recall': metrics['recall']
        })
        
        print(f"  {min_val}-{max_val}: {segment_size}件, {segment_intent_rate:.1%}, Lift:{lift:.2f}, F1:{metrics['f1_score']:.3f}")
    
    # 収入変動額の詳細境界分析
    print("\n【収入変動額の境界分析】")
    variation_boundaries = []
    variation_ranges = [
        (10, 40), (15, 45), (20, 50), (25, 55), (30, 60),
        (15, 50), (20, 55), (25, 60), (20, 40), (25, 45),
        (30, 50), (35, 65), (10, 50), (15, 60), (20, 70)
    ]
    
    for min_val, max_val in variation_ranges:
        mask = (data['収入変動額'] >= min_val) & (data['収入変動額'] <= max_val)
        segment_size = mask.sum()
        segment_intent_rate = data[mask][target_column].mean() if segment_size > 0 else 0
        lift = segment_intent_rate / overall_intent_rate if segment_intent_rate > 0 else 0
        
        metrics = calculate_metrics(mask, data, target_column)
        
        variation_boundaries.append({
            'range': f"{min_val}-{max_val}",
            'min': min_val, 'max': max_val,
            'segment_size': segment_size,
            'intent_rate': segment_intent_rate,
            'lift': lift,
            'f1_score': metrics['f1_score'],
            'precision': metrics['precision'],
            'recall': metrics['recall']
        })
        
        print(f"  {min_val}-{max_val}: {segment_size}件, {segment_intent_rate:.1%}, Lift:{lift:.2f}, F1:{metrics['f1_score']:.3f}")
    
    # 支出変動率の詳細境界分析
    print("\n【支出変動率の境界分析】")
    expenditure_boundaries = []
    expenditure_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
    
    for threshold in expenditure_thresholds:
        mask = data['支出変動率'] >= threshold
        segment_size = mask.sum()
        segment_intent_rate = data[mask][target_column].mean() if segment_size > 0 else 0
        lift = segment_intent_rate / overall_intent_rate if segment_intent_rate > 0 else 0
        
        metrics = calculate_metrics(mask, data, target_column)
        
        expenditure_boundaries.append({
            'threshold': threshold,
            'segment_size': segment_size,
            'intent_rate': segment_intent_rate,
            'lift': lift,
            'f1_score': metrics['f1_score'],
            'precision': metrics['precision'],
            'recall': metrics['recall']
        })
        
        print(f"  >= {threshold}: {segment_size}件, {segment_intent_rate:.1%}, Lift:{lift:.2f}, F1:{metrics['f1_score']:.3f}")
    
    # 最適境界値の特定
    print("\n=== 最適境界値の特定 ===")
    
    # F1スコア最大の条件を特定
    best_income = max(income_boundaries, key=lambda x: x['f1_score'])
    best_variation = max(variation_boundaries, key=lambda x: x['f1_score'])
    best_expenditure = max(expenditure_boundaries, key=lambda x: x['f1_score'])
    
    print(f"最適最高月収: {best_income['range']} (F1: {best_income['f1_score']:.3f})")
    print(f"最適収入変動額: {best_variation['range']} (F1: {best_variation['f1_score']:.3f})")
    print(f"最適支出変動率: >= {best_expenditure['threshold']} (F1: {best_expenditure['f1_score']:.3f})")
    
    # 2. 最適境界値での2条件組み合わせ評価
    print("\n=== 2. 最適境界値での2条件組み合わせ評価 ===")
    
    # 最適境界値での条件定義
    optimal_conditions = {
        'income': (data['最高月収'] >= best_income['min']) & (data['最高月収'] <= best_income['max']),
        'variation': (data['収入変動額'] >= best_variation['min']) & (data['収入変動額'] <= best_variation['max']),
        'expenditure': data['支出変動率'] >= best_expenditure['threshold']
    }
    
    # 2条件組み合わせの評価
    combination_results = []
    
    pairs = [('income', 'variation'), ('income', 'expenditure'), ('variation', 'expenditure')]
    pair_names = [
        f"最高月収{best_income['range']} & 収入変動額{best_variation['range']}",
        f"最高月収{best_income['range']} & 支出変動率>={best_expenditure['threshold']}",
        f"収入変動額{best_variation['range']} & 支出変動率>={best_expenditure['threshold']}"
    ]
    
    for (cond1, cond2), pair_name in zip(pairs, pair_names):
        combined_mask = optimal_conditions[cond1] & optimal_conditions[cond2]
        segment_size = combined_mask.sum()
        segment_intent_rate = data[combined_mask][target_column].mean() if segment_size > 0 else 0
        lift = segment_intent_rate / overall_intent_rate if segment_intent_rate > 0 else 0
        
        metrics = calculate_metrics(combined_mask, data, target_column)
        
        combination_results.append({
            'combination': pair_name,
            'segment_size': segment_size,
            'intent_rate': segment_intent_rate,
            'lift': lift,
            'f1_score': metrics['f1_score'],
            'precision': metrics['precision'],
            'recall': metrics['recall']
        })
        
        print(f"\n{pair_name}:")
        print(f"  該当者: {segment_size}件 ({segment_size/len(data)*100:.1f}%)")
        print(f"  利用意向率: {segment_intent_rate:.1%}")
        print(f"  Lift: {lift:.2f}")
        print(f"  F1スコア: {metrics['f1_score']:.3f}")
        print(f"  精度: {metrics['precision']:.3f}")
        print(f"  再現率: {metrics['recall']:.3f}")
    
    # 結果のランキング
    combination_results.sort(key=lambda x: x['f1_score'], reverse=True)
    
    print(f"\n=== 最適2条件組み合わせランキング（F1スコア順） ===")
    for i, result in enumerate(combination_results):
        print(f"{i+1}. {result['combination']}")
        print(f"   F1: {result['f1_score']:.3f}, Lift: {result['lift']:.2f}, 該当者: {result['segment_size']}件")
    
    # 可視化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('詳細境界値分析結果', fontsize=16, fontweight='bold')
    
    # 1. 最高月収境界値効果
    ax1 = axes[0, 0]
    income_df = pd.DataFrame(income_boundaries)
    income_df_sorted = income_df.sort_values('f1_score', ascending=False).head(10)
    bars = ax1.bar(range(len(income_df_sorted)), income_df_sorted['f1_score'], color='lightblue')
    ax1.set_title('最高月収境界値別F1スコア（上位10）')
    ax1.set_ylabel('F1スコア')
    ax1.set_xticks(range(len(income_df_sorted)))
    ax1.set_xticklabels(income_df_sorted['range'], rotation=45, ha='right')
    
    # 数値表示
    for bar, f1 in zip(bars, income_df_sorted['f1_score']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{f1:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. 収入変動額境界値効果
    ax2 = axes[0, 1]
    variation_df = pd.DataFrame(variation_boundaries)
    variation_df_sorted = variation_df.sort_values('f1_score', ascending=False).head(10)
    bars = ax2.bar(range(len(variation_df_sorted)), variation_df_sorted['f1_score'], color='lightgreen')
    ax2.set_title('収入変動額境界値別F1スコア（上位10）')
    ax2.set_ylabel('F1スコア')
    ax2.set_xticks(range(len(variation_df_sorted)))
    ax2.set_xticklabels(variation_df_sorted['range'], rotation=45, ha='right')
    
    for bar, f1 in zip(bars, variation_df_sorted['f1_score']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{f1:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 3. 支出変動率境界値効果
    ax3 = axes[0, 2]
    expenditure_df = pd.DataFrame(expenditure_boundaries)
    ax3.plot(expenditure_df['threshold'], expenditure_df['f1_score'], 'o-', color='orange', linewidth=2)
    ax3.set_title('支出変動率閾値別F1スコア')
    ax3.set_xlabel('支出変動率閾値')
    ax3.set_ylabel('F1スコア')
    ax3.grid(True, alpha=0.3)
    
    # 4. 2条件組み合わせF1比較
    ax4 = axes[1, 0]
    combo_names = [r['combination'].replace(' & ', '\n&\n') for r in combination_results]
    bars = ax4.bar(range(len(combination_results)), [r['f1_score'] for r in combination_results], 
                  color=['red', 'blue', 'green'])
    ax4.set_title('最適2条件組み合わせF1スコア')
    ax4.set_ylabel('F1スコア')
    ax4.set_xticks(range(len(combination_results)))
    ax4.set_xticklabels(combo_names, fontsize=8)
    
    for bar, f1 in zip(bars, [r['f1_score'] for r in combination_results]):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 5. F1 vs Lift 散布図
    ax5 = axes[1, 1]
    f1_scores = [r['f1_score'] for r in combination_results]
    lifts = [r['lift'] for r in combination_results]
    sizes = [r['segment_size']*20 for r in combination_results]
    
    scatter = ax5.scatter(f1_scores, lifts, s=sizes, alpha=0.6, c=['red', 'blue', 'green'])
    ax5.set_xlabel('F1スコア')
    ax5.set_ylabel('Lift')
    ax5.set_title('F1スコア vs Lift (バブルサイズ=サンプル数)')
    
    for i, (f1, lift) in enumerate(zip(f1_scores, lifts)):
        ax5.annotate(f'{i+1}', (f1, lift), xytext=(5, 5), textcoords='offset points')
    
    # 6. サンプル数比較
    ax6 = axes[1, 2]
    sample_sizes = [r['segment_size'] for r in combination_results]
    bars = ax6.bar(range(len(combination_results)), sample_sizes, color=['red', 'blue', 'green'])
    ax6.set_title('2条件組み合わせ該当者数')
    ax6.set_ylabel('該当者数')
    ax6.set_xticks(range(len(combination_results)))
    ax6.set_xticklabels([f'組み合わせ{i+1}' for i in range(len(combination_results))])
    
    for bar, size in zip(bars, sample_sizes):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{size}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存
    output_file = output_path / "detailed_boundary_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n可視化結果保存: {output_file}")
    
    # レポート保存
    report_file = output_path / "detailed_analysis_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 詳細境界値分析レポート\n\n")
        
        f.write("## 最適個別境界値\n")
        f.write(f"- **最高月収**: {best_income['range']}万円 (F1: {best_income['f1_score']:.3f})\n")
        f.write(f"- **収入変動額**: {best_variation['range']}万円 (F1: {best_variation['f1_score']:.3f})\n")
        f.write(f"- **支出変動率**: >= {best_expenditure['threshold']} (F1: {best_expenditure['f1_score']:.3f})\n\n")
        
        f.write("## 最適2条件組み合わせ\n")
        for i, result in enumerate(combination_results):
            f.write(f"{i+1}. **{result['combination']}**\n")
            f.write(f"   - F1スコア: {result['f1_score']:.3f}\n")
            f.write(f"   - Lift: {result['lift']:.2f}\n")
            f.write(f"   - 該当者: {result['segment_size']}件\n")
            f.write(f"   - 利用意向率: {result['intent_rate']:.1%}\n\n")
    
    print(f"レポート保存: {report_file}")
    
    return {
        'best_income': best_income,
        'best_variation': best_variation,
        'best_expenditure': best_expenditure,
        'combination_results': combination_results
    }

if __name__ == "__main__":
    import sys
    
    data_file = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    detailed_boundary_analysis(data_file, output_dir)