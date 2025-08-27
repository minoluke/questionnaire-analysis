"""
仮説条件の最適化スクリプト
3条件→2条件、境界緩和による最適条件の探索
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from pathlib import Path
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

def optimize_hypothesis_conditions(data_file=None, output_dir=None):
    """仮説条件の最適化"""
    
    # デフォルト設定
    if data_file is None:
        data_file = "data/processed/preprocessed_100data.csv"
    if output_dir is None:
        output_dir = "output/hypothesis_optimization"
    
    # データの読み込み
    data = pd.read_csv(data_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    target_column = '利用意向'
    overall_intent_rate = data[target_column].mean()
    
    print("=== 仮説条件最適化分析 ===")
    print(f"データ: {data_file}")
    print(f"サンプル数: {len(data)}件")
    print(f"全体利用意向率: {overall_intent_rate:.1%}")
    
    # 1. 各条件の個別分析
    print("\n=== 1. 各条件の個別効果分析 ===")
    
    # 元の条件
    original_conditions = {
        '最高月収': {'feature': '最高月収', 'min': 50, 'max': 70},
        '収入変動額': {'feature': '収入変動額', 'min': 25, 'max': 60},
        '支出変動率': {'feature': '支出変動率', 'min': 0.5, 'max': float('inf')}
    }
    
    individual_results = []
    
    for name, condition in original_conditions.items():
        feature = condition['feature']
        min_val, max_val = condition['min'], condition['max']
        
        if max_val == float('inf'):
            mask = data[feature] >= min_val
            condition_str = f"{feature} >= {min_val}"
        else:
            mask = (data[feature] >= min_val) & (data[feature] <= max_val)
            condition_str = f"{min_val} <= {feature} <= {max_val}"
        
        segment_size = mask.sum()
        segment_intent_rate = data[mask][target_column].mean() if segment_size > 0 else 0
        lift = segment_intent_rate / overall_intent_rate if segment_intent_rate > 0 else 0
        
        individual_results.append({
            'condition': name,
            'condition_str': condition_str,
            'segment_size': segment_size,
            'intent_rate': segment_intent_rate,
            'lift': lift
        })
        
        print(f"{name}: {condition_str}")
        print(f"  該当者: {segment_size}件 ({segment_size/len(data)*100:.1f}%)")
        print(f"  利用意向率: {segment_intent_rate:.1%}")
        print(f"  Lift: {lift:.2f}")
        print()
    
    # 2. 2条件組み合わせの評価
    print("=== 2. 2条件組み合わせ評価 ===")
    
    condition_pairs = list(combinations(original_conditions.keys(), 2))
    pair_results = []
    
    for pair in condition_pairs:
        cond1, cond2 = pair
        
        # 条件1
        feature1 = original_conditions[cond1]['feature']
        min1, max1 = original_conditions[cond1]['min'], original_conditions[cond1]['max']
        if max1 == float('inf'):
            mask1 = data[feature1] >= min1
        else:
            mask1 = (data[feature1] >= min1) & (data[feature1] <= max1)
        
        # 条件2
        feature2 = original_conditions[cond2]['feature']
        min2, max2 = original_conditions[cond2]['min'], original_conditions[cond2]['max']
        if max2 == float('inf'):
            mask2 = data[feature2] >= min2
        else:
            mask2 = (data[feature2] >= min2) & (data[feature2] <= max2)
        
        # 組み合わせ
        combined_mask = mask1 & mask2
        segment_size = combined_mask.sum()
        segment_intent_rate = data[combined_mask][target_column].mean() if segment_size > 0 else 0
        lift = segment_intent_rate / overall_intent_rate if segment_intent_rate > 0 else 0
        
        # 性能指標計算
        if segment_size > 0:
            tp = (combined_mask & (data[target_column] == 1)).sum()
            fp = (combined_mask & (data[target_column] == 0)).sum()
            fn = (~combined_mask & (data[target_column] == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = recall = f1_score = 0
        
        pair_results.append({
            'pair': f"{cond1} & {cond2}",
            'segment_size': segment_size,
            'intent_rate': segment_intent_rate,
            'lift': lift,
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall
        })
        
        print(f"{cond1} & {cond2}:")
        print(f"  該当者: {segment_size}件 ({segment_size/len(data)*100:.1f}%)")
        print(f"  利用意向率: {segment_intent_rate:.1%}")
        print(f"  Lift: {lift:.2f}, F1: {f1_score:.3f}")
        print()
    
    # 3. 境界値の段階的緩和
    print("=== 3. 境界値の段階的緩和 ===")
    
    # 最も有望な特徴量での境界緩和テスト
    features_to_optimize = ['最高月収', '収入変動額', '支出変動率']
    
    optimization_results = []
    
    # 最高月収の境界緩和
    print("最高月収の境界緩和:")
    income_ranges = [(40, 80), (45, 75), (50, 70), (30, 90), (35, 85)]
    for min_val, max_val in income_ranges:
        mask = (data['最高月収'] >= min_val) & (data['最高月収'] <= max_val)
        segment_size = mask.sum()
        segment_intent_rate = data[mask][target_column].mean() if segment_size > 0 else 0
        lift = segment_intent_rate / overall_intent_rate if segment_intent_rate > 0 else 0
        
        optimization_results.append({
            'condition': f"最高月収 {min_val}-{max_val}",
            'segment_size': segment_size,
            'intent_rate': segment_intent_rate,
            'lift': lift,
            'type': '最高月収'
        })
        
        print(f"  {min_val}-{max_val}: {segment_size}件, {segment_intent_rate:.1%}, Lift:{lift:.2f}")
    
    # 収入変動額の境界緩和
    print("\n収入変動額の境界緩和:")
    variation_ranges = [(20, 70), (15, 80), (25, 60), (10, 100), (30, 50)]
    for min_val, max_val in variation_ranges:
        mask = (data['収入変動額'] >= min_val) & (data['収入変動額'] <= max_val)
        segment_size = mask.sum()
        segment_intent_rate = data[mask][target_column].mean() if segment_size > 0 else 0
        lift = segment_intent_rate / overall_intent_rate if segment_intent_rate > 0 else 0
        
        optimization_results.append({
            'condition': f"収入変動額 {min_val}-{max_val}",
            'segment_size': segment_size,
            'intent_rate': segment_intent_rate,
            'lift': lift,
            'type': '収入変動額'
        })
        
        print(f"  {min_val}-{max_val}: {segment_size}件, {segment_intent_rate:.1%}, Lift:{lift:.2f}")
    
    # 支出変動率の境界緩和
    print("\n支出変動率の境界緩和:")
    expenditure_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for threshold in expenditure_thresholds:
        mask = data['支出変動率'] >= threshold
        segment_size = mask.sum()
        segment_intent_rate = data[mask][target_column].mean() if segment_size > 0 else 0
        lift = segment_intent_rate / overall_intent_rate if segment_intent_rate > 0 else 0
        
        optimization_results.append({
            'condition': f"支出変動率 >= {threshold}",
            'segment_size': segment_size,
            'intent_rate': segment_intent_rate,
            'lift': lift,
            'type': '支出変動率'
        })
        
        print(f"  >= {threshold}: {segment_size}件, {segment_intent_rate:.1%}, Lift:{lift:.2f}")
    
    # 4. 最適2条件組み合わせの探索
    print("\n=== 4. 最適2条件組み合わせ探索 ===")
    
    best_combinations = []
    
    # 緩和された条件での2条件組み合わせ
    relaxed_conditions = {
        '最高月収_緩和': (data['最高月収'] >= 40) & (data['最高月収'] <= 80),
        '収入変動額_緩和': (data['収入変動額'] >= 15) & (data['収入変動額'] <= 80), 
        '支出変動率_緩和': data['支出変動率'] >= 0.4
    }
    
    for cond1, cond2 in combinations(relaxed_conditions.keys(), 2):
        mask1 = relaxed_conditions[cond1]
        mask2 = relaxed_conditions[cond2]
        combined_mask = mask1 & mask2
        
        segment_size = combined_mask.sum()
        if segment_size >= 10:  # 最小サンプル数制約
            segment_intent_rate = data[combined_mask][target_column].mean()
            lift = segment_intent_rate / overall_intent_rate
            
            # 性能指標
            tp = (combined_mask & (data[target_column] == 1)).sum()
            fp = (combined_mask & (data[target_column] == 0)).sum()
            fn = (~combined_mask & (data[target_column] == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            best_combinations.append({
                'combination': f"{cond1} & {cond2}",
                'segment_size': segment_size,
                'intent_rate': segment_intent_rate,
                'lift': lift,
                'f1_score': f1_score,
                'precision': precision,
                'recall': recall
            })
            
            print(f"{cond1} & {cond2}:")
            print(f"  該当者: {segment_size}件")
            print(f"  利用意向率: {segment_intent_rate:.1%}")
            print(f"  Lift: {lift:.2f}, F1: {f1_score:.3f}")
    
    # 結果のソートと表示
    best_combinations.sort(key=lambda x: x['f1_score'], reverse=True)
    
    print("\n=== 最適条件ランキング（F1スコア順） ===")
    for i, result in enumerate(best_combinations[:5]):
        print(f"{i+1}. {result['combination']}")
        print(f"   該当者: {result['segment_size']}件")
        print(f"   利用意向率: {result['intent_rate']:.1%}")
        print(f"   F1: {result['f1_score']:.3f}, Lift: {result['lift']:.2f}")
        print()
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('仮説条件最適化結果', fontsize=16, fontweight='bold')
    
    # 1. 個別条件効果
    ax1 = axes[0, 0]
    individual_df = pd.DataFrame(individual_results)
    bars = ax1.bar(individual_df['condition'], individual_df['lift'], color='skyblue', alpha=0.7)
    ax1.set_title('個別条件のLift')
    ax1.set_ylabel('Lift')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, lift in zip(bars, individual_df['lift']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{lift:.2f}', ha='center', va='bottom')
    
    # 2. 2条件組み合わせ効果
    ax2 = axes[0, 1]
    pair_df = pd.DataFrame(pair_results)
    bars = ax2.bar(range(len(pair_df)), pair_df['f1_score'], color='lightcoral', alpha=0.7)
    ax2.set_title('2条件組み合わせのF1スコア')
    ax2.set_ylabel('F1スコア')
    ax2.set_xticks(range(len(pair_df)))
    ax2.set_xticklabels([p.replace(' & ', '\n&\n') for p in pair_df['pair']], fontsize=8)
    
    # 3. 境界値緩和効果（最高月収）
    ax3 = axes[1, 0]
    income_opts = [r for r in optimization_results if r['type'] == '最高月収']
    income_df = pd.DataFrame(income_opts)
    ax3.plot(range(len(income_df)), income_df['lift'], 'o-', color='green', linewidth=2)
    ax3.set_title('最高月収境界値緩和効果')
    ax3.set_ylabel('Lift')
    ax3.set_xticks(range(len(income_df)))
    ax3.set_xticklabels(income_df['condition'].str.replace('最高月収 ', ''), rotation=45)
    
    # 4. 最適組み合わせ比較
    ax4 = axes[1, 1]
    if best_combinations:
        top5 = best_combinations[:5]
        top5_df = pd.DataFrame(top5)
        
        # F1スコアとLiftの散布図
        scatter = ax4.scatter(top5_df['f1_score'], top5_df['lift'], 
                            s=top5_df['segment_size']*5, alpha=0.6, c='orange')
        ax4.set_xlabel('F1スコア')
        ax4.set_ylabel('Lift')
        ax4.set_title('最適組み合わせ (バブルサイズ=サンプル数)')
        
        # 各点にラベル
        for i, (f1, lift, size) in enumerate(zip(top5_df['f1_score'], top5_df['lift'], top5_df['segment_size'])):
            ax4.annotate(f'{i+1}', (f1, lift), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    
    # 保存
    output_file = output_path / "hypothesis_optimization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n可視化結果保存: {output_file}")
    
    # レポート保存
    report_file = output_path / "optimization_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 仮説条件最適化レポート\n\n")
        
        f.write("## 個別条件効果\n")
        for result in individual_results:
            f.write(f"- **{result['condition']}**: {result['segment_size']}件, {result['intent_rate']:.1%}, Lift {result['lift']:.2f}\n")
        
        f.write("\n## 最適2条件組み合わせ（上位5位）\n")
        for i, result in enumerate(best_combinations[:5]):
            f.write(f"{i+1}. **{result['combination']}**\n")
            f.write(f"   - 該当者: {result['segment_size']}件\n")
            f.write(f"   - 利用意向率: {result['intent_rate']:.1%}\n")
            f.write(f"   - F1スコア: {result['f1_score']:.3f}\n")
            f.write(f"   - Lift: {result['lift']:.2f}\n\n")
    
    print(f"レポート保存: {report_file}")
    
    return {
        'individual_results': individual_results,
        'pair_results': pair_results,
        'best_combinations': best_combinations,
        'optimization_results': optimization_results
    }

if __name__ == "__main__":
    import sys
    
    data_file = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    optimize_hypothesis_conditions(data_file, output_dir)