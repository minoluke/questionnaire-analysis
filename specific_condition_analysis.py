"""
特定条件での分析とプロット
最高月収: 45-75万円 & 収入変動額: 25-60万円
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from pathlib import Path
from statsmodels.stats.proportion import proportion_confint
import warnings
warnings.filterwarnings('ignore')

def analyze_specific_conditions(data_file=None, output_dir=None):
    """特定条件での詳細分析とプロット"""
    
    # デフォルト設定
    if data_file is None:
        data_file = "data/processed/preprocessed_100data.csv"
    if output_dir is None:
        output_dir = "output/specific_condition_analysis"
    
    # データの読み込み
    data = pd.read_csv(data_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    target_column = '利用意向'
    overall_intent_rate = data[target_column].mean()
    
    print("=== 特定条件分析 ===")
    print(f"データ: {data_file}")
    print(f"サンプル数: {len(data)}件")
    print(f"全体利用意向率: {overall_intent_rate:.1%}")
    
    # 条件設定
    income_condition = (data['最高月収'] >= 45) & (data['最高月収'] <= 75)
    variation_condition = (data['収入変動額'] >= 25) & (data['収入変動額'] <= 60)
    combined_condition = income_condition & variation_condition
    
    print(f"\n=== 条件設定 ===")
    print(f"最高月収: 45 ≤ x ≤ 75万円")
    print(f"収入変動額: 25 ≤ x ≤ 60万円")
    
    # 各条件の該当者数
    income_count = income_condition.sum()
    variation_count = variation_condition.sum()
    combined_count = combined_condition.sum()
    
    print(f"\n=== 該当者数 ===")
    print(f"最高月収条件: {income_count}件 ({income_count/len(data)*100:.1f}%)")
    print(f"収入変動額条件: {variation_count}件 ({variation_count/len(data)*100:.1f}%)")
    print(f"両条件満たす: {combined_count}件 ({combined_count/len(data)*100:.1f}%)")
    
    # 利用意向率の計算
    income_intent_rate = data[income_condition][target_column].mean() if income_count > 0 else 0
    variation_intent_rate = data[variation_condition][target_column].mean() if variation_count > 0 else 0
    combined_intent_rate = data[combined_condition][target_column].mean() if combined_count > 0 else 0
    
    print(f"\n=== 利用意向率 ===")
    print(f"最高月収条件: {income_intent_rate:.1%}")
    print(f"収入変動額条件: {variation_intent_rate:.1%}")
    print(f"両条件満たす: {combined_intent_rate:.1%}")
    print(f"全体: {overall_intent_rate:.1%}")
    
    # Lift計算
    income_lift = income_intent_rate / overall_intent_rate if overall_intent_rate > 0 else 0
    variation_lift = variation_intent_rate / overall_intent_rate if overall_intent_rate > 0 else 0
    combined_lift = combined_intent_rate / overall_intent_rate if overall_intent_rate > 0 else 0
    
    print(f"\n=== Lift ===")
    print(f"最高月収条件: {income_lift:.2f}")
    print(f"収入変動額条件: {variation_lift:.2f}")
    print(f"両条件満たす: {combined_lift:.2f}")
    
    # 性能指標計算関数
    def calculate_metrics(mask, data, target_column):
        if mask.sum() == 0:
            return {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'accuracy': 0}
        
        tp = (mask & (data[target_column] == 1)).sum()
        fp = (mask & (data[target_column] == 0)).sum()
        fn = (~mask & (data[target_column] == 1)).sum()
        tn = (~mask & (data[target_column] == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(data)
        
        return {
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': precision, 'recall': recall, 'f1_score': f1_score, 'accuracy': accuracy
        }
    
    # 各条件の性能指標
    income_metrics = calculate_metrics(income_condition, data, target_column)
    variation_metrics = calculate_metrics(variation_condition, data, target_column)
    combined_metrics = calculate_metrics(combined_condition, data, target_column)
    
    print(f"\n=== 性能指標 ===")
    print(f"最高月収条件:")
    print(f"  精度: {income_metrics['precision']:.3f}")
    print(f"  再現率: {income_metrics['recall']:.3f}")
    print(f"  F1スコア: {income_metrics['f1_score']:.3f}")
    print(f"  全体精度: {income_metrics['accuracy']:.3f}")
    
    print(f"収入変動額条件:")
    print(f"  精度: {variation_metrics['precision']:.3f}")
    print(f"  再現率: {variation_metrics['recall']:.3f}")
    print(f"  F1スコア: {variation_metrics['f1_score']:.3f}")
    print(f"  全体精度: {variation_metrics['accuracy']:.3f}")
    
    print(f"両条件満たす:")
    print(f"  精度: {combined_metrics['precision']:.3f}")
    print(f"  再現率: {combined_metrics['recall']:.3f}")
    print(f"  F1スコア: {combined_metrics['f1_score']:.3f}")
    print(f"  全体精度: {combined_metrics['accuracy']:.3f}")
    
    # 信頼区間
    if combined_count > 0:
        combined_successes = data[combined_condition][target_column].sum()
        ci_low, ci_high = proportion_confint(combined_successes, combined_count, method='wilson')
        print(f"\n=== 信頼区間（95%） ===")
        print(f"両条件満たす利用意向率: [{ci_low:.1%}, {ci_high:.1%}]")
    
    # カイ二乗検定
    if combined_count > 0 and len(data) - combined_count > 0:
        combined_intent = data[combined_condition][target_column].sum()
        combined_no_intent = combined_count - combined_intent
        
        non_combined_intent = data[~combined_condition][target_column].sum()
        non_combined_no_intent = (len(data) - combined_count) - non_combined_intent
        
        contingency_table = np.array([
            [combined_intent, combined_no_intent],
            [non_combined_intent, non_combined_no_intent]
        ])
        
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        print(f"\n=== カイ二乗検定 ===")
        print(f"カイ二乗値: {chi2:.4f}")
        print(f"p値: {p_value:.4f}")
        if p_value < 0.05:
            print("結果: 統計的に有意な差あり (p < 0.05)")
        else:
            print("結果: 統計的に有意な差なし (p >= 0.05)")
    
    # プロット作成
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('特定条件分析結果\n最高月収45-75万円 & 収入変動額25-60万円', fontsize=16, fontweight='bold')
    
    # 1. 散布図（条件範囲表示）
    ax1 = axes[0, 0]
    
    # 全データプロット
    colors = ['blue' if intent == 0 else 'red' for intent in data[target_column]]
    ax1.scatter(data['最高月収'], data['収入変動額'], c=colors, alpha=0.6, s=50)
    
    # 条件範囲をハイライト
    rect = plt.Rectangle((45, 25), 30, 35, fill=False, edgecolor='green', linewidth=3, linestyle='--')
    ax1.add_patch(rect)
    
    ax1.set_xlabel('最高月収（万円）')
    ax1.set_ylabel('収入変動額（万円）')
    ax1.set_title('散布図（条件範囲表示）')
    # 凡例を手動で作成
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='利用意向なし'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='利用意向あり'),
        Line2D([0], [0], color='green', linestyle='--', linewidth=3, label='条件範囲')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. 利用意向率比較
    ax2 = axes[0, 1]
    categories = ['最高月収\n45-75', '収入変動額\n25-60', '両条件\n満たす', '全体']
    rates = [income_intent_rate, variation_intent_rate, combined_intent_rate, overall_intent_rate]
    colors_bar = ['lightblue', 'lightgreen', 'orange', 'gray']
    
    bars = ax2.bar(categories, rates, color=colors_bar, alpha=0.7)
    ax2.set_ylabel('利用意向率')
    ax2.set_title('各条件での利用意向率')
    ax2.set_ylim(0, max(rates) * 1.2)
    
    for bar, rate in zip(bars, rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.1%}', ha='center', va='bottom')
    
    # 3. サンプル数分布
    ax3 = axes[0, 2]
    sizes = [combined_count, len(data) - combined_count]
    labels = [f'条件満たす\n{combined_count}件', f'その他\n{len(data) - combined_count}件']
    colors_pie = ['orange', 'lightgray']
    
    ax3.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax3.set_title('サンプル数分布')
    
    # 4. F1スコア比較
    ax4 = axes[1, 0]
    f1_scores = [income_metrics['f1_score'], variation_metrics['f1_score'], combined_metrics['f1_score']]
    f1_categories = ['最高月収\n45-75', '収入変動額\n25-60', '両条件\n満たす']
    
    bars = ax4.bar(f1_categories, f1_scores, color=['lightblue', 'lightgreen', 'orange'], alpha=0.7)
    ax4.set_ylabel('F1スコア')
    ax4.set_title('F1スコア比較')
    ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='最悪ライン(0.5)')
    ax4.legend()
    
    for bar, f1 in zip(bars, f1_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{f1:.3f}', ha='center', va='bottom')
    
    # 5. Lift比較
    ax5 = axes[1, 1]
    lifts = [income_lift, variation_lift, combined_lift]
    lift_categories = ['最高月収\n45-75', '収入変動額\n25-60', '両条件\n満たす']
    
    bars = ax5.bar(lift_categories, lifts, color=['lightblue', 'lightgreen', 'orange'], alpha=0.7)
    ax5.set_ylabel('Lift')
    ax5.set_title('Lift比較')
    ax5.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='ベースライン(1.0)')
    ax5.legend()
    
    for bar, lift in zip(bars, lifts):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{lift:.2f}', ha='center', va='bottom')
    
    # 6. 混同行列（両条件満たす場合）
    ax6 = axes[1, 2]
    if combined_count > 0:
        confusion_matrix = np.array([
            [combined_metrics['tn'], combined_metrics['fp']],
            [combined_metrics['fn'], combined_metrics['tp']]
        ])
        
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['予測:なし', '予測:あり'],
                   yticklabels=['実際:なし', '実際:あり'],
                   ax=ax6)
        ax6.set_title('混同行列（両条件満たす）')
    else:
        ax6.text(0.5, 0.5, '該当データなし', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('混同行列（両条件満たす）')
    
    plt.tight_layout()
    
    # 保存
    output_file = output_path / "specific_condition_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n可視化結果保存: {output_file}")
    
    # レポート保存
    report_file = output_path / "specific_condition_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 特定条件分析レポート\n\n")
        f.write("## 条件設定\n")
        f.write("- **最高月収**: 45 ≤ x ≤ 75万円\n")
        f.write("- **収入変動額**: 25 ≤ x ≤ 60万円\n\n")
        
        f.write("## 基本統計\n")
        f.write(f"- **両条件該当者**: {combined_count}件 ({combined_count/len(data)*100:.1f}%)\n")
        f.write(f"- **利用意向率**: {combined_intent_rate:.1%}\n")
        f.write(f"- **全体利用意向率**: {overall_intent_rate:.1%}\n")
        f.write(f"- **Lift**: {combined_lift:.2f}\n\n")
        
        f.write("## 性能指標\n")
        f.write(f"- **F1スコア**: {combined_metrics['f1_score']:.3f}\n")
        f.write(f"- **精度**: {combined_metrics['precision']:.3f}\n")
        f.write(f"- **再現率**: {combined_metrics['recall']:.3f}\n")
        f.write(f"- **全体精度**: {combined_metrics['accuracy']:.3f}\n\n")
        
        if combined_count > 0:
            f.write("## 統計的検定\n")
            f.write(f"- **カイ二乗値**: {chi2:.4f}\n")
            f.write(f"- **p値**: {p_value:.4f}\n")
            if p_value < 0.05:
                f.write("- **結果**: 統計的に有意な差あり\n\n")
            else:
                f.write("- **結果**: 統計的に有意な差なし\n\n")
            
            f.write(f"- **95%信頼区間**: [{ci_low:.1%}, {ci_high:.1%}]\n")
    
    print(f"レポート保存: {report_file}")
    
    return {
        'combined_count': combined_count,
        'combined_intent_rate': combined_intent_rate,
        'combined_lift': combined_lift,
        'combined_metrics': combined_metrics,
        'p_value': p_value if 'p_value' in locals() else None
    }

if __name__ == "__main__":
    import sys
    
    data_file = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    analyze_specific_conditions(data_file, output_dir)