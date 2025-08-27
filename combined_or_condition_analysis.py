"""
OR条件での分析とプロット
支出変動率≥0.5 OR 収入変動額≥20万円
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

def analyze_or_conditions(data_file=None, output_dir=None):
    """支出変動率≥0.5 OR 収入変動額≥20万円の詳細分析"""
    
    # デフォルト設定
    if data_file is None:
        data_file = "data/processed/preprocessed_100data.csv"
    if output_dir is None:
        output_dir = "output/combined_or_condition_analysis"
    
    # データの読み込み
    data = pd.read_csv(data_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    target_column = '利用意向'
    overall_intent_rate = data[target_column].mean()
    
    print("=== OR条件分析: 支出変動率≥0.5 OR 収入変動額≥20万円 ===")
    print(f"データ: {data_file}")
    print(f"サンプル数: {len(data)}件")
    print(f"全体利用意向率: {overall_intent_rate:.1%}")
    
    # 条件設定
    expenditure_condition = data['支出変動率'] >= 0.5
    income_condition = data['収入変動額'] >= 20
    or_condition = expenditure_condition | income_condition  # OR条件
    neither_condition = ~or_condition  # どちらも満たさない
    
    print(f"\n=== 条件設定 ===")
    print(f"支出変動率 ≥ 0.5")
    print(f"OR")
    print(f"収入変動額 ≥ 20万円")
    
    # 各条件の該当者数
    expenditure_count = expenditure_condition.sum()
    income_count = income_condition.sum()
    or_count = or_condition.sum()
    neither_count = neither_condition.sum()
    both_count = (expenditure_condition & income_condition).sum()
    
    print(f"\n=== 該当者数分析 ===")
    print(f"支出変動率≥0.5のみ: {expenditure_count}件 ({expenditure_count/len(data)*100:.1f}%)")
    print(f"収入変動額≥20万円のみ: {income_count}件 ({income_count/len(data)*100:.1f}%)")
    print(f"両方満たす: {both_count}件 ({both_count/len(data)*100:.1f}%)")
    print(f"OR条件満たす: {or_count}件 ({or_count/len(data)*100:.1f}%)")
    print(f"どちらも満たさない: {neither_count}件 ({neither_count/len(data)*100:.1f}%)")
    
    # 利用意向率の計算
    expenditure_intent_rate = data[expenditure_condition][target_column].mean() if expenditure_count > 0 else 0
    income_intent_rate = data[income_condition][target_column].mean() if income_count > 0 else 0
    or_intent_rate = data[or_condition][target_column].mean() if or_count > 0 else 0
    neither_intent_rate = data[neither_condition][target_column].mean() if neither_count > 0 else 0
    both_intent_rate = data[expenditure_condition & income_condition][target_column].mean() if both_count > 0 else 0
    
    print(f"\n=== 利用意向率 ===")
    print(f"支出変動率≥0.5: {expenditure_intent_rate:.1%}")
    print(f"収入変動額≥20万円: {income_intent_rate:.1%}")
    print(f"両方満たす: {both_intent_rate:.1%}")
    print(f"OR条件満たす: {or_intent_rate:.1%}")
    print(f"どちらも満たさない: {neither_intent_rate:.1%}")
    print(f"全体: {overall_intent_rate:.1%}")
    
    # Lift計算
    expenditure_lift = expenditure_intent_rate / overall_intent_rate if overall_intent_rate > 0 else 0
    income_lift = income_intent_rate / overall_intent_rate if overall_intent_rate > 0 else 0
    or_lift = or_intent_rate / overall_intent_rate if overall_intent_rate > 0 else 0
    neither_lift = neither_intent_rate / overall_intent_rate if overall_intent_rate > 0 else 0
    both_lift = both_intent_rate / overall_intent_rate if overall_intent_rate > 0 else 0
    
    print(f"\n=== Lift ===")
    print(f"支出変動率≥0.5: {expenditure_lift:.2f}")
    print(f"収入変動額≥20万円: {income_lift:.2f}")
    print(f"両方満たす: {both_lift:.2f}")
    print(f"OR条件満たす: {or_lift:.2f}")
    print(f"どちらも満たさない: {neither_lift:.2f}")
    
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
    
    # OR条件の性能指標
    or_metrics = calculate_metrics(or_condition, data, target_column)
    
    print(f"\n=== OR条件の性能指標 ===")
    print(f"精度: {or_metrics['precision']:.3f}")
    print(f"再現率: {or_metrics['recall']:.3f}")
    print(f"F1スコア: {or_metrics['f1_score']:.3f}")
    print(f"全体精度: {or_metrics['accuracy']:.3f}")
    
    # 信頼区間
    if or_count > 0:
        or_successes = data[or_condition][target_column].sum()
        or_ci_low, or_ci_high = proportion_confint(or_successes, or_count, method='wilson')
        
    if neither_count > 0:
        neither_successes = data[neither_condition][target_column].sum()
        neither_ci_low, neither_ci_high = proportion_confint(neither_successes, neither_count, method='wilson')
    
    print(f"\n=== 信頼区間（95%） ===")
    print(f"OR条件満たす: [{or_ci_low:.1%}, {or_ci_high:.1%}]")
    print(f"どちらも満たさない: [{neither_ci_low:.1%}, {neither_ci_high:.1%}]")
    
    # カイ二乗検定
    if or_count > 0 and neither_count > 0:
        or_intent = data[or_condition][target_column].sum()
        or_no_intent = or_count - or_intent
        
        neither_intent = data[neither_condition][target_column].sum()
        neither_no_intent = neither_count - neither_intent
        
        contingency_table = np.array([
            [or_intent, or_no_intent],
            [neither_intent, neither_no_intent]
        ])
        
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        print(f"\n=== カイ二乗検定 ===")
        print(f"分割表:")
        print(f"                  利用意向あり  利用意向なし")
        print(f"OR条件満たす           {or_intent:3d}        {or_no_intent:3d}")
        print(f"どちらも満たさない      {neither_intent:3d}        {neither_no_intent:3d}")
        print(f"カイ二乗値: {chi2:.4f}")
        print(f"p値: {p_value:.4f}")
        if p_value < 0.05:
            print("結果: 統計的に有意な差あり (p < 0.05)")
        else:
            print("結果: 統計的に有意な差なし (p >= 0.05)")
    
    # プロット作成
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('OR条件分析結果\n支出変動率≥0.5 OR 収入変動額≥20万円', fontsize=16, fontweight='bold')
    
    # 1. 利用意向率比較
    ax1 = axes[0, 0]
    categories = ['支出変動率\n≥0.5', '収入変動額\n≥20万円', 'OR条件\n満たす', 'どちらも\n満たさない', '全体']
    rates = [expenditure_intent_rate, income_intent_rate, or_intent_rate, neither_intent_rate, overall_intent_rate]
    colors_bar = ['lightsalmon', 'lightgreen', 'orange', 'lightgray', 'gray']
    
    bars = ax1.bar(categories, rates, color=colors_bar, alpha=0.7)
    ax1.set_ylabel('利用意向率')
    ax1.set_title('各条件での利用意向率')
    ax1.set_ylim(0, max(rates) * 1.2)
    
    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.1%}', ha='center', va='bottom', rotation=0 if rate < 0.4 else 90)
    
    # 2. ベン図風の該当者分布
    ax2 = axes[0, 1]
    
    # 各領域の計算
    only_expenditure = expenditure_count - both_count
    only_income = income_count - both_count
    
    labels = [f'支出変動率≥0.5のみ\n{only_expenditure}件', 
              f'収入変動額≥20万円のみ\n{only_income}件',
              f'両方満たす\n{both_count}件',
              f'どちらも満たさない\n{neither_count}件']
    sizes = [only_expenditure, only_income, both_count, neither_count]
    colors_pie = ['lightsalmon', 'lightgreen', 'orange', 'lightgray']
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax2.set_title('条件該当者分布')
    
    # 3. 散布図（条件領域表示）
    ax3 = axes[0, 2]
    
    # 全データプロット
    colors = ['blue' if intent == 0 else 'red' for intent in data[target_column]]
    ax3.scatter(data['収入変動額'], data['支出変動率'], c=colors, alpha=0.6, s=50)
    
    # 条件範囲をハイライト
    ax3.axvspan(20, data['収入変動額'].max(), alpha=0.2, color='green', label='収入変動額≥20')
    ax3.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='支出変動率≥0.5')
    
    ax3.set_xlabel('収入変動額（万円）')
    ax3.set_ylabel('支出変動率')
    ax3.set_title('散布図（OR条件範囲表示）')
    ax3.set_xlim(0, min(100, data['収入変動額'].quantile(0.95)))
    
    # 凡例を手動で作成
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='利用意向なし'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='利用意向あり'),
        Line2D([0], [0], color='green', alpha=0.5, linewidth=10, label='収入変動額≥20'),
        Line2D([0], [0], color='orange', linestyle='--', linewidth=2, label='支出変動率≥0.5')
    ]
    ax3.legend(handles=legend_elements, loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Lift比較
    ax4 = axes[1, 0]
    lift_categories = ['支出変動率\n≥0.5', '収入変動額\n≥20万円', '両方満たす', 'OR条件\n満たす', 'どちらも\n満たさない']
    lifts = [expenditure_lift, income_lift, both_lift, or_lift, neither_lift]
    colors_lift = ['lightsalmon', 'lightgreen', 'red', 'orange', 'lightgray']
    
    bars = ax4.bar(lift_categories, lifts, color=colors_lift, alpha=0.7)
    ax4.set_ylabel('Lift')
    ax4.set_title('Lift比較')
    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='ベースライン(1.0)')
    ax4.legend()
    
    for bar, lift in zip(bars, lifts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{lift:.2f}', ha='center', va='bottom')
    
    # 5. 信頼区間比較
    ax5 = axes[1, 1]
    groups = ['OR条件満たす', 'どちらも満たさない']
    means = [or_intent_rate, neither_intent_rate]
    ci_lows = [or_ci_low, neither_ci_low]
    ci_highs = [or_ci_high, neither_ci_high]
    
    bars = ax5.bar(groups, means, color=['orange', 'lightgray'], alpha=0.7)
    
    # エラーバー（信頼区間）
    errors_low = [means[i] - ci_lows[i] for i in range(len(means))]
    errors_high = [ci_highs[i] - means[i] for i in range(len(means))]
    
    ax5.errorbar(groups, means, yerr=[errors_low, errors_high], fmt='none', 
                color='black', capsize=5, capthick=2)
    
    ax5.set_ylabel('利用意向率')
    ax5.set_title('利用意向率と95%信頼区間')
    
    for bar, mean in zip(bars, means):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean:.1%}', ha='center', va='bottom')
    
    # 6. 混同行列（OR条件）
    ax6 = axes[1, 2]
    if or_count > 0:
        confusion_matrix = np.array([
            [or_metrics['tn'], or_metrics['fp']],
            [or_metrics['fn'], or_metrics['tp']]
        ])
        
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['予測:なし', '予測:あり'],
                   yticklabels=['実際:なし', '実際:あり'],
                   ax=ax6)
        ax6.set_title('混同行列（OR条件）')
    else:
        ax6.text(0.5, 0.5, '該当データなし', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('混同行列（OR条件）')
    
    plt.tight_layout()
    
    # 保存
    output_file = output_path / "combined_or_condition_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n可視化結果保存: {output_file}")
    
    # レポート保存
    report_file = output_path / "combined_or_condition_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# OR条件分析レポート: 支出変動率≥0.5 OR 収入変動額≥20万円\n\n")
        
        f.write("## 条件設定\n")
        f.write("- **支出変動率**: x ≥ 0.5\n")
        f.write("- **OR**\n")
        f.write("- **収入変動額**: x ≥ 20万円\n\n")
        
        f.write("## 基本統計\n")
        f.write(f"- **OR条件該当者**: {or_count}件 ({or_count/len(data)*100:.1f}%)\n")
        f.write(f"- **どちらも満たさない**: {neither_count}件 ({neither_count/len(data)*100:.1f}%)\n")
        f.write(f"- **OR条件利用意向率**: {or_intent_rate:.1%}\n")
        f.write(f"- **どちらも満たさない利用意向率**: {neither_intent_rate:.1%}\n")
        f.write(f"- **差**: {or_intent_rate - neither_intent_rate:.1%}\n")
        f.write(f"- **OR条件Lift**: {or_lift:.2f}\n\n")
        
        f.write("## 性能指標\n")
        f.write(f"- **F1スコア**: {or_metrics['f1_score']:.3f}\n")
        f.write(f"- **精度**: {or_metrics['precision']:.3f}\n")
        f.write(f"- **再現率**: {or_metrics['recall']:.3f}\n")
        f.write(f"- **全体精度**: {or_metrics['accuracy']:.3f}\n\n")
        
        f.write("## 詳細分布\n")
        f.write(f"- **支出変動率≥0.5のみ**: {only_expenditure}件\n")
        f.write(f"- **収入変動額≥20万円のみ**: {only_income}件\n")
        f.write(f"- **両方満たす**: {both_count}件\n")
        f.write(f"- **どちらも満たさない**: {neither_count}件\n\n")
        
        if or_count > 0:
            f.write("## 統計的検定\n")
            f.write(f"- **カイ二乗値**: {chi2:.4f}\n")
            f.write(f"- **p値**: {p_value:.4f}\n")
            if p_value < 0.05:
                f.write("- **結果**: 統計的に有意な差あり\n\n")
            else:
                f.write("- **結果**: 統計的に有意な差なし\n\n")
            
            f.write(f"- **OR条件95%信頼区間**: [{or_ci_low:.1%}, {or_ci_high:.1%}]\n")
            f.write(f"- **どちらも満たさない95%信頼区間**: [{neither_ci_low:.1%}, {neither_ci_high:.1%}]\n")
    
    print(f"レポート保存: {report_file}")
    
    return {
        'or_count': or_count,
        'neither_count': neither_count,
        'or_intent_rate': or_intent_rate,
        'neither_intent_rate': neither_intent_rate,
        'or_lift': or_lift,
        'or_metrics': or_metrics,
        'p_value': p_value if 'p_value' in locals() else None
    }

if __name__ == "__main__":
    import sys
    
    data_file = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    analyze_or_conditions(data_file, output_dir)