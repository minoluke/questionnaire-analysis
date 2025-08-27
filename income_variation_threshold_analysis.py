"""
収入変動額20万円を境界とした利用意向の違い分析
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

def analyze_income_variation_threshold(data_file=None, output_dir=None):
    """収入変動額20万円を境界とした利用意向の違い分析"""
    
    # デフォルト設定
    if data_file is None:
        data_file = "data/processed/preprocessed_100data.csv"
    if output_dir is None:
        output_dir = "output/income_variation_threshold_analysis"
    
    # データの読み込み
    data = pd.read_csv(data_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    target_column = '利用意向'
    overall_intent_rate = data[target_column].mean()
    
    print("=== 収入変動額20万円境界分析 ===")
    print(f"データ: {data_file}")
    print(f"サンプル数: {len(data)}件")
    print(f"全体利用意向率: {overall_intent_rate:.1%}")
    
    # 条件設定
    below_20 = data['収入変動額'] < 20
    above_ge_20 = data['収入変動額'] >= 20
    
    # 各グループの該当者数
    below_count = below_20.sum()
    above_count = above_ge_20.sum()
    
    print(f"\n=== グループ分析 ===")
    print(f"収入変動額 < 20万円: {below_count}件 ({below_count/len(data)*100:.1f}%)")
    print(f"収入変動額 ≥ 20万円: {above_count}件 ({above_count/len(data)*100:.1f}%)")
    
    # 利用意向率の計算
    below_intent_rate = data[below_20][target_column].mean() if below_count > 0 else 0
    above_intent_rate = data[above_ge_20][target_column].mean() if above_count > 0 else 0
    
    print(f"\n=== 利用意向率 ===")
    print(f"収入変動額 < 20万円: {below_intent_rate:.1%}")
    print(f"収入変動額 ≥ 20万円: {above_intent_rate:.1%}")
    print(f"差: {above_intent_rate - below_intent_rate:.1%}")
    print(f"全体: {overall_intent_rate:.1%}")
    
    # Lift計算
    below_lift = below_intent_rate / overall_intent_rate if overall_intent_rate > 0 else 0
    above_lift = above_intent_rate / overall_intent_rate if overall_intent_rate > 0 else 0
    
    print(f"\n=== Lift ===")
    print(f"収入変動額 < 20万円: {below_lift:.2f}")
    print(f"収入変動額 ≥ 20万円: {above_lift:.2f}")
    
    # 信頼区間
    if below_count > 0:
        below_successes = data[below_20][target_column].sum()
        below_ci_low, below_ci_high = proportion_confint(below_successes, below_count, method='wilson')
    
    if above_count > 0:
        above_successes = data[above_ge_20][target_column].sum()
        above_ci_low, above_ci_high = proportion_confint(above_successes, above_count, method='wilson')
    
    print(f"\n=== 信頼区間（95%） ===")
    print(f"収入変動額 < 20万円: [{below_ci_low:.1%}, {below_ci_high:.1%}]")
    print(f"収入変動額 ≥ 20万円: [{above_ci_low:.1%}, {above_ci_high:.1%}]")
    
    # カイ二乗検定
    if below_count > 0 and above_count > 0:
        below_intent = data[below_20][target_column].sum()
        below_no_intent = below_count - below_intent
        
        above_intent = data[above_ge_20][target_column].sum()
        above_no_intent = above_count - above_intent
        
        contingency_table = np.array([
            [below_intent, below_no_intent],
            [above_intent, above_no_intent]
        ])
        
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        print(f"\n=== カイ二乗検定 ===")
        print(f"分割表:")
        print(f"                利用意向あり  利用意向なし")
        print(f"収入変動額<20万円     {below_intent:3d}        {below_no_intent:3d}")
        print(f"収入変動額≥20万円     {above_intent:3d}        {above_no_intent:3d}")
        print(f"カイ二乗値: {chi2:.4f}")
        print(f"p値: {p_value:.4f}")
        if p_value < 0.05:
            print("結果: 統計的に有意な差あり (p < 0.05)")
        else:
            print("結果: 統計的に有意な差なし (p >= 0.05)")
    
    # t検定も実施
    if below_count > 0 and above_count > 0:
        below_data = data[below_20][target_column]
        above_data = data[above_ge_20][target_column]
        
        t_stat, t_p_value = stats.ttest_ind(below_data, above_data)
        
        print(f"\n=== t検定 ===")
        print(f"t統計量: {t_stat:.4f}")
        print(f"p値: {t_p_value:.4f}")
        if t_p_value < 0.05:
            print("結果: 統計的に有意な差あり (p < 0.05)")
        else:
            print("結果: 統計的に有意な差なし (p >= 0.05)")
    
    # より詳細な収入変動額別分析
    print(f"\n=== 詳細な収入変動額別分析 ===")
    thresholds = [10, 15, 20, 25, 30, 35]
    threshold_results = []
    
    for threshold in thresholds:
        below_threshold = data['収入変動額'] < threshold
        above_threshold = data['収入変動額'] >= threshold
        
        below_th_count = below_threshold.sum()
        above_th_count = above_threshold.sum()
        
        below_th_rate = data[below_threshold][target_column].mean() if below_th_count > 0 else 0
        above_th_rate = data[above_threshold][target_column].mean() if above_th_count > 0 else 0
        
        diff = above_th_rate - below_th_rate
        
        threshold_results.append({
            'threshold': threshold,
            'below_count': below_th_count,
            'above_count': above_th_count,
            'below_rate': below_th_rate,
            'above_rate': above_th_rate,
            'difference': diff
        })
        
        print(f"閾値{threshold}万円: < {below_th_rate:.1%} ({below_th_count}件) vs ≥ {above_th_rate:.1%} ({above_th_count}件), 差: {diff:.1%}")
    
    # プロット作成
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('収入変動額20万円境界分析結果', fontsize=16, fontweight='bold')
    
    # 1. 利用意向率比較
    ax1 = axes[0, 0]
    categories = ['収入変動額\n< 20万円', '収入変動額\n≥ 20万円', '全体']
    rates = [below_intent_rate, above_intent_rate, overall_intent_rate]
    colors_bar = ['lightblue', 'lightcoral', 'gray']
    
    bars = ax1.bar(categories, rates, color=colors_bar, alpha=0.7)
    ax1.set_ylabel('利用意向率')
    ax1.set_title('収入変動額20万円境界での利用意向率')
    ax1.set_ylim(0, max(rates) * 1.2)
    
    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.1%}', ha='center', va='bottom')
    
    # 2. サンプル数分布
    ax2 = axes[0, 1]
    sizes = [below_count, above_count]
    labels = [f'< 20万円\n{below_count}件', f'≥ 20万円\n{above_count}件']
    colors_pie = ['lightblue', 'lightcoral']
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax2.set_title('サンプル数分布')
    
    # 3. 収入変動額分布（利用意向別）
    ax3 = axes[0, 2]
    intent_0_data = data[data[target_column] == 0]['収入変動額']
    intent_1_data = data[data[target_column] == 1]['収入変動額']
    
    # データの範囲を確認して適切なbinsを設定（外れ値を除外）
    max_variation = min(data['収入変動額'].quantile(0.95), 80)  # 95%点または80の小さい方
    bins = np.linspace(0, max_variation, 15)
    
    ax3.hist(intent_0_data, bins=bins, alpha=0.6, label='利用意向なし', color='blue', density=True)
    ax3.hist(intent_1_data, bins=bins, alpha=0.6, label='利用意向あり', color='red', density=True)
    ax3.axvline(x=20, color='green', linestyle='--', linewidth=2, label='境界値20万円')
    ax3.set_xlabel('収入変動額（万円）')
    ax3.set_ylabel('密度')
    ax3.set_title('収入変動額分布（利用意向別）')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 閾値別利用意向率差
    ax4 = axes[1, 0]
    threshold_df = pd.DataFrame(threshold_results)
    ax4.plot(threshold_df['threshold'], threshold_df['difference'], 'o-', linewidth=2, markersize=8, color='purple')
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax4.axvline(x=20, color='green', linestyle='--', alpha=0.7, label='境界値20万円')
    ax4.set_xlabel('収入変動額閾値（万円）')
    ax4.set_ylabel('利用意向率の差（≥閾値 - <閾値）')
    ax4.set_title('閾値別利用意向率の差')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 各点に数値表示
    for i, row in threshold_df.iterrows():
        ax4.annotate(f'{row["difference"]:.1%}', 
                    (row['threshold'], row['difference']), 
                    xytext=(5, 5), textcoords='offset points')
    
    # 5. 信頼区間比較
    ax5 = axes[1, 1]
    groups = ['< 20万円', '≥ 20万円']
    means = [below_intent_rate, above_intent_rate]
    ci_lows = [below_ci_low, above_ci_low]
    ci_highs = [below_ci_high, above_ci_high]
    
    bars = ax5.bar(groups, means, color=['lightblue', 'lightcoral'], alpha=0.7)
    
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
    
    # 6. 箱ひげ図（利用意向別の収入変動額分布）
    ax6 = axes[1, 2]
    box_data = [
        data[data[target_column] == 0]['収入変動額'].values,
        data[data[target_column] == 1]['収入変動額'].values
    ]
    
    bp = ax6.boxplot(box_data, labels=['利用意向なし', '利用意向あり'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    # 境界線を追加
    ax6.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='境界値20万円')
    
    ax6.set_ylabel('収入変動額（万円）')
    ax6.set_title('利用意向別収入変動額の分布')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_file = output_path / "income_variation_threshold_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n可視化結果保存: {output_file}")
    
    # レポート保存
    report_file = output_path / "income_variation_threshold_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 収入変動額20万円境界分析レポート\n\n")
        
        f.write("## 基本統計\n")
        f.write(f"- **収入変動額 < 20万円**: {below_count}件 ({below_count/len(data)*100:.1f}%), 利用意向率 {below_intent_rate:.1%}\n")
        f.write(f"- **収入変動額 ≥ 20万円**: {above_count}件 ({above_count/len(data)*100:.1f}%), 利用意向率 {above_intent_rate:.1%}\n")
        f.write(f"- **利用意向率の差**: {above_intent_rate - below_intent_rate:.1%}\n\n")
        
        f.write("## 統計的検定結果\n")
        f.write(f"- **カイ二乗値**: {chi2:.4f}\n")
        f.write(f"- **p値**: {p_value:.4f}\n")
        if p_value < 0.05:
            f.write("- **結果**: 統計的に有意な差あり\n\n")
        else:
            f.write("- **結果**: 統計的に有意な差なし\n\n")
        
        f.write("## 信頼区間\n")
        f.write(f"- **収入変動額 < 20万円**: [{below_ci_low:.1%}, {below_ci_high:.1%}]\n")
        f.write(f"- **収入変動額 ≥ 20万円**: [{above_ci_low:.1%}, {above_ci_high:.1%}]\n\n")
        
        f.write("## 閾値別分析\n")
        for result in threshold_results:
            f.write(f"- **閾値{result['threshold']}万円**: 差 {result['difference']:.1%} "
                   f"(< {result['below_rate']:.1%} vs ≥ {result['above_rate']:.1%})\n")
    
    print(f"レポート保存: {report_file}")
    
    return {
        'below_count': below_count,
        'above_count': above_count,
        'below_intent_rate': below_intent_rate,
        'above_intent_rate': above_intent_rate,
        'difference': above_intent_rate - below_intent_rate,
        'p_value': p_value,
        'threshold_results': threshold_results
    }

if __name__ == "__main__":
    import sys
    
    data_file = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    analyze_income_variation_threshold(data_file, output_dir)