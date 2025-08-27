"""
仮説検証スクリプト
仮説: 最高月収50-70万円、収入変動額25-60万円、支出変動率0.5以上のセグメントで利用意向が高い
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from pathlib import Path

def validate_hypothesis(data_file=None, output_dir=None):
    """仮説検証の実行"""
    
    # デフォルト設定
    if data_file is None:
        data_file = "data/processed/preprocessed_100data.csv"
    if output_dir is None:
        output_dir = "output/hypothesis_validation"
    
    # データの読み込み
    data = pd.read_csv(data_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=== 仮説検証分析 ===")
    print(f"データ: {data_file}")
    print(f"サンプル数: {len(data)}件")
    
    # 仮説の条件定義
    hypothesis_conditions = {
        '最高月収': (50, 70),      # 50-70万円
        '収入変動額': (25, 60),    # 25-60万円
        '支出変動率': (0.5, float('inf'))  # 0.5以上
    }
    
    target_column = '利用意向'
    
    # A. 基本統計分析
    print("\n=== A. 基本統計分析 ===")
    
    # 1. セグメント該当者の特定
    segment_mask = pd.Series([True] * len(data))
    
    for feature, (min_val, max_val) in hypothesis_conditions.items():
        if max_val == float('inf'):
            condition = data[feature] >= min_val
            print(f"{feature} >= {min_val}: {condition.sum()}件該当")
        else:
            condition = (data[feature] >= min_val) & (data[feature] <= max_val)
            print(f"{min_val} <= {feature} <= {max_val}: {condition.sum()}件該当")
        
        segment_mask = segment_mask & condition
    
    # セグメント分析
    segment_data = data[segment_mask]
    non_segment_data = data[~segment_mask]
    
    segment_size = len(segment_data)
    segment_intent_rate = segment_data[target_column].mean() if segment_size > 0 else 0
    
    non_segment_size = len(non_segment_data)
    non_segment_intent_rate = non_segment_data[target_column].mean() if non_segment_size > 0 else 0
    
    overall_intent_rate = data[target_column].mean()
    
    print(f"\n--- セグメント分析結果 ---")
    print(f"仮説セグメント該当者: {segment_size}件")
    print(f"仮説セグメント利用意向率: {segment_intent_rate:.1%}")
    print(f"非該当者: {non_segment_size}件") 
    print(f"非該当者利用意向率: {non_segment_intent_rate:.1%}")
    print(f"全体利用意向率: {overall_intent_rate:.1%}")
    
    if segment_size > 0:
        lift = segment_intent_rate / overall_intent_rate
        print(f"Lift: {lift:.2f}")
    
    # 2. カイ二乗検定
    if segment_size > 0 and non_segment_size > 0:
        # 分割表の作成
        segment_intent = segment_data[target_column].sum()
        segment_no_intent = segment_size - segment_intent
        
        non_segment_intent = non_segment_data[target_column].sum()
        non_segment_no_intent = non_segment_size - non_segment_intent
        
        contingency_table = np.array([
            [segment_intent, segment_no_intent],
            [non_segment_intent, non_segment_no_intent]
        ])
        
        print(f"\n--- 分割表 ---")
        print(f"                利用意向あり  利用意向なし")
        print(f"仮説セグメント        {segment_intent:3d}        {segment_no_intent:3d}")
        print(f"非該当者             {non_segment_intent:3d}        {non_segment_no_intent:3d}")
        
        # カイ二乗検定
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        print(f"\n--- カイ二乗検定結果 ---")
        print(f"カイ二乗値: {chi2:.4f}")
        print(f"p値: {p_value:.4f}")
        print(f"自由度: {dof}")
        
        if p_value < 0.05:
            print("結果: 統計的に有意な差あり (p < 0.05)")
        else:
            print("結果: 統計的に有意な差なし (p >= 0.05)")
    
    # 3. 信頼区間の算出
    if segment_size > 0:
        # 二項分布の信頼区間（Wilson score interval）
        from statsmodels.stats.proportion import proportion_confint
        
        segment_successes = segment_data[target_column].sum()
        ci_low, ci_high = proportion_confint(segment_successes, segment_size, method='wilson')
        
        print(f"\n--- 信頼区間（95%） ---")
        print(f"仮説セグメント利用意向率の95%信頼区間: [{ci_low:.1%}, {ci_high:.1%}]")
        
        # 全体との比較
        overall_successes = data[target_column].sum()
        overall_ci_low, overall_ci_high = proportion_confint(overall_successes, len(data), method='wilson')
        print(f"全体利用意向率の95%信頼区間: [{overall_ci_low:.1%}, {overall_ci_high:.1%}]")
    
    # B. ルールベース分析
    print("\n=== B. ルールベース分析 ===")
    
    # 1. 既存分析結果での確認
    try:
        rules_file = "output/tables/simple_rules_results.csv"
        rules_df = pd.read_csv(rules_file)
        
        print(f"\n--- 既存ルール分析結果確認 ---")
        print(f"発見されたルール数: {len(rules_df)}個")
        
        # 仮説に関連するルールの検索
        related_rules = []
        for idx, row in rules_df.iterrows():
            condition = row['condition']
            if ('最高月収' in condition or '収入変動額' in condition or '支出変動率' in condition):
                related_rules.append(row)
        
        print(f"仮説関連ルール: {len(related_rules)}個")
        
        if related_rules:
            print("\n上位関連ルール:")
            for i, rule in enumerate(related_rules[:5]):
                print(f"{i+1}. {rule['condition']}")
                print(f"   F1スコア: {rule['f1_score']:.3f}, 利用意向率: {rule['intent_rate_in_rule']:.1%}")
        
    except FileNotFoundError:
        print("既存のルール分析結果が見つかりません")
    
    # 2. 3条件組み合わせルールの直接評価
    print(f"\n--- 3条件組み合わせルール評価 ---")
    
    # 仮説ルールの直接評価
    hypothesis_rule = (
        (data['最高月収'] >= 50) & (data['最高月収'] <= 70) &
        (data['収入変動額'] >= 25) & (data['収入変動額'] <= 60) &
        (data['支出変動率'] >= 0.5)
    )
    
    rule_matches = hypothesis_rule.sum()
    rule_intent_rate = data[hypothesis_rule][target_column].mean() if rule_matches > 0 else 0
    
    # 性能指標の計算
    if rule_matches > 0:
        # 混同行列
        tp = (hypothesis_rule & (data[target_column] == 1)).sum()  # True Positive
        fp = (hypothesis_rule & (data[target_column] == 0)).sum()  # False Positive
        fn = (~hypothesis_rule & (data[target_column] == 1)).sum() # False Negative
        tn = (~hypothesis_rule & (data[target_column] == 0)).sum() # True Negative
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(data)
        
        print(f"ルール該当者: {rule_matches}件")
        print(f"ルール内利用意向率: {rule_intent_rate:.1%}")
        print(f"精度 (Precision): {precision:.3f}")
        print(f"再現率 (Recall): {recall:.3f}")
        print(f"F1スコア: {f1_score:.3f}")
        print(f"全体精度: {accuracy:.3f}")
        print(f"Lift: {rule_intent_rate / overall_intent_rate:.2f}")
    else:
        print("仮説ルールに該当するデータがありません")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('仮説検証結果', fontsize=16, fontweight='bold')
    
    # 1. セグメント別利用意向率
    ax1 = axes[0, 0]
    categories = ['仮説セグメント', '非該当者', '全体']
    rates = [segment_intent_rate, non_segment_intent_rate, overall_intent_rate]
    colors = ['red', 'blue', 'gray']
    bars = ax1.bar(categories, rates, color=colors, alpha=0.7)
    ax1.set_ylabel('利用意向率')
    ax1.set_title('セグメント別利用意向率')
    ax1.set_ylim(0, max(rates) * 1.2)
    
    # 数値表示
    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.1%}', ha='center', va='bottom')
    
    # 2. サンプル数
    ax2 = axes[0, 1]
    sizes = [segment_size, non_segment_size]
    labels = ['仮説セグメント', '非該当者']
    ax2.pie(sizes, labels=labels, autopct=lambda pct: f'{int(pct/100*sum(sizes))}件\n({pct:.1f}%)', startangle=90)
    ax2.set_title('サンプル数分布')
    
    # 3. 特徴量分布（仮説セグメント vs その他）
    ax3 = axes[1, 0]
    features_to_plot = ['最高月収', '収入変動額', '支出変動率']
    
    if segment_size > 0:
        segment_means = [segment_data[f].mean() for f in features_to_plot]
        non_segment_means = [non_segment_data[f].mean() for f in features_to_plot]
        
        x = np.arange(len(features_to_plot))
        width = 0.35
        
        ax3.bar(x - width/2, segment_means, width, label='仮説セグメント', color='red', alpha=0.7)
        ax3.bar(x + width/2, non_segment_means, width, label='非該当者', color='blue', alpha=0.7)
        
        ax3.set_xlabel('特徴量')
        ax3.set_ylabel('平均値')
        ax3.set_title('特徴量平均値比較')
        ax3.set_xticks(x)
        ax3.set_xticklabels(features_to_plot, rotation=45, ha='right')
        ax3.legend()
    
    # 4. 混同行列（ルール適用結果）
    ax4 = axes[1, 1]
    if rule_matches > 0:
        confusion_matrix = np.array([[tp, fp], [fn, tn]])
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['予測:意向なし', '予測:意向あり'],
                   yticklabels=['実際:意向なし', '実際:意向あり'],
                   ax=ax4)
        ax4.set_title('混同行列（仮説ルール）')
    else:
        ax4.text(0.5, 0.5, '該当データなし', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('混同行列（仮説ルール）')
    
    plt.tight_layout()
    
    # 保存
    output_file = output_path / "hypothesis_validation.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n可視化結果保存: {output_file}")
    
    # レポート出力
    report_file = output_path / "hypothesis_validation_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 仮説検証レポート\n\n")
        f.write("## 仮説\n")
        f.write("最高月収50-70万円、収入変動額25-60万円、支出変動率0.5以上のセグメントで利用意向が高い\n\n")
        
        f.write("## 基本統計分析結果\n")
        f.write(f"- **仮説セグメント該当者**: {segment_size}件\n")
        f.write(f"- **仮説セグメント利用意向率**: {segment_intent_rate:.1%}\n")
        f.write(f"- **非該当者利用意向率**: {non_segment_intent_rate:.1%}\n")
        f.write(f"- **全体利用意向率**: {overall_intent_rate:.1%}\n")
        
        if segment_size > 0:
            f.write(f"- **Lift**: {segment_intent_rate / overall_intent_rate:.2f}\n")
        
        if segment_size > 0 and non_segment_size > 0:
            f.write(f"\n## 統計的検定\n")
            f.write(f"- **カイ二乗値**: {chi2:.4f}\n")
            f.write(f"- **p値**: {p_value:.4f}\n")
            if p_value < 0.05:
                f.write("- **結果**: 統計的に有意な差あり\n")
            else:
                f.write("- **結果**: 統計的に有意な差なし\n")
        
        f.write(f"\n## ルール性能\n")
        if rule_matches > 0:
            f.write(f"- **F1スコア**: {f1_score:.3f}\n")
            f.write(f"- **精度**: {precision:.3f}\n")
            f.write(f"- **再現率**: {recall:.3f}\n")
        else:
            f.write("- 該当データなし\n")
    
    print(f"レポート保存: {report_file}")
    
    return {
        'segment_size': segment_size,
        'segment_intent_rate': segment_intent_rate,
        'overall_intent_rate': overall_intent_rate,
        'p_value': p_value if 'p_value' in locals() else None,
        'f1_score': f1_score if 'f1_score' in locals() else None
    }

if __name__ == "__main__":
    import sys
    
    data_file = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    validate_hypothesis(data_file, output_dir)