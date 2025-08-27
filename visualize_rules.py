"""
簡易ルール分析結果の可視化
上位ルールの分類境界と利用意向を散布図で表示
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import japanize_matplotlib

def visualize_rules():
    """上位ルールを可視化"""
    
    # データの読み込み
    data_file = "data/processed/preprocessed_data_real.csv"
    data = pd.read_csv(data_file)
    
    # 上位3ルールの定義（実データ用）
    rules = [
        {
            'name': 'ルール1: 支出変動率 > 0.4 & 収入変動額 > 9万円',
            'feature1': '支出変動率', 'threshold1': 0.4,
            'feature2': '収入変動額', 'threshold2': 9.0,
            'f1': 0.824, 'color': 'red', 'alpha': 0.2
        },
        {
            'name': 'ルール2: 支出変動率 > 0.4 & 最高月収 > 35万円',
            'feature1': '支出変動率', 'threshold1': 0.4,
            'feature2': '最高月収', 'threshold2': 35.0,
            'f1': 0.750, 'color': 'blue', 'alpha': 0.2
        },
        {
            'name': 'ルール3: 最高支出 > 7.2万円 & 収入変動額 > 9万円',
            'feature1': '最高支出', 'threshold1': 72.059,
            'feature2': '収入変動額', 'threshold2': 9.0,
            'f1': 0.727, 'color': 'green', 'alpha': 0.2
        }
    ]
    
    # 図の作成
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('簡易ルール分析結果：上位3ルールの可視化', fontsize=16, fontweight='bold')
    
    for i, rule in enumerate(rules):
        ax = axes[i]
        
        # データの準備
        x_data = data[rule['feature1']]
        y_data = data[rule['feature2']]
        colors = ['red' if intent == 1 else 'blue' for intent in data['利用意向']]
        
        # 散布図の描画
        scatter = ax.scatter(x_data, y_data, c=colors, s=100, alpha=0.7, edgecolors='black')
        
        # ルール領域の描画（右上の矩形領域を塗りつぶし）
        x_max = max(x_data) * 1.1
        y_max = max(y_data) * 1.1
        
        # ルール適用範囲を矩形で表示
        rect = patches.Rectangle(
            (rule['threshold1'], rule['threshold2']),
            x_max - rule['threshold1'], y_max - rule['threshold2'],
            linewidth=2, edgecolor=rule['color'], facecolor=rule['color'], alpha=rule['alpha']
        )
        ax.add_patch(rect)
        
        # 境界線の描画
        ax.axvline(x=rule['threshold1'], color=rule['color'], linestyle='--', linewidth=2, label=f"{rule['feature1']} > {rule['threshold1']}")
        ax.axhline(y=rule['threshold2'], color=rule['color'], linestyle='--', linewidth=2, label=f"{rule['feature2']} > {rule['threshold2']}")
        
        # 軸設定
        ax.set_xlabel(rule['feature1'], fontsize=12, fontweight='bold')
        ax.set_ylabel(rule['feature2'], fontsize=12, fontweight='bold')
        ax.set_title(f"{rule['name']}\n(F1スコア: {rule['f1']:.3f})", fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
        
        # 軸の範囲を調整
        ax.set_xlim(-0.05 * max(x_data), x_max)
        ax.set_ylim(-0.05 * max(y_data), y_max)
        
        # ルール内のサンプル数をテキストで表示
        rule_mask = (data[rule['feature1']] > rule['threshold1']) & (data[rule['feature2']] > rule['threshold2'])
        rule_samples = rule_mask.sum()
        rule_intention_rate = data.loc[rule_mask, '利用意向'].mean() if rule_samples > 0 else 0
        
        ax.text(0.02, 0.98, f"ルール内: {rule_samples}件\n利用意向率: {rule_intention_rate:.1%}", 
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 凡例（全体用）
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='利用意向あり'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='利用意向なし'),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='ルール境界'),
        patches.Patch(color='gray', alpha=0.2, label='ルール適用範囲')
    ]
    
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # 保存
    output_file = "output/visualizations/rule_visualization.png"
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"ルール可視化を保存しました: {output_file}")
    
    # ルール詳細の表示
    print(f"\n=== ルール詳細統計 ===")
    for i, rule in enumerate(rules, 1):
        rule_mask = (data[rule['feature1']] > rule['threshold1']) & (data[rule['feature2']] > rule['threshold2'])
        rule_samples = rule_mask.sum()
        other_samples = len(data) - rule_samples
        
        if rule_samples > 0:
            rule_intention_rate = data.loc[rule_mask, '利用意向'].mean()
            other_intention_rate = data.loc[~rule_mask, '利用意向'].mean()
        else:
            rule_intention_rate = 0
            other_intention_rate = data['利用意向'].mean()
        
        overall_intention_rate = data['利用意向'].mean()
        lift = rule_intention_rate / overall_intention_rate if overall_intention_rate > 0 else 0
        
        print(f"\nルール{i}: {rule['name']}")
        print(f"  ルール内: {rule_samples}件 (利用意向率: {rule_intention_rate:.1%})")
        print(f"  ルール外: {other_samples}件 (利用意向率: {other_intention_rate:.1%})")
        print(f"  Lift: {lift:.2f}倍")
        print(f"  F1スコア: {rule['f1']:.3f}")

if __name__ == "__main__":
    visualize_rules()