"""
マッキンゼースライド用の利用意向率プロット
フォントサイズを大幅に拡大し、プレゼンテーション用に最適化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy import signal
from pathlib import Path
import japanize_matplotlib

# グローバルフォント設定
plt.rcParams['font.size'] = 20
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 26

def moving_average_smooth(x, y, window_size=3):
    """移動平均によるスムージング"""
    if len(y) < window_size:
        return x, y
    
    # パディングして端点を処理
    y_padded = np.pad(y, (window_size//2, window_size//2), mode='edge')
    y_smooth = np.convolve(y_padded, np.ones(window_size)/window_size, mode='valid')
    
    return x, y_smooth

def plot_usage_rate_for_mckinsey():
    # データ読み込み
    data = pd.read_csv("data/processed/preprocessed_100data.csv")
    
    # 対象特徴量（順序変更：支出変動率、収入変動率、自由資産率）
    features = ['支出変動率', '収入変動率', '月収に対する自由資産率']
    
    # 出力ディレクトリの作成
    output_path = Path("output/mckinsey_slides")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 図のサイズ設定（横一列、プレゼン用に高さも調整）
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    for i, feature in enumerate(features):
        ax = axes[i]
        
        # データの準備
        feature_data = data[[feature, '利用意向']].copy()
        feature_data = feature_data.dropna()
        
        # binの作成
        if feature == '収入変動率':
            # 収入変動率は8個のbin
            bins = [-0.5, 0.15, 0.25, 0.35, 0.6, 1.0, 1.5, 3.0, 6.0]
            feature_data['bin'] = pd.cut(feature_data[feature], bins=bins, include_lowest=True)
        else:
            # 他の特徴量は8個のbin
            n_bins = 8
            feature_data['bin'] = pd.cut(feature_data[feature], bins=n_bins, include_lowest=True)
        
        # 各binでの利用意向率を計算
        bin_stats = feature_data.groupby('bin', observed=True).agg({
            '利用意向': ['count', 'sum', 'mean']
        }).reset_index()
        
        # 列名を平坦化
        bin_stats.columns = ['bin', 'count', 'sum', 'usage_rate']
        
        # binの中央値を取得
        bin_stats['bin_center'] = bin_stats['bin'].apply(lambda x: x.mid)
        
        # サンプル数が少ないbinは除外（3個未満）
        bin_stats = bin_stats[bin_stats['count'] >= 3].copy()
        
        if len(bin_stats) < 4:
            # データが少ない場合は散布図のみ
            ax.scatter(bin_stats['bin_center'], bin_stats['usage_rate'], 
                      s=bin_stats['count']*20, alpha=0.7, color='blue')
            ax.plot(bin_stats['bin_center'], bin_stats['usage_rate'], 
                   color='blue', linewidth=4, alpha=0.8)
        else:
            # なめらかな曲線を作成
            x = np.array(bin_stats['bin_center'])
            y = np.array(bin_stats['usage_rate'])
            
            # 元のデータ点（サイズを大きく）
            ax.scatter(x, y, s=bin_stats['count']*15, alpha=0.7, color='red', 
                      edgecolors='darkred', linewidth=2, label='実データ', zorder=5)
            
            # 移動平均スムージング（線を太く）
            x_ma, y_ma = moving_average_smooth(x, y, window_size=3)
            ax.plot(x_ma, y_ma, color='blue', linewidth=5, alpha=0.9, 
                   label='利用意向率（移動平均）')
        
        # 軸設定（ラベルサイズ大）
        ax.set_xlabel(feature, fontsize=24, fontweight='bold')
        
        # Y軸ラベルは支出変動率のみ表示
        if i == 0:  # 支出変動率のみ
            ax.set_ylabel('利用意向率', fontsize=24, fontweight='bold')
        else:  # 収入変動率と自由資産率はY軸の数値を非表示
            ax.set_yticklabels([])
        
        # Y軸の範囲を0.1-0.6に統一
        ax.set_ylim(0.1, 0.6)
        
        # グリッド追加（見やすさ向上）
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        
        # 軸の線を太く
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 目盛りを太く
        ax.tick_params(width=2, length=8)
        
        # 凡例（最初のグラフのみ、大きく）
        if i == 0:
            ax.legend(loc='upper left', fontsize=18, frameon=True, 
                     fancybox=True, shadow=True, framealpha=0.9)
    
    plt.tight_layout(pad=2.0)
    
    # 保存（高解像度）
    output_file = output_path / "mckinsey_usage_rate_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()  # plt.show()の代わりにclose()を使用
    
    print(f"マッキンゼースライド用プロットを保存しました: {output_file}")
    
    # 各特徴量の詳細統計も出力
    print("\n=== 各特徴量の詳細統計 ===")
    for feature in features:
        print(f"\n{feature}:")
        feature_data = data[[feature, '利用意向']].dropna()
        correlation = feature_data[feature].corr(feature_data['利用意向'])
        print(f"  相関係数: {correlation:.3f}")
        print(f"  データ数: {len(feature_data)}件")
        
        # 分位数での利用意向率
        quartiles = feature_data[feature].quantile([0.25, 0.5, 0.75])
        print(f"  Q1以下の利用意向率: {feature_data[feature_data[feature] <= quartiles[0.25]]['利用意向'].mean():.1%}")
        print(f"  Q2以下の利用意向率: {feature_data[feature_data[feature] <= quartiles[0.5]]['利用意向'].mean():.1%}")
        print(f"  Q3以下の利用意向率: {feature_data[feature_data[feature] <= quartiles[0.75]]['利用意向'].mean():.1%}")
        print(f"  全体の利用意向率: {feature_data['利用意向'].mean():.1%}")

if __name__ == "__main__":
    plot_usage_rate_for_mckinsey()