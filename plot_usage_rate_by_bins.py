"""
特定の特徴量について、各binでの利用意向率をなめらかな線でプロット
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy import signal
from pathlib import Path
import japanize_matplotlib

def moving_average_smooth(x, y, window_size=3):
    """移動平均によるスムージング"""
    if len(y) < window_size:
        return x, y
    
    # パディングして端点を処理
    y_padded = np.pad(y, (window_size//2, window_size//2), mode='edge')
    y_smooth = np.convolve(y_padded, np.ones(window_size)/window_size, mode='valid')
    
    return x, y_smooth

def cosine_window_smooth(x, y, window_size=5):
    """コサインウィンドウによるスムージング"""
    if len(y) < window_size:
        return x, y
    
    # コサインウィンドウ（Hann窓）の作成
    window = signal.windows.hann(window_size)
    window = window / window.sum()  # 正規化
    
    # パディングして端点を処理
    y_padded = np.pad(y, (window_size//2, window_size//2), mode='edge')
    y_smooth = np.convolve(y_padded, window, mode='valid')
    
    return x, y_smooth

def plot_usage_rate_by_bins():
    # データ読み込み
    data = pd.read_csv("data/processed/preprocessed_100data.csv")
    
    # 対象特徴量
    features = ['支出変動率', '収入変動額', '月収に対する自由資産率', '収入変動率']
    
    # 出力ディレクトリの作成
    output_path = Path("output/visualizations")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 図のサイズ設定（横一列）
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    
    for i, feature in enumerate(features):
        ax = axes[i]
        
        # データの準備
        feature_data = data[[feature, '利用意向']].copy()
        feature_data = feature_data.dropna()
        
        # binの作成（7-10個程度に調整）
        if feature == '収入変動額':
            # 収入変動額は7個のbin
            bins = [-10, 3, 6, 10, 18, 30, 50, 200]
            feature_data['bin'] = pd.cut(feature_data[feature], bins=bins, include_lowest=True)
        elif feature == '収入変動率':
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
                      s=bin_stats['count']*10, alpha=0.7, color='blue')
            ax.plot(bin_stats['bin_center'], bin_stats['usage_rate'], 
                   color='blue', linewidth=2, alpha=0.8)
        else:
            # なめらかな曲線を作成
            x = np.array(bin_stats['bin_center'])
            y = np.array(bin_stats['usage_rate'])
            
            # 元のデータ点
            ax.scatter(x, y, s=bin_stats['count']*8, alpha=0.6, color='red', 
                      edgecolors='darkred', linewidth=1, label='実データ点', zorder=5)
            
            # 移動平均スムージング
            x_ma, y_ma = moving_average_smooth(x, y, window_size=3)
            ax.plot(x_ma, y_ma, color='blue', linewidth=3, alpha=0.9, 
                   label='利用意向率（移動平均）')
        
        # 軸設定
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('利用意向率', fontsize=12)
        # Y軸の範囲をデータに応じて調整
        y_min = max(0, min(bin_stats['usage_rate']) - 0.05)
        y_max = min(1, max(bin_stats['usage_rate']) + 0.1)
        ax.set_ylim(y_min, y_max)
        
        # 凡例
        if i == 0:  # 最初のグラフのみ凡例表示
            ax.legend(loc='upper left', fontsize=9, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # 保存
    output_file = output_path / "usage_rate_by_bins_smooth_advanced.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"移動平均による利用意向率プロットを保存しました: {output_file}")
    
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
        print(f"  Q4(全体)の利用意向率: {feature_data['利用意向'].mean():.1%}")

if __name__ == "__main__":
    plot_usage_rate_by_bins()