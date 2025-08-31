"""
各特徴量と利用意向の関係をプロットする
単変量分析の結果を可視化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import japanize_matplotlib

def plot_feature_distributions(data_file=None, ranking_file=None, output_dir=None):
    """特徴量分布と利用意向の関係を可視化"""
    
    # デフォルト設定
    if data_file is None:
        data_file = "data/processed/preprocessed_100data.csv"
    if ranking_file is None:
        ranking_file = "output/tables/univariate_ranking_table.csv"
    if output_dir is None:
        output_dir = "output/visualizations"
    
    # データの読み込み
    data = pd.read_csv(data_file)
    ranking_df = pd.read_csv(ranking_file)
    
    # 出力ディレクトリの作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 目的変数の確認
    target_column = '利用意向'
    
    # 上位特徴量を取得（user_idは除外）
    top_features = ranking_df[
        (ranking_df['type'] == 'numerical') & 
        (ranking_df['feature'] != target_column) &
        (ranking_df['feature'] != 'user_id')
    ]['feature'].tolist()  # 全特徴量を取得
    
    print(f"可視化対象特徴量: {top_features}")
    
    # 図のサイズとレイアウトを設定
    n_features = len(top_features)
    n_cols = 4  # 列数を増やして全特徴量を表示
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # 1. ヒストグラム（利用意向別）
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    if n_features == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('特徴量分布（利用意向別）', fontsize=16, fontweight='bold')
    
    for i, feature in enumerate(top_features):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # 利用意向別にデータを分割
        data_no = data[data[target_column] == 0][feature]
        data_yes = data[data[target_column] == 1][feature]
        
        # 共通のbinを設定
        combined_data = pd.concat([data_no, data_yes])
        bin_min, bin_max = combined_data.min(), combined_data.max()
        bins = np.linspace(bin_min, bin_max, 16)  # 15個のbinを作るため16個の境界値
        
        # ヒストグラム
        ax.hist(data_no, bins=bins, alpha=0.6, label='利用意向なし', color='blue', density=True)
        ax.hist(data_yes, bins=bins, alpha=0.6, label='利用意向あり', color='red', density=True)
        
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('密度', fontsize=10)
        ax.set_title(f'{feature}の分布', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 余った subplot を非表示
    for i in range(len(top_features), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    hist_file = output_path / "feature_histograms.png"
    plt.savefig(hist_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"ヒストグラム保存: {hist_file}")
    
    # 2. ボックスプロット
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    if n_features == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('特徴量分布（ボックスプロット）', fontsize=16, fontweight='bold')
    
    for i, feature in enumerate(top_features):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # ボックスプロット
        data_plot = [
            data[data[target_column] == 0][feature].dropna(),
            data[data[target_column] == 1][feature].dropna()
        ]
        
        bp = ax.boxplot(data_plot, labels=['利用意向なし', '利用意向あり'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        ax.set_ylabel(feature, fontsize=10)
        ax.set_title(f'{feature}のボックスプロット', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # 余った subplot を非表示
    for i in range(len(top_features), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    box_file = output_path / "feature_boxplots.png"
    plt.savefig(box_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"ボックスプロット保存: {box_file}")
    
    # 3. 散布図（上位2特徴量）
    if len(top_features) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('上位特徴量の散布図', fontsize=16, fontweight='bold')
        
        # 上位4つの特徴量でペアを作成
        pairs = [
            (top_features[0], top_features[1]),
            (top_features[0], top_features[2] if len(top_features) > 2 else top_features[1]),
            (top_features[1], top_features[2] if len(top_features) > 2 else top_features[0]),
            (top_features[2] if len(top_features) > 2 else top_features[0], 
             top_features[3] if len(top_features) > 3 else top_features[1])
        ]
        
        for i, (feat1, feat2) in enumerate(pairs):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            # 利用意向別に色分け
            colors = ['blue' if intent == 0 else 'red' for intent in data[target_column]]
            labels = ['利用意向なし' if intent == 0 else '利用意向あり' for intent in data[target_column]]
            
            scatter = ax.scatter(data[feat1], data[feat2], c=colors, alpha=0.6, s=50)
            
            ax.set_xlabel(feat1, fontsize=10)
            ax.set_ylabel(feat2, fontsize=10)
            ax.set_title(f'{feat1} vs {feat2}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # 凡例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='利用意向なし'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='利用意向あり')
        ]
        fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=2)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        scatter_file = output_path / "feature_scatterplots.png"
        plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"散布図保存: {scatter_file}")
    
    # 4. 相関行列ヒートマップ
    correlation_features = top_features + [target_column]
    corr_matrix = data[correlation_features].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
    plt.title('特徴量相関行列', fontsize=16, fontweight='bold')
    plt.tight_layout()
    corr_file = output_path / "feature_correlation.png"
    plt.savefig(corr_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"相関行列保存: {corr_file}")
    
    # 5. 特徴量重要度バープロット
    plt.figure(figsize=(16, 8))
    all_features_for_bar = ranking_df[ranking_df['feature'] != 'user_id']  # user_idを除外
    
    bars = plt.bar(range(len(all_features_for_bar)), all_features_for_bar['importance_score'])
    plt.xlabel('特徴量', fontsize=12)
    plt.ylabel('重要度スコア', fontsize=12)
    plt.title('特徴量重要度ランキング（全特徴量）', fontsize=14, fontweight='bold')
    
    # 特徴量名を斜めに表示
    plt.xticks(range(len(all_features_for_bar)), all_features_for_bar['feature'], rotation=45, ha='right')
    
    # バーに数値を表示
    for i, (bar, score) in enumerate(zip(bars, all_features_for_bar['importance_score'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    importance_file = output_path / "feature_importance.png"
    plt.savefig(importance_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"重要度バープロット保存: {importance_file}")
    
    # 統計サマリーの出力
    print(f"\n=== 分析サマリー ===")
    print(f"データファイル: {data_file}")
    print(f"サンプル数: {len(data)}件")
    print(f"利用意向率: {data[target_column].mean():.1%}")
    print(f"分析特徴量数: {len(top_features)}個")
    print(f"可視化ファイル: {output_dir} ディレクトリに保存")

if __name__ == "__main__":
    import sys
    
    # コマンドライン引数の処理
    data_file = sys.argv[1] if len(sys.argv) > 1 else None
    ranking_file = sys.argv[2] if len(sys.argv) > 2 else None
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    plot_feature_distributions(data_file, ranking_file, output_dir)