"""
データ前処理実行スクリプト
アンケートデータの前処理を実行し、分析用のデータを準備
"""

import sys
import os
from pathlib import Path

# パスを追加
sys.path.append(str(Path(__file__).parent / 'src'))

from analysis.preprocessing import DataPreprocessor

def main(input_file=None, output_file=None):
    """前処理メイン実行関数"""
    print("=== データ前処理を開始します ===")
    
    # 設定
    if input_file is None:
        input_file = "data/raw/sprint_data - data.csv"
    if output_file is None:
        output_file = "data/processed/preprocessed_data_real.csv"
    target_column = "利用意向"
    
    # 前処理クラスの初期化
    preprocessor = DataPreprocessor(target_column=target_column)
    
    # 前処理パイプラインの実行
    print(f"入力ファイル: {input_file}")
    print(f"出力ファイル: {output_file}")
    
    # 前処理オプション
    preprocessing_options = {
        'clean': True,              # データクリーニング
        'handle_missing': True,     # 欠損値処理
        'convert_types': True,      # データ型変換
        'create_features': True,    # 特徴量作成
        'remove_outliers': False   # 外れ値除去（デフォルトは無効）
    }
    
    # 前処理実行
    processed_data = preprocessor.preprocess_pipeline(
        input_file, 
        **preprocessing_options
    )
    
    if processed_data is None:
        print("前処理に失敗しました")
        return
    
    # 出力ディレクトリの作成
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 前処理済みデータの保存
    success = preprocessor.save_preprocessed_data(processed_data, output_file)
    
    if success:
        print("\n=== 前処理完了 ===")
        print(f"前処理済みデータ: {output_file}")
        print(f"データ形状: {processed_data.shape}")
        print(f"特徴量数: {len(processed_data.columns)}")
        
        # 基本統計の表示
        print("\n=== 前処理後の基本統計 ===")
        print(processed_data.describe())
        
        # 列名の確認
        print(f"\n=== 特徴量一覧 ===")
        for i, col in enumerate(processed_data.columns):
            print(f"{i+1:2d}. {col}")
        
        # 目的変数の分布確認
        if target_column in processed_data.columns:
            print(f"\n=== 目的変数 '{target_column}' の分布 ===")
            print(processed_data[target_column].value_counts())
            print(f"利用意向率: {processed_data[target_column].mean():.2%}")
    else:
        print("前処理済みデータの保存に失敗しました")

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main()
