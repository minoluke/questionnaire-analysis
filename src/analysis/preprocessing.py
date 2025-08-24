"""
データ前処理クラス
アンケートデータの前処理を実行
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """データ前処理クラス"""
    
    def __init__(self, target_column="利用意向"):
        """
        初期化
        
        Args:
            target_column (str): 目的変数の列名
        """
        self.target_column = target_column
        self.original_shape = None
        self.processed_shape = None
        
    def load_data(self, file_path):
        """
        データの読み込み
        
        Args:
            file_path (str): データファイルのパス
            
        Returns:
            pd.DataFrame: 読み込んだデータ
        """
        try:
            # CSVファイルの読み込み
            data = pd.read_csv(file_path)
            self.original_shape = data.shape
            print(f"データ読み込み完了: {data.shape}")
            return data
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            return None
    
    def clean_data(self, data):
        """
        データクリーニング
        
        Args:
            data (pd.DataFrame): クリーニング対象のデータ
            
        Returns:
            pd.DataFrame: クリーニング済みデータ
        """
        print("データクリーニングを実行中...")
        
        # 重複行の削除
        before_duplicates = data.duplicated().sum()
        data = data.drop_duplicates()
        after_duplicates = data.duplicated().sum()
        
        if before_duplicates > after_duplicates:
            print(f"重複行を {before_duplicates - after_duplicates} 行削除しました")
        
        # 列名の正規化（空白除去、特殊文字除去、型情報除去）
        data.columns = data.columns.str.strip().str.replace(' ', '_').str.replace(':', '').str.replace('int', '').str.replace('float', '').str.replace('bool', '').str.replace('class', '')
        
        # 数値列の処理
        numeric_columns = ['最高月収', '最低月収', '平均月収', '最高支出', '最低支出', 
                          '月収に対する自由資産率', '貯金絶対額']
        
        for col in numeric_columns:
            if col in data.columns:
                # 負の値の処理（収入・支出は負の値が不適切）
                if '収' in col or '支出' in col:
                    data[col] = data[col].abs()
                
                # 極端な値の処理（外れ値の上限設定）
                if '収' in col:
                    data[col] = np.where(data[col] > 2000000, 2000000, data[col])
                elif '支出' in col:
                    data[col] = np.where(data[col] > 1000000, 1000000, data[col])
                elif '貯金絶対額' in col:
                    data[col] = np.where(data[col] > 50000000, 50000000, data[col])
        
        print("データクリーニング完了")
        return data
    
    def handle_missing_values(self, data):
        """
        欠損値の処理
        
        Args:
            data (pd.DataFrame): 処理対象のデータ
            
        Returns:
            pd.DataFrame: 欠損値処理済みデータ
        """
        print("欠損値処理を実行中...")
        
        # 欠損値の確認
        missing_counts = data.isnull().sum()
        if missing_counts.sum() > 0:
            print("欠損値の状況:")
            print(missing_counts[missing_counts > 0])
        
        # 数値列の欠損値は中央値で補完
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data[col].isnull().sum() > 0:
                median_val = data[col].median()
                data[col].fillna(median_val, inplace=True)
                print(f"{col}: 中央値 {median_val} で補完")
        
        # カテゴリ列の欠損値は最頻値で補完
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if data[col].isnull().sum() > 0:
                mode_val = data[col].mode()[0]
                data[col].fillna(mode_val, inplace=True)
                print(f"{col}: 最頻値 '{mode_val}' で補完")
        
        print("欠損値処理完了")
        return data
    
    def convert_data_types(self, data):
        """
        データ型の変換
        
        Args:
            data (pd.DataFrame): 変換対象のデータ
            
        Returns:
            pd.DataFrame: 型変換済みデータ
        """
        print("データ型変換を実行中...")
        
        # 数値列の型変換
        numeric_columns = ['最高月収', '最低月収', '平均月収', '最高支出', '最低支出', 
                          '月収に対する自由資産率', '貯金絶対額']
        
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # カテゴリ列の型変換
        categorical_columns = ['業種/業態']
        for col in categorical_columns:
            if col in data.columns:
                data[col] = data[col].astype('category')
        
        # 目的変数の型変換
        if self.target_column in data.columns:
            data[self.target_column] = data[self.target_column].astype(int)
        
        print("データ型変換完了")
        return data
    
    def create_features(self, data):
        """
        特徴量の作成
        
        Args:
            data (pd.DataFrame): 特徴量作成対象のデータ
            
        Returns:
            pd.DataFrame: 特徴量追加済みデータ
        """
        print("特徴量作成を実行中...")
        
        # 収入変動額
        if all(col in data.columns for col in ['最高月収', '最低月収']):
            data['収入変動額'] = data['最高月収'] - data['最低月収']
        
        # 収入変動率（収入変動額/平均月収）
        if all(col in data.columns for col in ['最高月収', '最低月収', '平均月収']):
            data['収入変動率'] = (data['最高月収'] - data['最低月収']) / data['平均月収']
        
        # 支出変動率（(最大支出-最小支出)/(最大支出,最小支出の平均）
        if all(col in data.columns for col in ['最高支出', '最低支出']):
            data['支出変動率'] = (data['最高支出'] - data['最低支出']) / ((data['最高支出'] + data['最低支出']) / 2)
        
        # 業種のダミー変数化
        if '業種/業態' in data.columns:
            industry_dummies = pd.get_dummies(data['業種/業態'], prefix='業種')
            data = pd.concat([data, industry_dummies], axis=1)
        
        print("特徴量作成完了")
        return data
    
    def remove_outliers(self, data, method='iqr'):
        """
        外れ値の除去
        
        Args:
            data (pd.DataFrame): 外れ値除去対象のデータ
            method (str): 外れ値検出方法 ('iqr' または 'zscore')
            
        Returns:
            pd.DataFrame: 外れ値除去済みデータ
        """
        print("外れ値除去を実行中...")
        
        original_count = len(data)
        
        if method == 'iqr':
            # IQR法による外れ値検出
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col == self.target_column:
                    continue
                    
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 外れ値の除去
                data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        
        elif method == 'zscore':
            # Z-score法による外れ値検出
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col == self.target_column:
                    continue
                    
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                data = data[z_scores < 3]
        
        removed_count = original_count - len(data)
        if removed_count > 0:
            print(f"外れ値を {removed_count} 行除去しました")
        
        print("外れ値除去完了")
        return data
    
    def preprocess_pipeline(self, input_file, clean=True, handle_missing=True, 
                           convert_types=True, create_features=True, remove_outliers=False):
        """
        前処理パイプラインの実行
        
        Args:
            input_file (str): 入力ファイルパス
            clean (bool): データクリーニングの実行フラグ
            handle_missing (bool): 欠損値処理の実行フラグ
            convert_types (bool): データ型変換の実行フラグ
            create_features (bool): 特徴量作成の実行フラグ
            remove_outliers (bool): 外れ値除去の実行フラグ
            
        Returns:
            pd.DataFrame: 前処理済みデータ
        """
        # データ読み込み
        data = self.load_data(input_file)
        if data is None:
            return None
        
        # 前処理の実行
        if clean:
            data = self.clean_data(data)
        
        if handle_missing:
            data = self.handle_missing_values(data)
        
        if convert_types:
            data = self.convert_data_types(data)
        
        if create_features:
            data = self.create_features(data)
        
        if remove_outliers:
            data = self.remove_outliers(data)
        
        self.processed_shape = data.shape
        return data
    
    def save_preprocessed_data(self, data, output_file):
        """
        前処理済みデータの保存
        
        Args:
            data (pd.DataFrame): 保存するデータ
            output_file (str): 出力ファイルパス
            
        Returns:
            bool: 保存成功フラグ
        """
        try:
            # 出力ディレクトリの作成
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # CSVファイルとして保存
            data.to_csv(output_file, index=False, encoding='utf-8')
            
            print(f"前処理済みデータを保存しました: {output_file}")
            return True
            
        except Exception as e:
            print(f"データ保存エラー: {e}")
            return False
    
    def get_preprocessing_summary(self):
        """
        前処理の要約情報を取得
        
        Returns:
            dict: 前処理要約情報
        """
        summary = {
            'original_shape': self.original_shape,
            'processed_shape': self.processed_shape,
            'reduction_rate': None
        }
        
        if self.original_shape and self.processed_shape:
            original_size = self.original_shape[0] * self.original_shape[1]
            processed_size = self.processed_shape[0] * self.processed_shape[1]
            summary['reduction_rate'] = (original_size - processed_size) / original_size
        
        return summary
