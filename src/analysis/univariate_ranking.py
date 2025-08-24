"""
単変量ランキング分析
各特徴量の重要度を評価し、ランキングを作成
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class UnivariateRanking:
    """単変量ランキング分析クラス"""
    
    def __init__(self, target_column: str = '利用意向'):
        """
        Args:
            target_column: 目的変数（利用意向）の列名
        """
        self.target_column = target_column
        self.results = []
    
    def analyze_numerical_feature(self, data: pd.DataFrame, feature: str) -> Dict:
        """数値型特徴量の分析"""
        try:
            # 欠損値を除外
            valid_data = data[[feature, self.target_column]].dropna()
            if len(valid_data) < 10:  # 最小サンプル数チェック
                return None
            
            X = valid_data[feature]
            y = valid_data[self.target_column]
            
            # 単一AUC
            auc = roc_auc_score(y, X)
            
            # Mann-Whitney U検定の効果量
            group1 = X[y == 1]
            group0 = X[y == 0]
            if len(group1) > 0 and len(group0) > 0:
                u_stat, p_value = stats.mannwhitneyu(group1, group0, alternative='two-sided')
                # Cliff's delta近似
                cliff_delta = (2 * u_stat) / (len(group1) * len(group0)) - 1
            else:
                cliff_delta = 0
            
            # ポイント・バイシリアル相関
            correlation = stats.pointbiserialr(y, X)[0]
            
            # 解釈の生成
            interpretation = self._generate_interpretation(feature, correlation, auc)
            
            return {
                'feature': feature,
                'type': 'numerical',
                'auc': auc,
                'cliff_delta': cliff_delta,
                'correlation': correlation,
                'interpretation': interpretation,
                'sample_size': len(valid_data)
            }
            
        except Exception as e:
            print(f"Error analyzing numerical feature {feature}: {e}")
            return None
    
    def analyze_categorical_feature(self, data: pd.DataFrame, feature: str) -> Dict:
        """カテゴリ型特徴量の分析"""
        try:
            # 欠損値を除外
            valid_data = data[[feature, self.target_column]].dropna()
            if len(valid_data) < 10:
                return None
            
            # クロス集計表
            contingency = pd.crosstab(valid_data[feature], valid_data[self.target_column])
            
            # Fisher検定
            if contingency.shape == (2, 2):
                fisher_result = stats.fisher_exact(contingency)
                p_value = fisher_result[1]
            else:
                # カイ二乗検定
                chi2_result = stats.chi2_contingency(contingency)
                p_value = chi2_result[1]
            
            # Cramér's V
            chi2 = stats.chi2_contingency(contingency)[0]
            n = len(valid_data)
            min_dim = min(contingency.shape) - 1
            cramer_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
            
            # 各カテゴリの利用意向率
            intention_rates = valid_data.groupby(feature)[self.target_column].mean()
            max_rate_diff = intention_rates.max() - intention_rates.min()
            
            # 解釈の生成
            interpretation = self._generate_categorical_interpretation(feature, max_rate_diff, cramer_v)
            
            return {
                'feature': feature,
                'type': 'categorical',
                'p_value': p_value,
                'cramer_v': cramer_v,
                'max_rate_diff': max_rate_diff,
                'interpretation': interpretation,
                'sample_size': len(valid_data)
            }
            
        except Exception as e:
            print(f"Error analyzing categorical feature {feature}: {e}")
            return None
    
    def _generate_interpretation(self, feature: str, correlation: float, auc: float) -> str:
        """数値型特徴量の解釈を生成"""
        if abs(correlation) > 0.3:
            direction = "高いほど利用意向あり" if correlation > 0 else "高いほど利用意向なし"
            strength = "強い" if abs(correlation) > 0.5 else "中程度の"
        else:
            direction = "利用意向との関連性は弱い"
            strength = "弱い"
        
        return f"{strength}{direction} ({correlation:.2f})"
    
    def _generate_categorical_interpretation(self, feature: str, rate_diff: float, cramer_v: float) -> str:
        """カテゴリ型特徴量の解釈を生成"""
        if rate_diff > 0.2:
            strength = "強い" if rate_diff > 0.3 else "中程度の"
            return f"{strength}カテゴリ間差 (差: {rate_diff:.1%})"
        else:
            return f"カテゴリ間差は小さい (差: {rate_diff:.1%})"
    
    def analyze_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """全特徴量を分析してランキングを作成"""
        self.results = []
        
        # 目的変数を除外
        features = [col for col in data.columns if col != self.target_column]
        
        for feature in features:
            if data[feature].dtype in ['int64', 'float64']:
                result = self.analyze_numerical_feature(data, feature)
            else:
                result = self.analyze_categorical_feature(data, feature)
            
            if result:
                self.results.append(result)
        
        # 重要度スコアでランキング
        for result in self.results:
            if result['type'] == 'numerical':
                # AUCと相関係数の絶対値の平均
                result['importance_score'] = (result['auc'] + abs(result['correlation'])) / 2
            else:
                # Cramér's Vと最大率差の組み合わせ
                result['importance_score'] = (result['cramer_v'] + result['max_rate_diff']) / 2
        
        # 重要度スコアでソート
        self.results.sort(key=lambda x: x['importance_score'], reverse=True)
        
        # DataFrameに変換
        ranking_df = pd.DataFrame(self.results)
        return ranking_df
    
    def get_top_features(self, n: int = 5) -> List[str]:
        """上位N個の特徴量を取得"""
        if not self.results:
            return []
        return [result['feature'] for result in self.results[:n]]
