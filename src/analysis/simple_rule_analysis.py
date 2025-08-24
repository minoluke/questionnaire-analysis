"""
簡易ルール分析クラス
READMEに記載されている簡易ルール分析（1本のif-else）を実装
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SimpleRuleAnalyzer:
    """簡易ルール分析クラス"""
    
    def __init__(self, min_samples: int = 3):
        """
        Args:
            min_samples: ルールに必要な最小サンプル数
        """
        self.min_samples = min_samples
        self.best_rule = None
        self.all_rules = []
        
    def generate_threshold_candidates(self, data: pd.Series) -> List[float]:
        """しきい値候補を生成（分位点、中間値）"""
        candidates = []
        
        # 分位点（25%, 50%, 75%）
        quantiles = [0.25, 0.5, 0.75]
        for q in quantiles:
            candidates.append(data.quantile(q))
        
        # 中間値
        candidates.append(data.median())
        
        # 平均値
        candidates.append(data.mean())
        
        # 重複除去とソート
        candidates = sorted(list(set(candidates)))
        
        return candidates
    
    def evaluate_rule(self, data: pd.DataFrame, feature1: str, threshold1: float, 
                     feature2: str, threshold2: float, target_column: str) -> Dict:
        """ルールの評価"""
        try:
            # ルールの適用
            rule_mask = (data[feature1] > threshold1) & (data[feature2] > threshold2)
            
            # サンプル数の確認
            rule_samples = rule_mask.sum()
            if rule_samples < self.min_samples:
                return None
            
            # ルール内とルール外のデータ
            rule_data = data[rule_mask]
            other_data = data[~rule_mask]
            
            if len(rule_data) == 0 or len(other_data) == 0:
                return None
            
            # 利用意向率の計算
            rule_intention_rate = rule_data[target_column].mean()
            other_intention_rate = other_data[target_column].mean()
            overall_intention_rate = data[target_column].mean()
            
            # 予測値の作成
            y_true = data[target_column]
            y_pred = rule_mask.astype(int)
            
            # 性能指標の計算
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            
            # Liftの計算
            lift = rule_intention_rate / overall_intention_rate if overall_intention_rate > 0 else 0
            
            return {
                'feature1': feature1,
                'threshold1': threshold1,
                'feature2': feature2,
                'threshold2': threshold2,
                'rule_condition': f"{feature1} > {threshold1:.3f} AND {feature2} > {threshold2:.3f}",
                'rule_samples': rule_samples,
                'other_samples': len(other_data),
                'rule_intention_rate': rule_intention_rate,
                'other_intention_rate': other_intention_rate,
                'overall_intention_rate': overall_intention_rate,
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'lift': lift
            }
            
        except Exception as e:
            print(f"ルール評価エラー: {e}")
            return None
    
    def find_best_rule(self, data: pd.DataFrame, features: List[str], target_column: str) -> Dict:
        """最良のルールを探索"""
        print("簡易ルール分析を開始中...")
        
        if len(features) < 2:
            print("特徴量が不足しています（最低2個必要）")
            return {}
        
        # 特徴量の組み合わせ
        feature_pairs = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                feature_pairs.append((features[i], features[j]))
        
        print(f"特徴量ペア数: {len(feature_pairs)}")
        
        best_rule = None
        best_f1 = -1
        all_rules = []
        
        for feature1, feature2 in feature_pairs:
            print(f"分析中: {feature1} + {feature2}")
            
            # しきい値候補の生成
            thresholds1 = self.generate_threshold_candidates(data[feature1])
            thresholds2 = self.generate_threshold_candidates(data[feature2])
            
            print(f"  {feature1}のしきい値候補: {len(thresholds1)}個")
            print(f"  {feature2}のしきい値候補: {len(thresholds2)}個")
            
            # 全組み合わせの試行
            for t1 in thresholds1:
                for t2 in thresholds2:
                    rule_result = self.evaluate_rule(data, feature1, t1, feature2, t2, target_column)
                    
                    if rule_result:
                        all_rules.append(rule_result)
                        
                        # F1スコアで最良ルールを更新
                        if rule_result['f1_score'] > best_f1:
                            best_f1 = rule_result['f1_score']
                            best_rule = rule_result
        
        self.best_rule = best_rule
        self.all_rules = all_rules
        
        print(f"分析完了: {len(all_rules)}個の有効なルールを発見")
        
        if best_rule:
            print(f"最良ルール: {best_rule['rule_condition']}")
            print(f"F1スコア: {best_rule['f1_score']:.3f}")
            print(f"利用意向率: {best_rule['rule_intention_rate']:.1%}")
        
        return best_rule if best_rule else {}
    
    def get_rule_summary(self) -> Dict:
        """ルールの要約情報を取得"""
        if not self.all_rules:
            return {}
        
        # F1スコアでソート
        sorted_rules = sorted(self.all_rules, key=lambda x: x['f1_score'], reverse=True)
        
        # 上位ルールの統計
        top_rules = sorted_rules[:5] if len(sorted_rules) >= 5 else sorted_rules
        
        return {
            'total_rules': len(self.all_rules),
            'best_f1_score': self.best_rule['f1_score'] if self.best_rule else 0,
            'best_rule': self.best_rule['rule_condition'] if self.best_rule else None,
            'avg_f1_score': np.mean([r['f1_score'] for r in self.all_rules]),
            'avg_lift': np.mean([r['lift'] for r in self.all_rules]),
            'top_rules': top_rules
        }
    
    def export_rules_to_dataframe(self) -> pd.DataFrame:
        """ルールをDataFrameに変換"""
        if not self.all_rules:
            return pd.DataFrame()
        
        # 結果をDataFrameに変換
        rules_df = pd.DataFrame(self.all_rules)
        
        # 列の順序を整理
        columns_order = [
            'rule_condition', 'rule_samples', 'other_samples',
            'rule_intention_rate', 'other_intention_rate', 'overall_intention_rate',
            'accuracy', 'f1_score', 'precision', 'recall', 'lift',
            'feature1', 'threshold1', 'feature2', 'threshold2'
        ]
        
        # 存在する列のみ選択
        existing_columns = [col for col in columns_order if col in rules_df.columns]
        rules_df = rules_df[existing_columns]
        
        return rules_df
