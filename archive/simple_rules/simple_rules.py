"""
簡易ルール分析（決定木）
上位特徴量を使用して浅い決定木を作成し、実用的なルールを抽出
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class SimpleRules:
    """簡易ルール分析クラス"""
    
    def __init__(self, max_depth: int = 2, min_samples_split: int = 10):
        """
        Args:
            max_depth: 決定木の最大深さ
            min_samples_split: 分割に必要な最小サンプル数
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.rules = []
    
    def create_decision_tree(self, data: pd.DataFrame, features: List[str], target_column: str) -> bool:
        """決定木を作成"""
        try:
            # 使用する特徴量と目的変数のみを抽出
            X = data[features].copy()
            y = data[target_column]
            
            # 欠損値を除外
            valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < self.min_samples_split * 2:
                print(f"サンプル数が不足しています: {len(X)}")
                return False
            
            # 決定木の作成
            self.tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=42
            )
            
            self.tree.fit(X, y)
            return True
            
        except Exception as e:
            print(f"決定木の作成に失敗: {e}")
            return False
    
    def extract_rules(self, data: pd.DataFrame, features: List[str], target_column: str) -> List[Dict]:
        """決定木からルールを抽出"""
        if self.tree is None:
            print("決定木が作成されていません")
            return []
        
        try:
            # 決定木の構造を取得
            tree = self.tree
            n_nodes = tree.tree_.node_count
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            feature = tree.tree_.feature
            threshold = tree.tree_.threshold
            value = tree.tree_.value
            
            print(f"決定木の構造: ノード数={n_nodes}")
            print(f"特徴量インデックス: {feature}")
            print(f"閾値: {threshold}")
            
            self.rules = []
            
            def extract_node_rules(node_id, path_conditions):
                """ノードからルールを再帰的に抽出"""
                if children_left[node_id] == children_right[node_id]:  # 葉ノード
                    # 葉ノードの情報を取得
                    node_samples = value[node_id][0]
                    total_samples = np.sum(node_samples)
                    intention_rate = node_samples[1] / total_samples if total_samples > 0 else 0
                    
                    print(f"葉ノード {node_id}: サンプル数={total_samples}, 利用意向率={intention_rate:.3f}")
                    
                    # サンプル数が1件でもルールとして抽出
                    rule = {
                        'condition': self._format_conditions(path_conditions),
                        'intention_rate': intention_rate,
                        'sample_size': int(total_samples),
                        'confidence': self._calculate_confidence(intention_rate, total_samples)
                    }
                    self.rules.append(rule)
                    print(f"ルール追加: {rule['condition']}")
                    return
                
                # 分割条件
                split_feature_idx = feature[node_id]
                if split_feature_idx >= 0 and split_feature_idx < len(features):
                    split_feature = features[split_feature_idx]
                    split_threshold = threshold[node_id]
                    
                    print(f"分割ノード {node_id}: {split_feature} ≤ {split_threshold:.2f}")
                    
                    # 左側の条件（<=）
                    left_condition = f"{split_feature} ≤ {split_threshold:.2f}"
                    left_path = path_conditions + [left_condition]
                    extract_node_rules(children_left[node_id], left_path)
                    
                    # 右側の条件（>）
                    right_condition = f"{split_feature} > {split_threshold:.2f}"
                    right_path = path_conditions + [right_condition]
                    extract_node_rules(children_right[node_id], right_path)
                else:
                    print(f"無効な特徴量インデックス: {split_feature_idx}")
            
            # ルートノードから開始
            extract_node_rules(0, [])
            
            print(f"抽出されたルール数: {len(self.rules)}")
            
            # 利用意向率でソート
            self.rules.sort(key=lambda x: x['intention_rate'], reverse=True)
            
            return self.rules
            
        except Exception as e:
            print(f"ルールの抽出に失敗: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _format_conditions(self, conditions: List[str]) -> str:
        """条件を読みやすい形式に整形"""
        if not conditions:
            return "すべて"
        
        if len(conditions) == 1:
            return conditions[0]
        
        return " AND ".join(conditions)
    
    def _calculate_confidence(self, intention_rate: float, sample_size: int) -> float:
        """信頼区間の近似計算"""
        if sample_size == 0:
            return 0.0
        
        # Wilson信頼区間の近似
        z = 1.96  # 95%信頼区間
        se = np.sqrt(intention_rate * (1 - intention_rate) / sample_size)
        margin = z * se
        
        return max(0.0, min(1.0, margin))
    
    def evaluate_rules(self, data: pd.DataFrame, features: List[str], target_column: str) -> Dict:
        """ルールの評価"""
        if not self.rules:
            return {}
        
        try:
            # 各ルールの適用結果を評価
            rule_evaluations = []
            
            for rule in self.rules:
                # 条件に合致するサンプルを抽出
                mask = self._apply_rule_condition(data, rule['condition'], features)
                matched_data = data[mask]
                
                if len(matched_data) > 0:
                    actual_rate = matched_data[target_column].mean()
                    accuracy = abs(actual_rate - rule['intention_rate'])
                    
                    evaluation = {
                        'rule': rule['condition'],
                        'predicted_rate': rule['intention_rate'],
                        'actual_rate': actual_rate,
                        'accuracy': accuracy,
                        'matched_samples': len(matched_data)
                    }
                    rule_evaluations.append(evaluation)
            
            # 全体の評価
            if rule_evaluations:
                avg_accuracy = np.mean([e['accuracy'] for e in rule_evaluations])
                total_coverage = sum([e['matched_samples'] for e in rule_evaluations]) / len(data)
            else:
                avg_accuracy = 0.0
                total_coverage = 0.0
            
            return {
                'rule_evaluations': rule_evaluations,
                'average_accuracy': avg_accuracy,
                'total_coverage': total_coverage,
                'total_rules': len(self.rules)
            }
            
        except Exception as e:
            print(f"ルールの評価に失敗: {e}")
            return {}
    
    def _apply_rule_condition(self, data: pd.DataFrame, condition: str, features: List[str]) -> pd.Series:
        """ルール条件をデータに適用"""
        try:
            if condition == "すべて":
                return pd.Series([True] * len(data), index=data.index)
            
            # 条件を解析（簡易版）
            mask = pd.Series([True] * len(data), index=data.index)
            
            for feature in features:
                if feature in condition:
                    if "≤" in condition:
                        # 数値の閾値条件
                        threshold = float(condition.split("≤")[1].strip().split()[0])
                        mask &= (data[feature] <= threshold)
                    elif ">" in condition:
                        # 数値の閾値条件
                        threshold = float(condition.split(">")[1].strip().split()[0])
                        mask &= (data[feature] > threshold)
            
            return mask
            
        except Exception as e:
            print(f"条件の適用に失敗: {e}")
            return pd.Series([False] * len(data), index=data.index)
    
    def evaluate_tree(self, data: pd.DataFrame, features: List[str], target_column: str) -> np.ndarray:
        """決定木の交差検証評価"""
        try:
            from sklearn.model_selection import cross_val_score
            
            X = data[features].copy()
            y = data[target_column]
            
            # 欠損値を除外
            valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 10:
                print("サンプル数が不足しています")
                return None
            
            # 5分割交差検証
            cv_scores = cross_val_score(self.tree, X, y, cv=min(5, len(X)), scoring='accuracy')
            return cv_scores
            
        except Exception as e:
            print(f"決定木の評価に失敗: {e}")
            return None
    
    def get_feature_importance(self, features: List[str]) -> Dict[str, float]:
        """特徴量重要度を取得"""
        if self.tree is None:
            return {}
        
        try:
            importance_dict = {}
            for i, feature in enumerate(features):
                if i < len(self.tree.feature_importances_):
                    importance_dict[feature] = self.tree.feature_importances_[i]
                else:
                    importance_dict[feature] = 0.0
            
            return importance_dict
            
        except Exception as e:
            print(f"特徴量重要度の取得に失敗: {e}")
            return {}
