"""
CART（Classification and Regression Trees）分析クラス
scikit-learnのDecisionTreeClassifierを使用して、実用的なルールを抽出
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CARTAnalyzer:
    """CART分析クラス"""
    
    def __init__(self, max_depth: int = 3, min_samples_split: int = 5, 
                 min_samples_leaf: int = 2, random_state: int = 42):
        """
        Args:
            max_depth: 決定木の最大深さ
            min_samples_split: 分割に必要な最小サンプル数
            min_samples_leaf: 葉ノードの最小サンプル数
            random_state: 乱数シード
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        self.tree = None
        self.feature_names = None
        self.label_encoder = LabelEncoder()
        self.rules = []
        
    def prepare_data(self, data: pd.DataFrame, features: List[str], target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """データの準備"""
        # 使用する特徴量と目的変数を抽出
        X = data[features].copy()
        y = data[target_column]
        
        # カテゴリ変数のエンコーディング
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = self.label_encoder.fit_transform(X[col].astype(str))
        
        # 欠損値の処理
        X = X.fillna(X.median())
        
        return X, y
    
    def fit_tree(self, data: pd.DataFrame, features: List[str], target_column: str) -> bool:
        """決定木の学習"""
        try:
            # データの準備
            X, y = self.prepare_data(data, features, target_column)
            self.feature_names = features
            
            # 決定木の作成
            self.tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                criterion='gini'  # 分類問題用
            )
            
            # 学習
            self.tree.fit(X, y)
            
            print(f"決定木の学習完了: 深さ={self.tree.get_depth()}, ノード数={self.tree.tree_.node_count}")
            return True
            
        except Exception as e:
            print(f"決定木の学習に失敗: {e}")
            return False
    
    def extract_rules(self, data: pd.DataFrame) -> List[Dict]:
        """決定木からルールを抽出"""
        if self.tree is None:
            print("決定木が学習されていません")
            return []
        
        try:
            # 決定木の構造を取得
            tree = self.tree.tree_
            n_nodes = tree.node_count
            children_left = tree.children_left
            children_right = tree.children_right
            feature = tree.feature
            threshold = tree.threshold
            value = tree.value
            
            self.rules = []
            
            def extract_node_rules(node_id: int, path_conditions: List[str], depth: int = 0):
                """ノードからルールを再帰的に抽出"""
                if depth > self.max_depth:
                    return
                
                if children_left[node_id] == children_right[node_id]:  # 葉ノード
                    # 葉ノードの情報を取得
                    node_samples = value[node_id][0]
                    total_samples = np.sum(node_samples)
                    
                    # 利用意向率を計算
                    intention_rate = node_samples[1] / total_samples if total_samples > 0 else 0
                    
                    # ルールの作成（min_samples_leafの制限を緩和）
                    rule = {
                        'condition': self._format_conditions(path_conditions),
                        'intention_rate': intention_rate,
                        'sample_size': int(total_samples),
                        'depth': depth,
                        'node_id': node_id,
                        'confidence': self._calculate_confidence(intention_rate, total_samples),
                        'support': total_samples / len(data)  # サポート（全データに対する割合）
                    }
                    self.rules.append(rule)
                    return
                
                # 分割条件
                if feature[node_id] >= 0 and feature[node_id] < len(self.feature_names):
                    split_feature = self.feature_names[feature[node_id]]
                    split_threshold = threshold[node_id]
                    
                    # 左側の条件（<=）
                    left_condition = f"{split_feature} ≤ {split_threshold:.3f}"
                    left_path = path_conditions + [left_condition]
                    extract_node_rules(children_left[node_id], left_path, depth + 1)
                    
                    # 右側の条件（>）
                    right_condition = f"{split_feature} > {split_threshold:.3f}"
                    right_path = path_conditions + [right_condition]
                    extract_node_rules(children_right[node_id], right_path, depth + 1)
            
            # ルートノードから開始
            extract_node_rules(0, [])
            
            # 利用意向率でソート
            self.rules.sort(key=lambda x: x['intention_rate'], reverse=True)
            
            print(f"ルール抽出完了: {len(self.rules)}個のルール")
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
    
    def evaluate_tree(self, data: pd.DataFrame, features: List[str], target_column: str) -> Dict:
        """決定木の性能評価"""
        if self.tree is None:
            return {}
        
        try:
            # データの準備
            X, y = self.prepare_data(data, features, target_column)
            
            # 交差検証
            cv_scores = cross_val_score(self.tree, X, y, cv=5, scoring='accuracy')
            
            # 予測
            y_pred = self.tree.predict(X)
            
            # 性能指標
            accuracy = accuracy_score(y, y_pred)
            
            # 分類レポート
            class_report = classification_report(y, y_pred, output_dict=True)
            
            # 混同行列
            conf_matrix = confusion_matrix(y, y_pred)
            
            return {
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'accuracy': accuracy,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix
            }
            
        except Exception as e:
            print(f"決定木の評価に失敗: {e}")
            return {}
    
    def get_feature_importance(self) -> Dict[str, float]:
        """特徴量重要度を取得"""
        if self.tree is None:
            return {}
        
        try:
            importance_dict = {}
            for i, feature in enumerate(self.feature_names):
                importance_dict[feature] = self.tree.feature_importances_[i]
            
            # 重要度でソート
            sorted_importance = dict(sorted(importance_dict.items(), 
                                          key=lambda x: x[1], reverse=True))
            return sorted_importance
            
        except Exception as e:
            print(f"特徴量重要度の取得に失敗: {e}")
            return {}
    
    def visualize_tree(self, save_path: Optional[str] = None):
        """決定木の可視化"""
        if self.tree is None:
            print("決定木が学習されていません")
            return
        
        try:
            plt.figure(figsize=(20, 10))
            plot_tree(self.tree, 
                     feature_names=self.feature_names,
                     class_names=['利用しない', '利用する'],
                     filled=True,
                     rounded=True,
                     fontsize=10)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"決定木の可視化を保存: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"決定木の可視化に失敗: {e}")
    
    def export_tree_text(self, save_path: Optional[str] = None) -> str:
        """決定木をテキスト形式で出力"""
        if self.tree is None:
            return ""
        
        try:
            tree_text = export_text(self.tree, 
                                   feature_names=self.feature_names,
                                   class_names=['利用しない', '利用する'])
            
            if save_path:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(tree_text)
                print(f"決定木のテキスト出力を保存: {save_path}")
            
            return tree_text
            
        except Exception as e:
            print(f"決定木のテキスト出力に失敗: {e}")
            return ""
    
    def get_rules_summary(self) -> Dict:
        """ルールの要約情報を取得"""
        if not self.rules:
            return {}
        
        try:
            # 実用性による分類
            practical_rules = [r for r in self.rules if r['sample_size'] >= 5]
            partial_rules = [r for r in self.rules if 3 <= r['sample_size'] < 5]
            low_rules = [r for r in self.rules if r['sample_size'] < 3]
            
            # 深さによる分類
            depth_1_rules = [r for r in self.rules if r['depth'] == 1]
            depth_2_rules = [r for r in self.rules if r['depth'] == 2]
            depth_3_rules = [r for r in self.rules if r['depth'] == 3]
            
            return {
                'total_rules': len(self.rules),
                'practical_rules': len(practical_rules),
                'partial_rules': len(partial_rules),
                'low_rules': len(low_rules),
                'depth_1_rules': len(depth_1_rules),
                'depth_2_rules': len(depth_2_rules),
                'depth_3_rules': len(depth_3_rules),
                'avg_intention_rate': np.mean([r['intention_rate'] for r in self.rules]),
                'avg_sample_size': np.mean([r['sample_size'] for r in self.rules]),
                'avg_support': np.mean([r['support'] for r in self.rules])
            }
            
        except Exception as e:
            print(f"ルール要約の取得に失敗: {e}")
            return {}
