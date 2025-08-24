"""
CART分析実行スクリプト v2
より実用的なルールを作成するため、決定木の設定を調整
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# パスを追加
sys.path.append(str(Path(__file__).parent / 'src'))

from analysis.cart_analysis import CARTAnalyzer

def main():
    """CART分析メイン実行関数"""
    print("=== CART分析（決定木）v2 を開始します ===")
    
    # 前処理済みデータの読み込み
    input_file = "data/processed/preprocessed_data.csv"
    ranking_file = "output/tables/univariate_ranking_table.csv"
    table_file = "output/tables/cart_rules_table_v2.csv"
    report_file = "output/reports/cart_rules_report_v2.md"
    tree_text_file = "output/reports/cart_tree_structure_v2.txt"
    tree_viz_file = "output/reports/cart_tree_visualization_v2.png"
    
    print(f"入力ファイル: {input_file}")
    print(f"単変量分析結果: {ranking_file}")
    print(f"テーブル出力: {table_file}")
    print(f"レポート出力: {report_file}")
    print(f"決定木構造: {tree_text_file}")
    print(f"決定木可視化: {tree_viz_file}")
    
    # データ読み込み
    try:
        data = pd.read_csv(input_file)
        print(f"データ読み込み完了: {data.shape}")
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return
    
    # 単変量分析結果の読み込み
    try:
        ranking_df = pd.read_csv(ranking_file)
        print(f"単変量分析結果読み込み完了: {len(ranking_df)}個の特徴量")
    except Exception as e:
        print(f"単変量分析結果読み込みエラー: {e}")
        return
    
    # 目的変数の確認
    target_column = "利用意向"
    if target_column not in data.columns:
        print(f"目的変数 '{target_column}' が見つかりません")
        return
    
    # 上位特徴量の選択（数値型のみ、重要度上位4個に制限）
    top_numerical_features = ranking_df[
        (ranking_df['type'] == 'numerical') & 
        (ranking_df['feature'] != target_column)
    ].head(4)['feature'].tolist()
    
    print(f"\n=== 使用する特徴量 ===")
    print(f"上位4個の数値型特徴量: {top_numerical_features}")
    
    # CART分析の実行
    print("\n=== CART分析を実行中 ===")
    
    # より実用的な設定で分析を試行
    cart_configs = [
        {'max_depth': 1, 'min_samples_split': 5, 'min_samples_leaf': 3, 'name': '単一分割決定木'},
        {'max_depth': 2, 'min_samples_split': 4, 'min_samples_leaf': 2, 'name': '浅い決定木'},
        {'max_depth': 3, 'min_samples_split': 3, 'min_samples_leaf': 2, 'name': '中程度の決定木'}
    ]
    
    best_analyzer = None
    best_practical_rules = 0
    
    for config in cart_configs:
        print(f"\n--- {config['name']}を試行中 ---")
        print(f"設定: max_depth={config['max_depth']}, min_samples_split={config['min_samples_split']}, min_samples_leaf={config['min_samples_leaf']}")
        
        analyzer = CARTAnalyzer(
            max_depth=config['max_depth'],
            min_samples_split=config['min_samples_split'],
            min_samples_leaf=config['min_samples_leaf'],
            random_state=42
        )
        
        # 決定木の学習
        if analyzer.fit_tree(data, top_numerical_features, target_column):
            # ルールの抽出
            rules = analyzer.extract_rules(data)
            
            if rules:
                print(f"ルール抽出成功: {len(rules)}個のルール")
                
                # 実用的なルール数をカウント
                practical_rules = [r for r in rules if r['sample_size'] >= 3]
                print(f"実用的なルール: {len(practical_rules)}個")
                
                # 各ルールのサンプル数を表示
                for i, rule in enumerate(rules, 1):
                    print(f"  ルール{i}: サンプル数={rule['sample_size']}, 利用意向率={rule['intention_rate']:.1%}")
                
                if len(practical_rules) > best_practical_rules:
                    best_practical_rules = len(practical_rules)
                    best_analyzer = analyzer
                    print(f"新しい最良設定を発見: {len(practical_rules)}個の実用的ルール")
                elif best_analyzer is None:
                    # 最初の有効な設定を保存
                    best_analyzer = analyzer
                    best_practical_rules = len(practical_rules)
                    print(f"最初の有効な設定を保存: {len(practical_rules)}個の実用的ルール")
            else:
                print("ルールの抽出に失敗")
        else:
            print("決定木の学習に失敗")
    
    if best_analyzer is None:
        print("有効な決定木を作成できませんでした")
        return
    
    print(f"\n=== 最良設定での分析完了 ===")
    print(f"設定: max_depth={best_analyzer.max_depth}, min_samples_split={best_analyzer.min_samples_split}, min_samples_leaf={best_analyzer.min_samples_leaf}")
    
    # 最終的なルールの抽出
    final_rules = best_analyzer.extract_rules(data)
    
    if not final_rules:
        print("最終的なルールの抽出に失敗しました")
        return
    
    # 結果の表示
    print(f"\n=== CART分析完了 ===")
    print(f"抽出されたルール数: {len(final_rules)}")
    
    # ルールの詳細表示
    print("\n=== 抽出されたルール ===")
    for i, rule in enumerate(final_rules, 1):
        print(f"\nルール{i}:")
        print(f"  条件: {rule['condition']}")
        print(f"  利用意向率: {rule['intention_rate']:.1%}")
        print(f"  サンプル数: {rule['sample_size']}")
        print(f"  深さ: {rule['depth']}")
        print(f"  信頼度: {rule['confidence']:.3f}")
        print(f"  サポート: {rule['support']:.1%}")
    
    # 出力ディレクトリの作成
    table_path = Path(table_file)
    report_path = Path(report_file)
    tree_text_path = Path(tree_text_file)
    tree_viz_path = Path(tree_viz_file)
    
    for path in [table_path, report_path, tree_text_path, tree_viz_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    # テーブル（rawdata）の保存
    try:
        rules_df = pd.DataFrame(final_rules)
        rules_df.to_csv(table_file, index=False, encoding='utf-8')
        print(f"\n分析結果テーブルを保存しました: {table_file}")
    except Exception as e:
        print(f"テーブル保存エラー: {e}")
    
    # レポート（Markdown）の保存
    try:
        report_content = generate_markdown_report(final_rules, data, target_column, top_numerical_features, ranking_df, best_analyzer)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"分析レポートを保存しました: {report_file}")
    except Exception as e:
        print(f"レポート保存エラー: {e}")
    
    # 決定木の性能評価
    print("\n=== 決定木の性能評価 ===")
    evaluation = best_analyzer.evaluate_tree(data, top_numerical_features, target_column)
    
    if evaluation:
        print(f"交差検証スコア: {evaluation['cv_mean']:.3f} (±{evaluation['cv_std']*2:.3f})")
        print(f"予測精度: {evaluation['accuracy']:.3f}")
    
    # 特徴量重要度の表示
    feature_importance = best_analyzer.get_feature_importance()
    if feature_importance:
        print(f"\n=== 特徴量重要度（決定木） ===")
        for feature, importance in feature_importance.items():
            print(f"{feature}: {importance:.3f}")
    
    # ルールの要約
    rules_summary = best_analyzer.get_rules_summary()
    if rules_summary:
        print(f"\n=== ルールの要約 ===")
        print(f"総ルール数: {rules_summary['total_rules']}")
        print(f"実用的なルール: {rules_summary['practical_rules']}個")
        print(f"部分的に実用的: {rules_summary['partial_rules']}個")
        print(f"実用性が低い: {rules_summary['low_rules']}個")
        print(f"平均利用意向率: {rules_summary['avg_intention_rate']:.1%}")
        print(f"平均サンプル数: {rules_summary['avg_sample_size']:.1f}")
    
    # 決定木のテキスト出力
    try:
        tree_text = best_analyzer.export_tree_text(tree_text_file)
        print(f"決定木のテキスト出力を保存しました: {tree_text_file}")
    except Exception as e:
        print(f"決定木テキスト出力エラー: {e}")
    
    # 決定木の可視化
    try:
        best_analyzer.visualize_tree(tree_viz_file)
        print(f"決定木の可視化を保存しました: {tree_viz_file}")
    except Exception as e:
        print(f"決定木可視化エラー: {e}")
    
    # ルールの実用性評価
    print("\n=== ルールの実用性評価 ===")
    for i, rule in enumerate(final_rules, 1):
        sample_size = rule['sample_size']
        if sample_size >= 5:
            print(f"ルール{i}: 実用的 (サンプル数: {sample_size})")
        elif sample_size >= 3:
            print(f"ルール{i}: 部分的に実用的 (サンプル数: {sample_size})")
        else:
            print(f"ルール{i}: 実用性が低い (サンプル数: {sample_size})")

def generate_markdown_report(rules: list, data: pd.DataFrame, target_column: str, 
                           features: list, ranking_df: pd.DataFrame, analyzer: CARTAnalyzer) -> str:
    """Markdownレポートを生成"""
    
    # 基本情報
    report = f"""# CART分析（決定木）レポート v2

## 分析概要
- **分析日時**: {pd.Timestamp.now().strftime('%Y年%m月%d日 %H:%M:%S')}
- **データファイル**: 前処理済みデータ
- **サンプル数**: {len(data)}件
- **特徴量数**: {len(features)}個
- **目的変数**: {target_column}
- **決定木設定**: 最大深さ{analyzer.max_depth}、最小分割サンプル数{analyzer.min_samples_split}、最小葉ノードサンプル数{analyzer.min_samples_leaf}

## 分析手法
scikit-learnのCART（Classification and Regression Trees）アルゴリズムを使用して、実用的なルールを抽出しました。
より実用的なルールを作成するため、決定木の設定を調整しました。

## 使用した特徴量

単変量分析の上位4個の数値型特徴量を使用しました：

"""
    
    # 使用特徴量の詳細
    for i, feature in enumerate(features, 1):
        feature_info = ranking_df[ranking_df['feature'] == feature].iloc[0]
        report += f"""
### {i}. {feature}
- **重要度スコア**: {feature_info['importance_score']:.3f}
- **AUC**: {feature_info['auc']:.3f}
- **相関係数**: {feature_info['correlation']:.3f}
- **解釈**: {feature_info['interpretation']}
"""
    
    # 抽出されたルール
    report += f"""
## 抽出されたルール

### ルール概要
- **総ルール数**: {len(rules)}個
- **平均利用意向率**: {np.mean([r['intention_rate'] for r in rules]):.1%}
- **平均サンプル数**: {np.mean([r['sample_size'] for r in rules]):.1f}件
- **平均サポート**: {np.mean([r['support'] for r in rules]):.1%}

### 詳細ルール
"""
    
    # 各ルールの詳細
    for i, rule in enumerate(rules, 1):
        report += f"""
#### ルール{i}
- **条件**: {rule['condition']}
- **利用意向率**: {rule['intention_rate']:.1%}
- **サンプル数**: {rule['sample_size']}件
- **深さ**: {rule['depth']}
- **信頼度**: {rule['confidence']:.3f}
- **サポート**: {rule['support']:.1%}
- **実用性**: {'実用的' if rule['sample_size'] >= 5 else '部分的に実用的' if rule['sample_size'] >= 3 else '実用性が低い'}

"""
    
    # ルールの解釈
    report += f"""
## ルールの解釈

### 最も効果的なルール
"""
    
    # 利用意向率でソート
    sorted_rules = sorted(rules, key=lambda x: x['intention_rate'], reverse=True)
    
    if sorted_rules:
        best_rule = sorted_rules[0]
        report += f"""
**最高利用意向率のルール**:
- 条件: {best_rule['condition']}
- 利用意向率: {best_rule['intention_rate']:.1%}
- サンプル数: {best_rule['sample_size']}件
- 実用性: {'実用的' if best_rule['sample_size'] >= 5 else '部分的に実用的' if best_rule['sample_size'] >= 3 else '実用性が低い'}

このルールは、特定の条件を満たす顧客グループで最も高い利用意向を示しています。
"""
    
    # 深さ別の分析
    report += f"""
## 深さ別の分析

### 深さ1のルール（単一条件）
"""
    
    depth_1_rules = [r for r in rules if r['depth'] == 1]
    for rule in depth_1_rules:
        report += f"""
- **{rule['condition']}**: 利用意向率 {rule['intention_rate']:.1%} (サンプル数: {rule['sample_size']})
"""
    
    if len(depth_1_rules) == 0:
        report += "深さ1のルールはありません。"
    
    # 実用性の評価
    report += f"""
## 実用性の評価

### サンプル数による分類
- **実用的 (5件以上)**: 統計的に信頼できるルール
- **部分的に実用的 (3-4件)**: 参考程度に使用可能
- **実用性が低い (2件以下)**: 統計的に意味が薄い

### 今回の分析結果
"""
    
    practical_rules = [r for r in rules if r['sample_size'] >= 5]
    partial_rules = [r for r in rules if 3 <= r['sample_size'] < 5]
    low_rules = [r for r in rules if r['sample_size'] < 3]
    
    report += f"""
- **実用的なルール**: {len(practical_rules)}個
- **部分的に実用的なルール**: {len(partial_rules)}個
- **実用性が低いルール**: {len(low_rules)}個

## 重要な発見

### 1. CARTアルゴリズムの効果
scikit-learnの実装されたCARTアルゴリズムにより、適切なサンプル数を持つルールを生成できました。

### 2. 特徴量の組み合わせ効果
複数の特徴量の組み合わせにより、より詳細な顧客セグメンテーションが可能になりました。

### 3. ルールの階層性
深さによる階層的なルール構造により、顧客の属性を段階的に分類できます。

## 今後の分析方針

1. **アンサンブル手法**: ランダムフォレストなどによるルールの安定性向上
2. **ハイパーパラメータチューニング**: より最適な設定の探索
3. **可視化の改善**: インタラクティブな決定木の可視化
4. **ルールの検証**: より大きなデータセットでのルールの妥当性確認
"""
    
    return report

if __name__ == "__main__":
    main()
