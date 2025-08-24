"""
簡易ルール分析（決定木）実行スクリプト v2
より実用的なルールを作成するため、決定木の設定を調整
"""

import sys
import os
from pathlib import Path
import pandas as pd

# パスを追加
sys.path.append(str(Path(__file__).parent / 'src'))

from analysis.simple_rules import SimpleRules

def main():
    """簡易ルール分析メイン実行関数"""
    print("=== 簡易ルール分析（決定木）v2 を開始します ===")
    
    # 前処理済みデータの読み込み
    input_file = "data/processed/preprocessed_data.csv"
    ranking_file = "output/tables/univariate_ranking_table.csv"
    table_file = "output/tables/simple_rules_table_v2.csv"
    report_file = "output/reports/simple_rules_report_v2.md"
    
    print(f"入力ファイル: {input_file}")
    print(f"単変量分析結果: {ranking_file}")
    print(f"テーブル出力: {table_file}")
    print(f"レポート出力: {report_file}")
    
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
    
    # 上位特徴量の選択（数値型のみ、重要度上位3個に制限）
    top_numerical_features = ranking_df[
        (ranking_df['type'] == 'numerical') & 
        (ranking_df['feature'] != target_column)
    ].head(3)['feature'].tolist()
    
    print(f"\n=== 使用する特徴量 ===")
    print(f"上位3個の数値型特徴量: {top_numerical_features}")
    
    # 簡易ルール分析の実行（深さ1に制限）
    print("\n=== 簡易ルール分析を実行中 ===")
    rules_analyzer = SimpleRules(max_depth=1, min_samples_split=3)
    
    # 決定木の作成
    if not rules_analyzer.create_decision_tree(data, top_numerical_features, target_column):
        print("決定木の作成に失敗しました")
        return
    
    # ルールの抽出
    rules = rules_analyzer.extract_rules(data, top_numerical_features, target_column)
    
    if not rules:
        print("ルールの抽出に失敗しました")
        return
    
    print(f"ルール抽出完了: {len(rules)}個のルール")
    
    # 結果の表示
    print(f"\n=== 簡易ルール分析完了 ===")
    print(f"抽出されたルール数: {len(rules)}")
    
    # ルールの詳細表示
    print("\n=== 抽出されたルール ===")
    for i, rule in enumerate(rules, 1):
        print(f"\nルール{i}:")
        print(f"  条件: {rule['condition']}")
        print(f"  利用意向率: {rule['intention_rate']:.1%}")
        print(f"  サンプル数: {rule['sample_size']}")
        print(f"  信頼度: {rule['confidence']:.3f}")
    
    # 出力ディレクトリの作成
    table_path = Path(table_file)
    report_path = Path(report_file)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # テーブル（rawdata）の保存
    try:
        rules_df = pd.DataFrame(rules)
        rules_df.to_csv(table_file, index=False, encoding='utf-8')
        print(f"\n分析結果テーブルを保存しました: {table_file}")
    except Exception as e:
        print(f"テーブル保存エラー: {e}")
    
    # レポート（Markdown）の保存
    try:
        report_content = generate_markdown_report(rules, data, target_column, top_numerical_features, ranking_df)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"分析レポートを保存しました: {report_file}")
    except Exception as e:
        print(f"レポート保存エラー: {e}")
    
    # 決定木の性能評価
    print("\n=== 決定木の性能評価 ===")
    cv_scores = rules_analyzer.evaluate_tree(data, top_numerical_features, target_column)
    if cv_scores is not None:
        print(f"交差検証スコア: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
    
    # 特徴量重要度の表示
    feature_importance = rules_analyzer.get_feature_importance(top_numerical_features)
    if feature_importance:
        print(f"\n=== 特徴量重要度（決定木） ===")
        for feature, importance in feature_importance.items():
            print(f"{feature}: {importance:.3f}")
    
    # ルールの実用性評価
    print("\n=== ルールの実用性評価 ===")
    for i, rule in enumerate(rules, 1):
        sample_size = rule['sample_size']
        if sample_size >= 5:
            print(f"ルール{i}: 実用的 (サンプル数: {sample_size})")
        elif sample_size >= 3:
            print(f"ルール{i}: 部分的に実用的 (サンプル数: {sample_size})")
        else:
            print(f"ルール{i}: 実用性が低い (サンプル数: {sample_size})")

def generate_markdown_report(rules: list, data: pd.DataFrame, target_column: str, 
                           features: list, ranking_df: pd.DataFrame) -> str:
    """Markdownレポートを生成"""
    import numpy as np
    
    # 基本情報
    report = f"""# 簡易ルール分析（決定木）レポート v2

## 分析概要
- **分析日時**: {pd.Timestamp.now().strftime('%Y年%m月%d日 %H:%M:%S')}
- **データファイル**: 前処理済みデータ
- **サンプル数**: {len(data)}件
- **使用特徴量数**: {len(features)}個
- **目的変数**: {target_column}
- **決定木設定**: 最大深さ1、最小分割サンプル数3

## 改善点
前回の分析では各ルールのサンプル数が1件ずつとなり、実用性に欠けていました。
今回の分析では以下の改善を行いました：

1. **決定木の深さを1に制限**: よりシンプルで解釈しやすいルール
2. **特徴量数を3個に制限**: 過学習を防ぎ、重要な特徴量に集中
3. **最小分割サンプル数を3に設定**: 統計的に意味のあるサンプル数を確保

## 使用した特徴量

単変量分析の上位3個の数値型特徴量を使用しました：

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

### 詳細ルール
"""
    
    # 各ルールの詳細
    for i, rule in enumerate(rules, 1):
        report += f"""
#### ルール{i}
- **条件**: {rule['condition']}
- **利用意向率**: {rule['intention_rate']:.1%}
- **サンプル数**: {rule['sample_size']}件
- **信頼度**: {rule['confidence']:.3f}
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

## 今後の分析方針

1. **データ拡充**: より多くのサンプルでルールの安定性を確認
2. **特徴量選択**: より効果的な特徴量の組み合わせを探索
3. **アンサンブル手法**: 複数の決定木によるルールの安定性向上
4. **可視化**: 決定木の構造とルールの関係性の可視化
"""
    
    return report

if __name__ == "__main__":
    main()
