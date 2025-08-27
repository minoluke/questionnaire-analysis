"""
簡易ルール分析実行スクリプト
READMEに記載されている簡易ルール分析（1本のif-else）を実行
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# パスを追加
sys.path.append(str(Path(__file__).parent / 'src'))

from analysis.simple_rule_analysis import SimpleRuleAnalyzer

def main(input_file=None, ranking_file=None, table_file=None, report_file=None):
    """簡易ルール分析メイン実行関数"""
    print("=== 簡易ルール分析（1本のif-else）を開始します ===")
    
    # デフォルトパスの設定
    if input_file is None:
        input_file = "data/processed/preprocessed_data_real.csv"
    if ranking_file is None:
        ranking_file = "output/tables/univariate_ranking_table.csv"
    if table_file is None:
        table_file = "output/tables/simple_rule_table.csv"
    if report_file is None:
        report_file = "output/reports/simple_rule_report.md"
    
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
    
    # 使用する特徴量の選択（上位4つの数値型特徴量を使用）
    print("上位4つの数値型特徴量の全ペア組み合わせで分析します。")
    available_features = ranking_df[
        (ranking_df['type'] == 'numerical') & 
        (ranking_df['feature'] != target_column)
    ].head(4)['feature'].tolist()
    
    print(f"\n=== 使用する特徴量 ===")
    print(f"特徴量: {available_features}")
    
    # 簡易ルール分析の実行
    print("\n=== 簡易ルール分析を実行中 ===")
    
    analyzer = SimpleRuleAnalyzer(min_samples=3)
    best_rule = analyzer.find_best_rule(data, available_features, target_column)
    
    if not best_rule:
        print("有効なルールが見つかりませんでした")
        return
    
    # 結果の表示
    print(f"\n=== 簡易ルール分析完了 ===")
    print(f"最良ルール: {best_rule['rule_condition']}")
    print(f"F1スコア: {best_rule['f1_score']:.3f}")
    print(f"精度: {best_rule['accuracy']:.3f}")
    print(f"ルール内利用意向率: {best_rule['rule_intention_rate']:.1%}")
    print(f"ルール外利用意向率: {best_rule['other_intention_rate']:.1%}")
    print(f"全体利用意向率: {best_rule['overall_intention_rate']:.1%}")
    print(f"Lift: {best_rule['lift']:.3f}")
    
    # 出力ディレクトリの作成
    table_path = Path(table_file)
    report_path = Path(report_file)
    
    for path in [table_path, report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    # テーブル（rawdata）の保存
    try:
        rules_df = analyzer.export_rules_to_dataframe()
        rules_df.to_csv(table_file, index=False, encoding='utf-8')
        print(f"\n分析結果テーブルを保存しました: {table_file}")
    except Exception as e:
        print(f"テーブル保存エラー: {e}")
    
    # レポート（Markdown）の保存
    try:
        report_content = generate_markdown_report(best_rule, analyzer, data, target_column, available_features, ranking_df)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"分析レポートを保存しました: {report_file}")
    except Exception as e:
        print(f"レポート保存エラー: {e}")
    
    # ルールの要約
    rules_summary = analyzer.get_rule_summary()
    if rules_summary:
        print(f"\n=== ルールの要約 ===")
        print(f"総ルール数: {rules_summary['total_rules']}")
        print(f"最良F1スコア: {rules_summary['best_f1_score']:.3f}")
        print(f"平均F1スコア: {rules_summary['avg_f1_score']:.3f}")
        print(f"平均Lift: {rules_summary['avg_lift']:.3f}")
    
    # 上位ルールの表示
    top_rules = []
    if rules_summary.get('top_rules'):
        top_rules = rules_summary['top_rules']
        print(f"\n=== 上位5個のルール ===")
        for i, rule in enumerate(top_rules, 1):
            print(f"\n{i}. {rule['rule_condition']}")
            print(f"   F1スコア: {rule['f1_score']:.3f}")
            print(f"   利用意向率: {rule['rule_intention_rate']:.1%}")
            print(f"   サンプル数: {rule['rule_samples']}")
    
    # 可視化の実行
    print("\n=== 可視化を生成中 ===")
    try:
        from visualize_rules import visualize_rules
        visualize_rules(data, top_rules[:3])
        print("可視化が完了しました: output/visualizations/rule_visualization.png")
    except Exception as e:
        print(f"可視化エラー: {e}")

def generate_markdown_report(best_rule: dict, analyzer: SimpleRuleAnalyzer, 
                           data: pd.DataFrame, target_column: str, 
                           features: list, ranking_df: pd.DataFrame) -> str:
    """Markdownレポートを生成"""
    
    # 基本情報
    report = f"""# 簡易ルール分析レポート

## 分析概要
- **分析日時**: {pd.Timestamp.now().strftime('%Y年%m月%d日 %H:%M:%S')}
- **データファイル**: 前処理済みデータ
- **サンプル数**: {len(data)}件
- **特徴量数**: {len(features)}個
- **目的変数**: {target_column}
- **最小サンプル数**: {analyzer.min_samples}件

## 分析手法
READMEに記載されている簡易ルール分析（1本のif-else）を実装しました。

**目的**: 説明しやすい1行ルールを作る
```
if (特徴量1 > t1) AND (特徴量2 > t2) then 利用意向=あり else なし
```

**やり方**:
1. しきい値候補を決める（各指標の分位点や中間値）
2. すべての組合せ (t1, t2) を試し、**精度（または F1）**が最大の組を選ぶ
3. 該当人数が少なすぎるルールは除外（例：該当者 ≥ 3）

## 使用した特徴量

"""
    
    # 使用特徴量の詳細
    for i, feature in enumerate(features, 1):
        if feature in ranking_df['feature'].values:
            feature_info = ranking_df[ranking_df['feature'] == feature].iloc[0]
            report += f"""
### {i}. {feature}
- **重要度スコア**: {feature_info['importance_score']:.3f}
- **AUC**: {feature_info['auc']:.3f}
- **相関係数**: {feature_info['correlation']:.3f}
- **解釈**: {feature_info['interpretation']}
"""
        else:
            report += f"""
### {i}. {feature}
- **重要度スコア**: 不明
- **AUC**: 不明
- **相関係数**: 不明
- **解釈**: 不明
"""
    
    # 上位ルールの詳細
    rules_summary = analyzer.get_rule_summary()
    top_rules = rules_summary.get('top_rules', []) if rules_summary else []
    
    report += f"""
## 発見されたルール

### 上位5個のルール

"""
    
    # 上位5個のルールを追加
    for i, rule in enumerate(top_rules[:5], 1):
        report += f"""
#### {i}. {rule['rule_condition']}
- **F1スコア**: {rule['f1_score']:.3f}
- **ルール内利用意向率**: {rule['rule_intention_rate']:.1%} ({rule['rule_samples']}件)
- **ルール外利用意向率**: {rule['other_intention_rate']:.1%} ({rule['other_samples']}件)
- **精度**: {rule['accuracy']:.3f}
- **Lift**: {rule['lift']:.3f}
"""

    # 最良ルールの詳細
    report += f"""

## 最良ルール詳細

### 最終ルール
**{best_rule['rule_condition']}**

### 性能指標

**ルール内の利用意向率** = {best_rule['rule_intention_rate']:.1%} ({best_rule['rule_samples']}件)

**それ以外の利用意向率** = {best_rule['other_intention_rate']:.1%} ({best_rule['other_samples']}件)

**全体利用意向率** = {best_rule['overall_intention_rate']:.1%}

**全体精度** = {best_rule['accuracy']:.3f}

**F1スコア** = {best_rule['f1_score']:.3f}

**Lift** = {best_rule['lift']:.3f} (ルール内意向率 ÷ 全体意向率)

## ルールの解釈

このルールは、特定の条件を満たす顧客グループで利用意向が高くなることを示しています。

- **条件**: {best_rule['rule_condition']}
- **効果**: ルール内の利用意向率は、全体の{best_rule['lift']:.1f}倍
- **信頼性**: F1スコア{best_rule['f1_score']:.3f}で、精度{best_rule['accuracy']:.3f}

## 統計的検証

### サンプル数の妥当性
- **ルール内サンプル数**: {best_rule['rule_samples']}件
- **ルール外サンプル数**: {best_rule['other_samples']}件
- **最小サンプル数要件**: {analyzer.min_samples}件以上 ✅

### 性能指標
- **精度**: {best_rule['accuracy']:.3f}
- **再現率**: {best_rule['recall']:.3f}
- **F1スコア**: {best_rule['f1_score']:.3f}
- **Lift**: {best_rule['lift']:.3f}

## 注意事項

**小標本のため参考値**。簡単な再チェック（LOOCV やブートストラップ）を実施することを推奨します。

## 今後の改善点

1. **交差検証**: LOOCVによるルールの安定性確認
2. **ブートストラップ**: 信頼区間の計算
3. **特徴量選択**: より効果的な特徴量の組み合わせの探索
4. **しきい値最適化**: より細かいしきい値の調整
"""
    
    return report

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main()
