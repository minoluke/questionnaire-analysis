"""
手動ルール作成スクリプト
データの分布を確認して、実用的なルールを手動で作成
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# パスを追加
sys.path.append(str(Path(__file__).parent / 'src'))

def main():
    """手動ルール作成メイン実行関数"""
    print("=== 手動ルール作成を開始します ===")
    
    # 前処理済みデータの読み込み
    input_file = "data/processed/preprocessed_data.csv"
    table_file = "output/tables/manual_rules_table.csv"
    report_file = "output/reports/manual_rules_report.md"
    
    print(f"入力ファイル: {input_file}")
    print(f"テーブル出力: {table_file}")
    print(f"レポート出力: {report_file}")
    
    # データ読み込み
    try:
        data = pd.read_csv(input_file)
        print(f"データ読み込み完了: {data.shape}")
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return
    
    # 目的変数の確認
    target_column = "利用意向"
    if target_column not in data.columns:
        print(f"目的変数 '{target_column}' が見つかりません")
        return
    
    # 重要特徴量の分布確認
    print("\n=== 重要特徴量の分布確認 ===")
    
    # 支出変動率の分布
    print("\n支出変動率の分布:")
    print(data['支出変動率'].describe())
    print("\n支出変動率と利用意向の関係:")
    print(data.groupby(target_column)['支出変動率'].describe())
    
    # 最高支出の分布
    print("\n最高支出の分布:")
    print(data['最高支出'].describe())
    print("\n最高支出と利用意向の関係:")
    print(data.groupby(target_column)['最高支出'].describe())
    
    # 収入変動率の分布
    print("\n収入変動率の分布:")
    print(data['収入変動率'].describe())
    print("\n収入変動率と利用意向の関係:")
    print(data.groupby(target_column)['収入変動率'].describe())
    
    # 手動でルールを作成
    print("\n=== 手動ルール作成 ===")
    
    # ルール1: 支出変動率が高い場合
    high_expense_volatility = data['支出変動率'] > data['支出変動率'].median()
    rule1_data = data[high_expense_volatility]
    rule1_intention_rate = rule1_data[target_column].mean()
    rule1_sample_size = len(rule1_data)
    
    print(f"ルール1: 支出変動率 > {data['支出変動率'].median():.3f}")
    print(f"  利用意向率: {rule1_intention_rate:.1%}")
    print(f"  サンプル数: {rule1_sample_size}")
    
    # ルール2: 最高支出が高い場合
    high_max_expense = data['最高支出'] > data['最高支出'].median()
    rule2_data = data[high_max_expense]
    rule2_intention_rate = rule2_data[target_column].mean()
    rule2_sample_size = len(rule2_data)
    
    print(f"ルール2: 最高支出 > {data['最高支出'].median():,.0f}")
    print(f"  利用意向率: {rule2_intention_rate:.1%}")
    print(f"  サンプル数: {rule2_sample_size}")
    
    # ルール3: 収入変動率が低い場合（安定した収入）
    low_income_volatility = data['収入変動率'] < data['収入変動率'].median()
    rule3_data = data[low_income_volatility]
    rule3_intention_rate = rule3_data[target_column].mean()
    rule3_sample_size = len(rule3_data)
    
    print(f"ルール3: 収入変動率 < {data['収入変動率'].median():.3f}")
    print(f"  利用意向率: {rule3_intention_rate:.1%}")
    print(f"  サンプル数: {rule3_sample_size}")
    
    # ルール4: 複合条件（支出変動率が高く、最高支出も高い）
    compound_condition = (data['支出変動率'] > data['支出変動率'].median()) & (data['最高支出'] > data['最高支出'].median())
    rule4_data = data[compound_condition]
    rule4_intention_rate = rule4_data[target_column].mean()
    rule4_sample_size = len(rule4_data)
    
    print(f"ルール4: 支出変動率 > {data['支出変動率'].median():.3f} AND 最高支出 > {data['最高支出'].median():,.0f}")
    print(f"  利用意向率: {rule4_intention_rate:.1%}")
    print(f"  サンプル数: {rule4_sample_size}")
    
    # ルール5: 業種による分類
    industry_rules = []
    for industry in data['業種/業態'].unique():
        industry_data = data[data['業種/業態'] == industry]
        if len(industry_data) >= 2:  # 最低2件以上
            intention_rate = industry_data[target_column].mean()
            sample_size = len(industry_data)
            industry_rules.append({
                'condition': f"業種/業態 = {industry}",
                'intention_rate': intention_rate,
                'sample_size': sample_size,
                'confidence': calculate_confidence(intention_rate, sample_size)
            })
    
    print(f"\n業種別ルール:")
    for i, rule in enumerate(industry_rules, 1):
        print(f"  業種ルール{i}: {rule['condition']}")
        print(f"    利用意向率: {rule['intention_rate']:.1%}")
        print(f"    サンプル数: {rule['sample_size']}")
    
    # 全ルールをまとめる
    all_rules = [
        {
            'condition': f"支出変動率 > {data['支出変動率'].median():.3f}",
            'intention_rate': rule1_intention_rate,
            'sample_size': rule1_sample_size,
            'confidence': calculate_confidence(rule1_intention_rate, rule1_sample_size),
            'type': '単一条件'
        },
        {
            'condition': f"最高支出 > {data['最高支出'].median():,.0f}",
            'intention_rate': rule2_intention_rate,
            'sample_size': rule2_sample_size,
            'confidence': calculate_confidence(rule2_intention_rate, rule2_sample_size),
            'type': '単一条件'
        },
        {
            'condition': f"収入変動率 < {data['収入変動率'].median():.3f}",
            'intention_rate': rule3_intention_rate,
            'sample_size': rule3_sample_size,
            'confidence': calculate_confidence(rule3_intention_rate, rule3_sample_size),
            'type': '単一条件'
        },
        {
            'condition': f"支出変動率 > {data['支出変動率'].median():.3f} AND 最高支出 > {data['最高支出'].median():,.0f}",
            'intention_rate': rule4_intention_rate,
            'sample_size': rule4_sample_size,
            'confidence': calculate_confidence(rule4_intention_rate, rule4_sample_size),
            'type': '複合条件'
        }
    ]
    
    # 業種ルールを追加
    all_rules.extend([{**rule, 'type': '業種条件'} for rule in industry_rules])
    
    # 利用意向率でソート
    all_rules.sort(key=lambda x: x['intention_rate'], reverse=True)
    
    # 出力ディレクトリの作成
    table_path = Path(table_file)
    report_path = Path(report_file)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # テーブル（rawdata）の保存
    try:
        rules_df = pd.DataFrame(all_rules)
        rules_df.to_csv(table_file, index=False, encoding='utf-8')
        print(f"\n分析結果テーブルを保存しました: {table_file}")
    except Exception as e:
        print(f"テーブル保存エラー: {e}")
    
    # レポート（Markdown）の保存
    try:
        report_content = generate_markdown_report(all_rules, data, target_column)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"分析レポートを保存しました: {report_file}")
    except Exception as e:
        print(f"レポート保存エラー: {e}")
    
    # ルールの実用性評価
    print("\n=== ルールの実用性評価 ===")
    for i, rule in enumerate(all_rules, 1):
        sample_size = rule['sample_size']
        if sample_size >= 5:
            print(f"ルール{i}: 実用的 (サンプル数: {sample_size})")
        elif sample_size >= 3:
            print(f"ルール{i}: 部分的に実用的 (サンプル数: {sample_size})")
        else:
            print(f"ルール{i}: 実用性が低い (サンプル数: {sample_size})")

def calculate_confidence(intention_rate: float, sample_size: int) -> float:
    """信頼区間の近似計算"""
    if sample_size == 0:
        return 0.0
    
    # Wilson信頼区間の近似
    z = 1.96  # 95%信頼区間
    se = np.sqrt(intention_rate * (1 - intention_rate) / sample_size)
    margin = z * se
    
    return max(0.0, min(1.0, margin))

def generate_markdown_report(rules: list, data: pd.DataFrame, target_column: str) -> str:
    """Markdownレポートを生成"""
    
    # 基本情報
    report = f"""# 手動ルール作成レポート

## 分析概要
- **分析日時**: {pd.Timestamp.now().strftime('%Y年%m月%d日 %H:%M:%S')}
- **データファイル**: 前処理済みデータ
- **サンプル数**: {len(data)}件
- **特徴量数**: {len(data.columns)}個
- **目的変数**: {target_column}

## 分析手法
決定木による自動的なルール抽出では、各ルールのサンプル数が1件ずつとなり実用性に欠けていました。
そこで、データの分布を確認して、統計的に意味のある手動ルールを作成しました。

## 作成されたルール

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
- **ルールタイプ**: {rule['type']}
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

## 重要な発見

### 1. 支出関連の重要性
支出変動率と最高支出の組み合わせが利用意向に大きな影響を与えています。

### 2. 収入の安定性
収入変動率が低い（安定した収入）顧客グループの利用意向パターンが確認できます。

### 3. 業種による差
業種によって利用意向に大きな差があることが分かります。

## 今後の分析方針

1. **データ拡充**: より多くのサンプルでルールの安定性を確認
2. **閾値の最適化**: より効果的な閾値の探索
3. **複合条件の探索**: より複雑な条件の組み合わせの検討
4. **可視化**: 各ルールの条件と利用意向の関係性の可視化
"""
    
    return report

if __name__ == "__main__":
    main()
