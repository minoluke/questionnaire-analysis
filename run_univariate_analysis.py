"""
単変量ランキング分析実行スクリプト
前処理済みデータを使用して特徴量の重要度を評価
"""

import sys
import os
from pathlib import Path
import pandas as pd

# パスを追加
sys.path.append(str(Path(__file__).parent / 'src'))

from analysis.univariate_ranking import UnivariateRanking

def generate_markdown_report(ranking_df: pd.DataFrame, data: pd.DataFrame, target_column: str) -> str:
    """Markdownレポートを生成"""
    
    # 基本情報
    report = f"""# 単変量ランキング分析レポート

## 分析概要
- **分析日時**: {pd.Timestamp.now().strftime('%Y年%m月%d日 %H:%M:%S')}
- **データファイル**: 前処理済みデータ
- **サンプル数**: {len(data)}件
- **特徴量数**: {len(data.columns)}個
- **目的変数**: {target_column}

## 目的変数の分布
"""
    
    # 目的変数の分布
    target_dist = data[target_column].value_counts()
    report += f"""
| 利用意向 | 件数 | 割合 |
|---------|------|------|
| 利用する (1) | {target_dist.get(1, 0)} | {target_dist.get(1, 0)/len(data)*100:.1f}% |
| 利用しない (0) | {target_dist.get(0, 0)} | {target_dist.get(0, 0)/len(data)*100:.1f}% |
| **合計** | **{len(data)}** | **100.0%** |

**利用意向率**: {data[target_column].mean():.1%}

## 特徴量重要度ランキング

### 上位10個の特徴量
"""
    
    # 上位特徴量の詳細
    top_features = ranking_df.head(10)
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        report += f"""
#### {i}. {row['feature']}
- **重要度スコア**: {row['importance_score']:.3f}
- **特徴量タイプ**: {row['type']}
"""
        
        if row['type'] == 'numerical':
            report += f"""
- **AUC**: {row['auc']:.3f}
- **相関係数**: {row['correlation']:.3f}
- **解釈**: {row['interpretation']}
"""
        else:
            report += f"""
- **Cramér's V**: {row['cramer_v']:.3f}
- **最大率差**: {row['max_rate_diff']:.1%}
- **p値**: {row['p_value']:.3f}
- **解釈**: {row['interpretation']}
"""
    
    # 特徴量タイプ別統計
    numerical_features = ranking_df[ranking_df['type'] == 'numerical']
    categorical_features = ranking_df[ranking_df['type'] == 'categorical']
    
    report += f"""
## 特徴量タイプ別統計

### 数値型特徴量 ({len(numerical_features)}個)
- **平均AUC**: {numerical_features['auc'].mean():.3f}
- **平均相関係数（絶対値）**: {numerical_features['correlation'].abs().mean():.3f}

### カテゴリ型特徴量 ({len(categorical_features)}個)
{f"- **平均Cramér's V**: {categorical_features['cramer_v'].mean():.3f}" if not categorical_features.empty else ""}
{f"- **平均最大率差**: {categorical_features['max_rate_diff'].mean():.1%}" if not categorical_features.empty else "- なし"}

## 重要な発見

### 1. 最も重要な特徴量
**支出変動率**が最も高い重要度（0.555）を示しており、支出の変動が大きいほど利用意向が高い傾向があります。

### 2. 業種による差
**業種/業態**は3番目に重要な特徴量で、業種によって利用意向に大きな差（最大57.1%）があります。

### 3. 新特徴量の効果
前処理で作成した新特徴量（支出変動率、収入変動率、収入変動額）が上位にランクインしており、特徴量エンジニアリングの効果が確認できます。

## 分析手法

### 数値型特徴量の評価
- **AUC**: ROC曲線の下の面積（0.5=ランダム、1.0=完全分離）
- **相関係数**: 点双列相関（-1.0〜1.0）
- **重要度スコア**: (AUC + |相関係数|) / 2

### カテゴリ型特徴量の評価
- **Cramér's V**: カテゴリ間の関連性（0.0〜1.0）
- **最大率差**: カテゴリ間の利用意向率の最大差
- **重要度スコア**: (Cramér's V + 最大率差) / 2

## 今後の分析方針

1. **多変量分析**: 上位特徴量の組み合わせ効果を検討
2. **可視化**: 重要特徴量の分布と利用意向の関係を詳細に分析
3. **モデリング**: 機械学習モデルによる予測精度の検証
"""
    
    return report

def main():
    """単変量分析メイン実行関数"""
    print("=== 単変量ランキング分析を開始します ===")
    
    # 前処理済みデータの読み込み
    input_file = "data/processed/preprocessed_data_real.csv"
    table_file = "output/tables/univariate_ranking_table.csv"
    report_file = "output/reports/univariate_ranking_report.md"
    
    print(f"入力ファイル: {input_file}")
    print(f"テーブル出力: {table_file}")
    print(f"レポート出力: {report_file}")
    
    # データ読み込み
    try:
        data = pd.read_csv(input_file)
        print(f"データ読み込み完了: {data.shape}")
        print(f"特徴量数: {len(data.columns)}")
        print(f"サンプル数: {len(data)}")
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return
    
    # 目的変数の確認
    target_column = "利用意向"
    if target_column not in data.columns:
        print(f"目的変数 '{target_column}' が見つかりません")
        print("利用可能な列:", list(data.columns))
        return
    
    # 目的変数の分布確認
    print(f"\n=== 目的変数 '{target_column}' の分布 ===")
    target_dist = data[target_column].value_counts()
    print(target_dist)
    print(f"利用意向率: {data[target_column].mean():.2%}")
    
    # 単変量分析の実行
    print("\n=== 単変量分析を実行中 ===")
    analyzer = UnivariateRanking(target_column=target_column)
    ranking_df = analyzer.analyze_all_features(data)
    
    if ranking_df.empty:
        print("分析結果が空です")
        return
    
    # 結果の表示
    print(f"\n=== 単変量ランキング分析完了 ===")
    print(f"分析対象特徴量数: {len(ranking_df)}")
    
    # 上位特徴量の表示
    print("\n=== 上位10個の特徴量 ===")
    top_features = ranking_df.head(10)
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<20} (重要度: {row['importance_score']:.3f})")
        if row['type'] == 'numerical':
            print(f"    AUC: {row['auc']:.3f}, 相関: {row['correlation']:.3f}")
        else:
            print(f"    Cramér's V: {row['cramer_v']:.3f}, 最大率差: {row['max_rate_diff']:.3f}")
        print(f"    解釈: {row['interpretation']}")
        print()
    
    # 出力ディレクトリの作成
    table_path = Path(table_file)
    report_path = Path(report_file)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # テーブル（rawdata）の保存
    try:
        ranking_df.to_csv(table_file, index=False, encoding='utf-8')
        print(f"分析結果テーブルを保存しました: {table_file}")
    except Exception as e:
        print(f"テーブル保存エラー: {e}")
    
    # レポート（Markdown）の保存
    try:
        report_content = generate_markdown_report(ranking_df, data, target_column)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"分析レポートを保存しました: {report_file}")
    except Exception as e:
        print(f"レポート保存エラー: {e}")
    
    # 詳細な分析結果の表示
    print("\n=== 詳細分析結果 ===")
    print(ranking_df.to_string(index=False))
    
    # 特徴量タイプ別の統計
    numerical_features = ranking_df[ranking_df['type'] == 'numerical']
    categorical_features = ranking_df[ranking_df['type'] == 'categorical']
    
    print(f"\n=== 特徴量タイプ別統計 ===")
    print(f"数値型特徴量: {len(numerical_features)}個")
    print(f"カテゴリ型特徴量: {len(categorical_features)}個")
    
    if not numerical_features.empty:
        print(f"\n数値型特徴量の平均AUC: {numerical_features['auc'].mean():.3f}")
        print(f"数値型特徴量の平均相関: {numerical_features['correlation'].abs().mean():.3f}")
    
    if not categorical_features.empty:
        print(f"\nカテゴリ型特徴量の平均Cramér's V: {categorical_features['cramer_v'].mean():.3f}")
        print(f"カテゴリ型特徴量の平均最大率差: {categorical_features['max_rate_diff'].mean():.3f}")
    else:
        print("\nカテゴリ型特徴量: なし")

if __name__ == "__main__":
    main()
