"""
アンケート分析メインスクリプト
単変量ランキングと簡易ルール分析を実行
"""

import pandas as pd
import sys
import os
from pathlib import Path

# パスを追加
sys.path.append(str(Path(__file__).parent / 'src'))

from analysis.univariate_ranking import UnivariateRanking
from analysis.simple_rules import SimpleRules
from analysis.utils import load_data, save_results, create_summary_report

def main():
    """メイン実行関数"""
    print("=== アンケート分析を開始します ===")
    
    # データファイルのパス（前処理済みデータを優先）
    raw_data_file = "data/raw/sprint_data - dummy.csv"
    processed_data_file = "data/processed/preprocessed_data.csv"
    
    # 前処理済みデータがある場合はそれを使用、なければ生データを使用
    if os.path.exists(processed_data_file):
        data_file = processed_data_file
        print(f"前処理済みデータを使用: {data_file}")
    else:
        data_file = raw_data_file
        print(f"生データを使用: {data_file}")
        print("前処理済みデータがない場合は、preprocess.pyを先に実行してください")
    
    # データの読み込み
    print(f"データファイルを読み込み中: {data_file}")
    data = load_data(data_file)
    
    if data is None:
        print("データの読み込みに失敗しました")
        return
    
    print(f"データ読み込み完了: {len(data)}行, {len(data.columns)}列")
    print(f"列名: {list(data.columns)}")
    
    # 目的変数の確認（利用意向）
    target_column = 'intention'
    if target_column not in data.columns:
        print(f"目的変数 '{target_column}' が見つかりません")
        print("利用可能な列名を確認してください")
        return
    
    print(f"目的変数: {target_column}")
    print(f"利用意向の分布: {data[target_column].value_counts().to_dict()}")
    
    # 1. 単変量ランキング分析
    print("\n=== 単変量ランキング分析を実行中 ===")
    ranking_analyzer = UnivariateRanking(target_column=target_column)
    ranking_results = ranking_analyzer.analyze_all_features(data)
    
    if ranking_results.empty:
        print("単変量分析で結果が得られませんでした")
        return
    
    print("単変量分析完了")
    print(f"分析された特徴量数: {len(ranking_results)}")
    
    # 上位特徴量を取得
    top_features = ranking_analyzer.get_top_features(n=3)
    print(f"上位3特徴量: {top_features}")
    
    # 結果を保存
    ranking_file = "output/tables/feature_ranking.csv"
    save_results(ranking_results, ranking_file, 'csv')
    print(f"特徴量ランキングを保存: {ranking_file}")
    
    # 2. 簡易ルール分析
    print("\n=== 簡易ルール分析を実行中 ===")
    rules_analyzer = SimpleRules(max_depth=2, min_samples_split=5)
    
    # 上位特徴量で決定木を作成
    if top_features:
        success = rules_analyzer.create_decision_tree(data, top_features, target_column)
        
        if success:
            print("決定木の作成完了")
            
            # ルールを抽出
            rules = rules_analyzer.extract_rules(data, top_features, target_column)
            
            if rules:
                print(f"ルール抽出完了: {len(rules)}個のルール")
                
                # ルールを保存
                rules_file = "output/tables/decision_rules.json"
                save_results(rules, rules_file, 'json')
                print(f"決定木ルールを保存: {rules_file}")
                
                # ルールの評価
                evaluation = rules_analyzer.evaluate_rules(data, top_features, target_column)
                if evaluation:
                    print(f"ルール評価完了: 平均精度 {evaluation['average_accuracy']:.3f}")
            else:
                print("ルールの抽出に失敗しました")
        else:
            print("決定木の作成に失敗しました")
    
    # 3. サマリーレポートの作成
    print("\n=== サマリーレポートを作成中 ===")
    
    # 結果をまとめる
    results_summary = {
        'ranking': ranking_results.to_dict('records'),
        'rules': rules if 'rules' in locals() else []
    }
    
    # レポートを作成
    report_content = create_summary_report(data, results_summary)
    
    # レポートを保存
    report_file = "output/reports/analysis_summary.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"サマリーレポートを保存: {report_file}")
    
    print("\n=== 分析完了 ===")
    print(f"出力ファイル:")
    print(f"  - 特徴量ランキング: {ranking_file}")
    if 'rules' in locals() and rules:
        print(f"  - 決定木ルール: {rules_file}")
    print(f"  - サマリーレポート: {report_file}")

if __name__ == "__main__":
    main()
