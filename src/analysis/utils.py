"""
共通ユーティリティ関数
データ処理と分析に必要な共通機能を提供
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import json

def load_data(file_path: str) -> pd.DataFrame:
    """データファイルを読み込む"""
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def save_results(data: Any, file_path: str, format_type: str = 'csv') -> bool:
    """結果をファイルに保存する"""
    try:
        if format_type == 'csv':
            data.to_csv(file_path, index=False)
        elif format_type == 'json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        return False

def create_summary_report(data: pd.DataFrame, results: Dict) -> str:
    """サマリーレポートをMarkdown形式で作成"""
    report = f"""# アンケート分析レポート

## データ概要
- サンプル数: {len(data)}
- 特徴量数: {len(data.columns)}
- 分析日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 重要特徴量ランキング
"""
    
    if 'ranking' in results:
        report += "\n| 特徴量 | 重要度スコア | 解釈 |\n"
        report += "|--------|--------------|------|\n"
        for item in results['ranking']:
            report += f"| {item['feature']} | {item['score']:.3f} | {item['interpretation']} |\n"
    
    if 'rules' in results:
        report += "\n## 実用的なルール\n"
        for i, rule in enumerate(results['rules'], 1):
            report += f"\n### ルール{i}\n"
            report += f"- 条件: {rule['condition']}\n"
            report += f"- 利用意向率: {rule['intention_rate']:.1%}\n"
            report += f"- サンプル数: {rule['sample_size']}\n"
    
    return report
