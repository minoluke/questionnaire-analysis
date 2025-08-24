# アンケート分析システム

アンケートデータから特徴量の重要度を分析し、実用的なルールを抽出するシステムです。

## 機能

### 1. データ前処理

- 欠損値の処理
- データ型の変換
- 新特徴量の作成
  - **収入変動額**: 最高月収 - 最低月収
  - **収入変動率**: 収入変動額 / 平均月収
  - **支出変動率**: (最高支出 - 最低支出) / ((最高支出 + 最低支出) / 2)
- カテゴリ変数のエンコーディング

### 2. 単変量ランキング分析

- 各特徴量の重要度を評価
- **数値型特徴量**: AUC、相関係数、効果量を計算
- **カテゴリ型特徴量**: Cramer's V、最大率差を計算
- 特徴量のランキングを作成

### 3. 簡易ルール分析（1 本の if-else）

**目的**
説明しやすい 1 行ルールを作る：
if (自由資産率 > t1) AND (平均月収 > t2) then 利用意向=あり else なし

**やり方（短く）**

しきい値候補を決める（各指標の分位点や中間値）。

すべての組合せ (t1, t2) を試し、**精度（または F1）**が最大の組を選ぶ。

該当人数が少なすぎるルールは除外（例：該当者 ≥ 3）。

**出力（レポートに載せるもの）**

最終ルール：自由資産率 > ○○ かつ 平均月収 > ○○

数字：

ルール内の利用意向率 = a/b

それ以外の利用意向率 = c/d

全体精度（or F1）／必要なら Lift（= ルール内意向率 ÷ 全体意向率）

注記：小標本のため参考値。簡単な再チェック（LOOCV やブートストラップ）を実施。

## ディレクトリ構成

```
questionnaire-analysis/
├── data/                          # データファイル
│   ├── raw/                       # 生データ
│   │   └── sprint_data - dummy.csv
│   └── processed/                 # 処理済みデータ
│       └── preprocessed_data.csv
├── src/                           # ソースコード
│   └── analysis/                  # 分析モジュール
│       ├── preprocessing.py       # データ前処理
│       ├── univariate_ranking.py # 単変量分析
│       └── utils.py              # ユーティリティ
├── output/                        # 出力結果
│   ├── tables/                    # テーブル形式の結果
│   │   ├── univariate_ranking_table.csv
│   │   └── manual_rules_table.csv
│   └── reports/                   # レポート
│       ├── univariate_ranking_report.md
│       └── manual_rules_report.md
├── archive/                       # アーカイブ
│   ├── simple_rules/             # 旧simple rules分析
│   └── decision_trees/           # 旧決定木分析
├── config/                        # 設定ファイル
│   └── analysis_config.yaml
├── preprocess.py                  # 前処理実行スクリプト
├── run_univariate_analysis.py    # 単変量分析実行スクリプト
└── main.py                        # メイン実行スクリプト
```

## 使用方法

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. データの準備

- `data/raw/` ディレクトリに分析したい CSV ファイルを配置
- 目的変数（利用意向）の列名は `利用意向` を使用

### 3. 分析の実行

#### データ前処理

```bash
python preprocess.py
```

#### 単変量ランキング分析

```bash
python run_univariate_analysis.py
```

## 出力ファイル

### テーブル（rawdata）

- **単変量分析結果**: `output/tables/univariate_ranking_table.csv`

### レポート（Markdown）

- **単変量分析レポート**: `output/reports/univariate_ranking_report.md`

## 設定

`config/analysis_config.yaml` で分析パラメータを調整できます。

## 必要なデータ形式

- CSV ファイル形式
- 目的変数（利用意向）は 0/1 の二値
- 数値型・カテゴリ型の特徴量に対応

## 技術仕様

### 使用ライブラリ

- **pandas**: データ処理
- **numpy**: 数値計算
- **scikit-learn**: 機械学習（単変量分析）
- **scipy**: 統計分析
- **matplotlib**: 可視化

### 分析手法

- **単変量分析**: AUC、相関係数、Cramer's V、最大率差
- **統計的検証**: 信頼区間、サンプル数による実用性評価
