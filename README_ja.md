# EasyBench

[![Tests](https://github.com/smurak/easybench/actions/workflows/test.yml/badge.svg)](https://github.com/smurak/easybench/actions)
[![Docs](https://readthedocs.org/projects/easybench-ja/badge/?version=latest)](https://easybench.readthedocs.io/ja/)
[![PyPI version](https://badge.fury.io/py/easybench.svg)](https://pypi.org/project/easybench/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[**ドキュメント**](https://easybench.readthedocs.io/ja/)

シンプルで使いやすいPythonベンチマークライブラリ。

[English](README.md) | 日本語

## 特徴

- 3種類のベンチマークスタイル：デコレータ、クラスベース、コマンドライン
- 実行時間とメモリ使用量の測定（[制限あり](#メモリ測定の制限)）
- 豊富な可視化（ボックスプロット、バイオリンプロット、ラインプロット、ヒストグラム、バープロット）
- 高度なオプション：ウォームアップ実行、複数ループ測定、外れ値トリミング
- 異なる入力サイズでの関数性能を比較するパラメータ化ベンチマーク
- pytestライクなフィクスチャとライフサイクルフック（setup/teardown）
- 柔軟な設定：時間/メモリ単位、進捗追跡、フィルタリング
- 複数の出力形式（テキストテーブル、CSV、JSON、pandas.DataFrame）
- カスタム出力に対応する拡張可能なレポーティングシステム

## インストール

```bash
pip install easybench
```

### オプション依存関係

EasyBenchは追加機能のためのオプション依存関係をサポートしています：

```bash
# 可視化と分析サポート付きでインストール
pip install easybench[all]
```

`all` オプションには以下が含まれます：

- `matplotlib`: ベンチマーク結果の可視化とプロット作成用
- `seaborn`: 高度な統計的可視化用
- `pandas`: ベンチマーク結果のDataFrame出力用
- `tqdm`: ベンチマーク実行中の進行状況表示用

## クイックスタート

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/smurak/easybench/blob/main/notebooks/easybench_quickstart.ipynb)

`easybench`を使用する方法は3種類あります：

1. `@bench`デコレータ

    ```python
    from easybench import bench
    
    # 関数パラメータと共に@benchを追加
    @bench(item=123, big_list=lambda: list(range(1_000_000)))
    def insert_first(item, big_list):
        big_list.insert(0, item)
    ```

> [!TIP]  
> 各試行ごとに新しいデータを使用する場合は、そのデータを返す関数またはラムダをパラメータで指定します。  
> (上記の `lambda: list(range(1_000_000))` のように。)

2. `EasyBench`クラス

    ```python
    from easybench import EasyBench, BenchConfig

    class BenchListOperation(EasyBench):
        # ベンチマーク設定
        bench_config = BenchConfig(
            trials=10,
            memory=True,
            sort_by="avg"
        )

        # 各試行のセットアップ
        def setup_trial(self):
            self.big_list = list(range(1_000_000))

        # ベンチマークメソッド（bench_で始まる必要があります）
        def bench_insert_first(self):
            self.big_list.insert(0, 123)

        # 複数のベンチマークメソッドが定義可能です
        def bench_pop_first(self):
            self.big_list.pop(0)

    if __name__ == "__main__":
        # ベンチマーク実行
        BenchListOperation().bench()
    ```

3. `easybench`コマンド

    1. `benchmarks`ディレクトリを作成
    2. `bench_*.py`スクリプトをディレクトリに配置：

        ```python
        from easybench import fixture

        # 各試行用のフィクスチャ
        @fixture(scope="trial")
        def big_list():
            return list(range(1_000_000))

        # ベンチマーク関数（bench_で始まる必要があります）
        def bench_insert_first(big_list):
            big_list.insert(0, 123)

        # 複数のベンチマーク関数が定義可能です
        def bench_pop_first(big_list):
                big_list.pop(0)
        ```

    3. `easybench`コマンドを実行

        ```bash
        easybench --trials 10 --memory --sort-by avg
        ```

**ベンチマーク結果の例：**

* 単一ベンチマーク

    ```
    Benchmark Results (5 trials):
    
    Function        Avg Time (s)  Min Time (s)  Max Time (s)
    --------------------------------------------------------
    insert_first        0.001568      0.001071      0.003265
    ```

* 複数ベンチマーク

  ![EasyBench Benchmark Result](https://raw.githubusercontent.com/smurak/easybench/main/images/easybench_screenshot.png)

* ボックスプロットによる可視化

  ![Boxplot Visualization](https://raw.githubusercontent.com/smurak/easybench/main/images/visualization_boxplot.png)

* バイオリンプロットによる可視化

  ![Violinplot Visualization](https://raw.githubusercontent.com/smurak/easybench/main/images/visualization_violinplot.png)

* ラインプロットによる可視化

  ![Lineplot Visualization](https://raw.githubusercontent.com/smurak/easybench/main/images/visualization_lineplot.png)

* ヒストグラムによる可視化

  ![Histplot Visualization](https://raw.githubusercontent.com/smurak/easybench/main/images/visualization_histplot.png)

* バープロットによる可視化

  ![Barplot Visualization](https://raw.githubusercontent.com/smurak/easybench/main/images/visualization_barplot.png)

## 使用方法

詳細な使用方法については、[**ドキュメント**](https://easybench.readthedocs.io/ja/)を参照してください。

## メモリ測定の制限

> [!NOTE]
> EasyBenchはPython組み込みの`tracemalloc`モジュールを使用してメモリ使用量を測定します。  
> これには重要な制限があります：
>
> - `tracemalloc`はPythonのメモリマネージャを通じて行われたメモリ割り当てのみを追跡します
> - C拡張（NumPy、Pandas、その他のネイティブライブラリなど）によって割り当てられたメモリは、多くの場合Pythonのメモリマネージャをバイパスするため、正確に測定されません
> - 報告されるメモリ使用量はPythonオブジェクトのみを反映し、プロセス全体のメモリ消費量ではありません
>
> C拡張を多用するアプリケーションでは、より正確な測定のために`memory_profiler`やシステムモニタリングツールなどの外部プロファイラの使用を検討してください。


## ライセンス

[MIT](./LICENSE)
