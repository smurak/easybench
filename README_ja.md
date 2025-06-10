# EasyBench

[![Tests](https://github.com/smurak/easybench/actions/workflows/test.yml/badge.svg)](https://github.com/smurak/easybench/actions)
[![Docs](https://readthedocs.org/projects/easybench/badge/?version=latest)](https://easybench.readthedocs.io/)
[![PyPI version](https://badge.fury.io/py/easybench.svg)](https://pypi.org/project/easybench/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[**ドキュメント**](https://easybench.readthedocs.io/)

シンプルで使いやすいPythonベンチマークライブラリ。

[English](README.md) | 日本語

## 特徴

- 3種類の柔軟なベンチマークスタイル（デコレータ、クラスベース、コマンドライン）
- 実行時間とメモリ使用量の両方を計測可能（[制限事項を参照](#メモリ測定の制限)）
- ボックスプロットによるベンチマーク結果の可視化で分布や外れ値を分析
- パラメータ化されたベンチマークで同一関数を異なる入力サイズで比較
- pytestライクなフィクスチャシステムでテストデータを簡単にセットアップ
- 完全なライフサイクルフック（setup/teardown）で細かいベンチマーク制御が可能
- 通常の関数実行とベンチマーク付き実行をオンデマンドで切り替え
- ソートやフォーマットオプションなど、カスタマイズ可能なベンチマーク設定
- 複数のベンチマークを一度に実行するコマンドラインツール
- 複数の出力形式（テキストテーブル、CSV、JSON、pandas.DataFrame）に対応
- カスタム出力先のための拡張可能なレポーティングシステム

## インストール

```bash
pip install easybench
```

## クイックスタート

`easybench`を使用する方法は3種類あります：

1. `@bench`デコレータ

    ```python
    from easybench import bench
    
    # 関数パラメータと共に@benchを追加
    @bench(item=123, big_list=lambda: list(range(1_000_000)))
    def add_item(item, big_list):
        big_list.append(item)
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
        def bench_append(self):
            self.big_list.append(123)
    
        # 複数のベンチマークメソッドが定義可能です
        def bench_pop(self):
            self.big_list.pop()
    
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
        def bench_append(big_list):
            big_list.append(123)
        
        # 複数のベンチマーク関数が定義可能です
        def bench_pop(big_list):
            big_list.pop()
        ```

    3. `easybench`コマンドを実行

        ```bash
        easybench --trials 10 --memory --sort-by avg
        ```

**ベンチマーク結果の例：**

* 単一ベンチマーク

    ```
    Benchmark Results (5 trials):
    
    Function   Avg Time (s) Min Time (s) Max Time (s)
    ----------------------------------------------
    add_item   0.002393     0.000939     0.007362   
    ```

* 複数ベンチマーク

  ![EasyBench Benchmark Result](https://raw.githubusercontent.com/smurak/easybench/main/images/easybench_screenshot.png)

* ボックスプロットによる視覚化

  ![Boxplot Visualization](https://raw.githubusercontent.com/smurak/easybench/main/images/visualization_boxplot.png)


## 使用方法

### デコレータベースのベンチマーク（`@bench`デコレータ）

#### **基本的な使い方**

`@bench`デコレータは関数をベンチマークする一番簡単な方法です：

```python
from easybench import bench

# 関数パラメータと共に@benchを追加
@bench(item=123, big_list=list(range(1_000_000)))
def add_item(item, big_list):
    big_list.append(item)
```

#### **毎試行で入力値を再作成する**

上記の例では、`big_list`は一度作成され、すべての試行で同じリストが使われます。  
各試行ごとに新しいデータを使用する場合は、そのデータを返す関数またはラムダをパラメータで指定します：

```python
from easybench import bench

# 各試行ごとに新しいリストを作成
@bench(item=-1, big_list=lambda: list(range(1_000_000)))
def append(item, big_list):
    big_list.append(item)
```

#### **関数パラメータ**（`@bench.fn_params`）

関数自体をパラメータとして使いたい場合は、`@bench.fn_params`デコレータを使用します：

```python
def pop_first(some_list):
    """リストの最初の要素を削除"""
    some_list.pop(0)

@bench(big_list=list(range(1_000_000)))
@bench.fn_params(func=pop_first)
def apply_function(big_list, func):
    func(big_list)
```

#### **設定**（`@bench.config`）

ベンチマーク設定をカスタマイズするには、`@bench.config`デコレータを使用します：

```python
@bench(big_list=list(range(10_000_000)))
@bench.config(trials=10, memory=True)
def pop_last(big_list):
    big_list.pop()
```

> [!TIP] 
> `@bench.config`デコレータは他のbenchデコレータより前（下）に配置してください。  


主な設定オプション：

* `trials`: 試行回数 (デフォルト: `5`)
* `memory`: メモリ使用量も測定
  * `False`: メモリ測定を無効化 (デフォルト)
  * `True`: キロバイト単位で表示
  * `"B"`, `"KB"`, `"MB"`, `"GB"`: バイト、キロバイト、メガバイト、ギガバイト単位で表示
* その他のオプションについては、以下の「設定オプション」を参照

#### **複数パラメータセットによるベンチマーク計測** (`BenchParams`)

1つの関数に関して、複数のパラメータセットでベンチマーク測定を行いたい場合、  
`BenchParams`で作成したパラメータセットのリストを`@bench`デコレータに渡します:

```python
from easybench import bench, BenchParams

# パラメータセットを定義
small = BenchParams(
    name="Small",                                 # パラメータセット名
    params={"lst": lambda: list(range(10_000))},  # @bench用パラメータ
)
large = BenchParams(
    name="Large",
    params={"lst": lambda: list(range(1_000_000))}
)

# 複数のパラメータセットでベンチマークを測定
@bench([small, large])
def pop_first(lst):
    return lst.pop(0)
```

#### **実行時ベンチマーク**

必要に応じてベンチマークを実行したい場合は、`.bench()`メソッドを使用します：

```python
@bench
def append_item(item, big_list):
    big_list.append(item)
    return len(big_list)

# 通常の関数として実行（ベンチマークなし）
result = append_item(3, list(range(1_000_000)))

# ベンチマークと共に実行
result = append_item.bench(3, list(range(1_000_000)))
print(result)  # 10000001
```
* 試行回数はデフォルトでは `1` 回です。
* 複数回試行したい場合は、`bench_trials`パラメータを指定します：
  ```python
  result = append_item.bench(3, list(range(1_000_000)), bench_trials=10)
  ```
* 複数回試行の場合、`.bench()`メソッドの戻り値は初回試行時の戻り値となります。
  ```python
  result = append_item.bench(3, [1,2,3], bench_trials=10)
  print(result)  # 4
  ```

### クラスベースのベンチマーク（`EasyBench`クラス）

複数のベンチマークを比較したり、より複雑なセットアップを行う場合には、クラスベースのアプローチが便利です：

```python
from easybench import EasyBench, BenchConfig

class BenchListOperation(EasyBench):

    # ベンチマーク設定
    bench_config = BenchConfig(
        trials=10,     # 試行回数
        memory=True,   # メモリ使用量を測定
        sort_by="avg"  # 平均時間でソート
    )

    # 各試行の前に実行
    def setup_trial(self):
        self.big_list = list(range(10_000_000))

    # ベンチマークメソッド（bench_で始まる必要があります）
    def bench_append(self):
        self.big_list.append(-1)

    def bench_insert_start(self):
        self.big_list.insert(0, -1)

if __name__ == "__main__":
    BenchListOperation().bench()
```

クラスベースアプローチの使い方：
1. `EasyBench`を継承したクラスを作成
2. `bench_config`クラス変数でベンチマーク設定を構成
3. `setup_trial`メソッドで各試行の準備
4. `bench_`で始まるメソッドがベンチマーク対象になります
5. `bench()`メソッドを呼び出してベンチマークを実行
   * ベンチマークの結果を画面に表示し、実測値の辞書を戻り値として返します

#### ライフサイクルメソッド

クラスベースのベンチマークでは、以下のライフサイクルメソッドを使用できます：

```python
class BenchExample(EasyBench):
    def setup_class(self):
        # クラス内のすべてのベンチマークの前に一度実行
        pass
        
    def teardown_class(self):
        # クラス内のすべてのベンチマークの後に一度実行
        pass
        
    def setup_function(self):
        # 各ベンチマーク関数の前に実行
        pass
        
    def teardown_function(self):
        # 各ベンチマーク関数の後に実行
        pass
        
    def setup_trial(self):
        # 各試行の前に実行
        pass
        
    def teardown_trial(self):
        # 各試行の後に実行
        pass
```

#### パラメータ化されたベンチマーク（`parametrize`デコレータ）

`parametrize`デコレータを使用すると、同じベンチマークメソッドを異なるパラメータセットで実行できます：

```python
from easybench import BenchParams, EasyBench, parametrize

class BenchListOperations(EasyBench):
    # BenchParamsを使用してパラメータセットを定義
    small_params = BenchParams(
        name="小さいリスト",
        params={"size": 10_000}
    )
    
    large_params = BenchParams(
        name="大きいリスト",
        params={"size": 1_000_000}
    )
    
    # parametrizeデコレータにパラメータセットのリストを渡して適用
    @parametrize([small_params, large_params])
    def bench_create_list(self, size):
        return list(range(size))

if __name__ == "__main__":
    BenchListOperations().bench()
```

これにより、ベンチマークメソッドは各パラメータセット毎に実行され、結果にはパラメータセット名が含まれます：

```
Benchmark Results (5 trials):

Function                     Avg Time (s) Min Time (s) Max Time (s)
----------------------------------------------------------------
bench_create_list (小さいリスト)   0.000541     0.000256     0.001111    
bench_create_list (大きいリスト)   0.052366     0.035805     0.090981    
```

#### フィクスチャ（`fixture`デコレータ）

複数のベンチマーク関数で共通のテストデータを提供するには、pytestスタイルのフィクスチャを使用できます：

```python
from easybench import EasyBench, fixture

# フィクスチャを定義
@fixture(scope="trial")
def big_list():
    return list(range(10_000_000))

class BenchListOperation(EasyBench):
    # フィクスチャを引数として受け取る
    def bench_append(self, big_list):
        big_list.append(-1)

    def bench_insert_start(self, big_list):
        big_list.insert(0, -1)

if __name__ == "__main__":
    BenchListOperation().bench()
```

`fixture`デコレータの`scope`パラメータはフィクスチャの生存期間を指定します：
- `"trial"`: 各試行ごとに作成 (デフォルト)
- `"function"`: ベンチマーク関数ごとに一度作成
- `"class"`: ベンチマーククラスごとに一度作成


#### 設定オプション

`BenchConfig`クラスでは以下の設定が利用可能です：

```python
from easybench import BenchConfig, EasyBench

class MyBenchmark(EasyBench):
    bench_config = BenchConfig(
        trials=5,            # 試行回数
        sort_by="avg",       # ソート基準
        reverse=False,       # ソート順序（False=昇順、True=降順）
        memory="MB",         # メモリ測定を有効化し、メガバイト単位で表示
        color=True,          # 結果にカラー出力を使用
        show_output=False,   # 関数の戻り値をベンチマーク結果に表示
        reporters=[]         # カスタムレポーター (後述の解説を参照)
    )
```

ソートオプション（`sort_by`）：
- `"def"`: 定義順 (デフォルト)
- `"avg"`: 平均実行時間
- `"min"`: 最小実行時間
- `"max"`: 最大実行時間
- `"avg_memory"`: 平均メモリ使用量 (`memory=True`の場合)
- `"max_memory"`: 最大メモリ使用量 (`memory=True`の場合)

メモリ測定オプション（`memory`）：
- `False`: メモリ測定を無効化 (デフォルト)
- `True`: メモリ測定を有効化し、キロバイト単位で表示
- `"B"`: バイト単位でメモリ使用量を表示
- `"KB"`: キロバイト単位でメモリ使用量を表示
- `"MB"`: メガバイト単位でメモリ使用量を表示
- `"GB"`: ギガバイト単位でメモリ使用量を表示

#### メモリ測定の制限

> [!NOTE]
> EasyBenchはPython組み込みの`tracemalloc`モジュールを使用してメモリ使用量を測定します。  
> これには重要な制限があります：
>
> - `tracemalloc`はPythonのメモリマネージャを通じて行われたメモリ割り当てのみを追跡します
> - C拡張（NumPy、Pandas、その他のネイティブライブラリなど）によって割り当てられたメモリは、多くの場合Pythonのメモリマネージャをバイパスするため、正確に測定されません
> - 報告されるメモリ使用量はPythonオブジェクトのみを反映し、プロセス全体のメモリ消費量ではありません
>
> C拡張を多用するアプリケーションでは、より正確な測定のために`memory_profiler`やシステムモニタリングツールなどの外部プロファイラの使用を検討してください。

### コマンドラインインターフェース（`easybench`コマンド）

複数のベンチマークを一度に実行するには、`easybench`コマンドを使用します：

```bash
easybench [オプション] [パス]
```

* デフォルトでは、`benchmarks`ディレクトリ内の`bench_*.py`という名前のファイルを実行します
* ベンチマークファイルを含むディレクトリを指定するか、特定のベンチマークファイルを直接指定できます
* ベンチマークスクリプトは以下のルールに従う必要があります：
  * クラスベースのベンチマークでは、クラス名が`Bench`で始まる
  * 関数ベースのベンチマークでは、関数名が`bench_`で始まる

#### コマンドオプション

```bash
easybench [--trials N] [--memory] [--sort-by METRIC] [--reverse] [--no-color] [--show-output] [パス]
```

- `--trials N`: 試行回数 (デフォルト: 5)
- `--memory`: メモリ測定を有効化
- `--sort-by METRIC`: ソート基準 (def/avg/min/max/avg_memory/max_memory)
- `--reverse`: 結果を降順でソート
- `--no-color`: カラー出力を無効化
- `--show-output`: 関数の戻り値を表示
- `パス`: ベンチマークファイルを含むディレクトリまたは特定のベンチマークファイル (デフォルト: "benchmarks")

#### 関数ベースのベンチマーク例

コマンドラインから実行する関数ベースのベンチマークの例：

```python
# ファイル名: benchmarks/bench_list_operations.py
from easybench import fixture

@fixture(scope="trial")
def big_list():
    return list(range(10_000_000))

def bench_append(big_list):
    """リストの末尾に要素を追加"""
    big_list.append(-1)

def bench_insert_start(big_list):
    """リストの先頭に要素を挿入"""
    big_list.insert(0, -1)
```

このファイルを`benchmarks`フォルダに保存し、`easybench`コマンドを実行して両方の関数をベンチマークし、結果を比較します：

```bash
easybench --trials 10 --memory
```

## 高度な使用方法

### カスタム出力形式（`Formatter`と`Reporter`）

EasyBenchでは、**レポーター**（Reporter）という仕組みを使用してベンチマーク結果の出力を行います。デフォルトでは`ConsoleReporter`が使用されます。

`ConsoleReporter`はコンソール画面にデータを出力するレポーターで、デフォルトでは表形式（`TableFormatter`）でデータを整形して表示します。EasyBenchでは、この`Formatter`（出力形式）と`Reporter`（出力方法）を変更することで、様々な形式での出力が可能になります。

#### レポーターの使用方法

レポーターを使用するには、ベンチマーク設定（`BenchConfig`または`@bench.config`）の`reporters`パラメータにリストとして設定します。`reporters`はリスト形式のため、複数の出力方法を同時に指定できます。

レポーターを指定する方法は以下の3つがあります：

1. **文字列で指定**：レポーター名を文字列として指定する
   - `"console"`: 標準的なテーブル形式のコンソール出力
   - `"simple"`: シンプルなコンソール出力
   - `"plot"`: ボックスプロットによる可視化
   - `"*.csv"` または `"*.json"`: ファイル出力

2. **引数付きで指定**：`(レポーター名, パラメータ辞書)`の形式で指定する

3. **Reporterオブジェクトで指定**：Reporterクラスのインスタンスを直接指定する

* 使用例

    ```python
    from easybench import BenchConfig
    from easybench.reporters import FileReporter
    
    # 様々なレポーター設定
    bench_config = BenchConfig(
        trials=10,
        reporters=[
            "console",                                  # 文字列で指定
            ("simple", {"metric": "min"}),              # 引数付きで指定
            ("plot", {"log_scale": False}),             # 引数付きでプロット指定
            "results.csv",                              # ファイルパスで指定
            FileReporter("results.csv"),                # オブジェクトで指定
        ]
    )
    
    # より単純な設定
    bench_config = BenchConfig(reporters=["console"])       # コンソール出力のみ
    bench_config = BenchConfig(reporters=["plot"])          # ボックスプロットのみ
    bench_config = BenchConfig(reporters=["output.csv"])    # CSVファイル出力のみ
    ```

#### カスタムレポーターの作成

高度なユースケースでは、レポーターを自作できます：

```python
from easybench.reporters import (
    Reporter, TableFormatter, JSONFormatter, CSVFormatter
)

# カスタムレポーターの例 - Web APIに結果を送信
class WebAPIReporter(Reporter):
    def __init__(self, api_url, auth_token):
        super().__init__(JSONFormatter())  # JSON形式を使用
        self.api_url = api_url
        self.auth_token = auth_token
    
    def _send(self, formatted_output):
        # フォーマットされた結果をAPIエンドポイントに送信
        import requests
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        requests.post(self.api_url, headers=headers, json=formatted_output)

# BenchConfigで使用
bench_config = BenchConfig(
    reporters=[
        ConsoleReporter(),  # コンソールにも表示
        WebAPIReporter("https://api.example.com/benchmarks", "my_token")
    ]
)
```

### ボックスプロットによる可視化 (`BoxplotFormatter`)

ベンチマーク結果をボックスプロット（箱ひげ図）として視覚化することができます。  
これは複数試行間の分布や外れ値を分析するのに役立ちます。

```python
from easybench import BenchConfig, EasyBench
from easybench.visualization import BoxplotFormatter, PlotReporter


class BenchList(EasyBench):
    bench_config = BenchConfig(
        trials=100,
        reporters=[
            PlotReporter(
                BoxplotFormatter(
                    showfliers=True,        # 外れ値を表示するかどうか
                    log_scale=True,         # 対数スケールを使用
                    engine="seaborn",       # プロットエンジンとしてseabornを使用
                    orientation="horizontal", # ボックスプロットの方向（水平または垂直）
                    width=0.5,              # ボックスの幅（seabornのboxplotに直接渡される）
                    linewidth=0.5,          # ラインの太さ（seabornのboxplotに直接渡される）
                )
            )
        ],
    )

    def setup_trial(self):
        self.big_list = list(range(1_000_000))

    def bench_append(self):
        self.big_list.append(-1)

    def bench_insert_start(self):
        self.big_list.insert(0, -1)

    def bench_insert_middle(self):
        self.big_list.insert(len(self.big_list) // 2, -1)

    def bench_pop(self):
        self.big_list.pop()

    def bench_pop_zero(self):
        self.big_list.pop(0)


if __name__ == "__main__":
    import seaborn as sns

    # seabornのスタイル設定（オプション）
    sns.set_theme(style="darkgrid", palette="Set2")
    BenchList().bench()
```

![Boxplot Visualization](https://raw.githubusercontent.com/smurak/easybench/main/images/visualization_boxplot.png)

#### `BoxplotFormatter` の主なオプション

- `showfliers`: 外れ値を表示するかどうか（デフォルト: `True`）
- `log_scale`: 対数スケールを使用するかどうか（デフォルト: `False`）
- `data_limit`: グラフの値の範囲を指定する（例: `(0, 0.01)`）
- `trim_outliers`: 外れ値をトリミングするパーセンタイル（0.0〜0.5）
- `winsorize_outliers`: 外れ値を強制的に範囲内に収めるパーセンタイル（0.0〜0.5）
- `figsize`: 図のサイズ（デフォルト: `(10, 6)`）
- `engine`: プロットエンジン（`"matplotlib"` または `"seaborn"`）
- `orientation`: ボックスプロットの方向（`"vertical"` または `"horizontal"`）

#### `PlotReporter` のオプション

- `formatter`: 使用するプロットフォーマッタ（例: `BoxplotFormatter`）
- `show`: プロットを画面に表示するかどうか（デフォルト: `True`）
- `save_path`: プロットを保存するファイルパス
- `dpi`: 画像の解像度（デフォルト: `100`）

> [!NOTE]
> ボックスプロットを使用するには、`matplotlib`をインストールする必要があります:
> ```bash
> pip install matplotlib
> ```
> seabornエンジンを使用する場合は、追加で`seaborn`もインストールしてください:
> ```bash
> pip install seaborn
> ```

## ライセンス

[MIT](./LICENSE)


