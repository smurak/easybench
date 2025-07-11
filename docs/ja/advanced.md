# 高度な使用方法

## カスタム出力（`Formatter`と`Reporter`）

EasyBenchでは、**レポーター**（Reporter）という仕組みを使用してベンチマーク結果の出力を行います。デフォルトでは`ConsoleReporter`が使用されます。

`ConsoleReporter`はコンソール画面にデータを出力するレポーターで、デフォルトでは表形式（`TableFormatter`）でデータを整形して表示します。EasyBenchでは、この`Formatter`（出力形式）と`Reporter`（出力方法）を変更することで、様々な形式での出力が可能になります。

### Reporterの使用

レポーターを使用するには、ベンチマーク設定（`BenchConfig`または`@bench.config`）の`reporters`パラメータにリストとして設定します。`reporters`はリスト形式のため、複数の出力方法を同時に指定できます。

Reporterを指定する3つの方法があります：

1. **文字列で指定**：レポーター名を文字列として指定する

    - `"console"`: 標準的なテーブル形式のコンソール出力
    - `"simple"`: シンプルなコンソール出力
    - `"boxplot"`: ボックスプロットによる可視化
    - `"violinplot"`: バイオリンプロットによる可視化
    - `"boxplot-sns"`: seabornスタイルのボックスプロットによる可視化
    - `"violinlot-sns"`: seabornスタイルのバイオリンプロットによる可視化
    - `"lineplot"`: ラインプロットによる可視化
    - `"lineplot-sns"`: seabornスタイルのラインプロットによる可視化
    - `"histplot"`: ヒストプロット(ヒストグラム)による可視化
    - `"histplot-sns"`: seabornスタイルのヒストプロットによる可視化
    - `"barplot"`: バープロットによる可視化
    - `"barplot-sns"`: seabornスタイルのバープロットによる可視化
    - `"*.csv"` または `"*.json"`: ファイル出力

2. **引数付きで指定**：`(reporter_name, parameter_dict)`の形式で指定

3. **Reporterオブジェクトで指定**：Reporterクラスのインスタンスを直接指定


使用例：

```python
from easybench import BenchConfig
from easybench.reporters import FileReporter

# 様々な指定方法による複数の出力フォーマット
config = BenchConfig(
    ...
    reporters=[
        "console",                          # 文字列として指定
        ("simple", {"metric": "min"}),      # 引数付きで指定
        ("boxplot", {"log_scale": False}),  # 引数付きでプロット
        "results.csv",                      # ファイルパスとして指定
        FileReporter("results.json"),       # オブジェクトとして指定
    ]
)
```

### カスタムReporterの作成

高度なユースケースでは、レポーターを自作できます：

```python
from easybench.reporters import (
    Reporter, TableFormatter, JSONFormatter, CSVFormatter
)

# カスタムレポーターの例 - 結果をWebAPIに送信
class WebAPIReporter(Reporter):
    def __init__(self, api_url, auth_token):
        super().__init__(JSONFormatter())  # JSON形式を使用
        self.api_url = api_url
        self.auth_token = auth_token
    
    def report_formatted(self, formatted_output):
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

## ボックスプロット可視化（`BoxPlotFormatter`）

ベンチマーク結果をボックスプロットとして可視化できます。  
これは複数のトライアル間の分布や外れ値を分析するのに役立ちます：

```python
from easybench import BenchConfig, EasyBench, customize
from easybench.visualization import BoxPlotFormatter, PlotReporter


class BenchList(EasyBench):
    bench_config = BenchConfig(
        trials=100,
        warmups=100,
        loops_per_trial=100,
        reporters=[
            PlotReporter(
                BoxPlotFormatter(
                    showfliers=True,           # 外れ値を表示
                    log_scale=True,            # 対数スケールを使用
                    engine="seaborn",          # プロットエンジンとしてseabornを使用
                    orientation="horizontal",  # 水平または垂直の向き
                    width=0.5,                 # ボックスの幅（seabornのboxplotに直接渡される）
                    linewidth=0.5,             # 線の幅（seabornのboxplotに直接渡される）
                ),
            ),
        ],
    )

    def setup_trial(self):
        self.big_list = list(range(1_000_000))

    @customize(loops_per_trial=1000)
    def bench_append(self):
        self.big_list.append(-1)

    def bench_insert_start(self):
        self.big_list.insert(0, -1)

    def bench_insert_middle(self):
        self.big_list.insert(len(self.big_list) // 2, -1)

    @customize(loops_per_trial=1000)
    def bench_pop(self):
        self.big_list.pop()

    def bench_pop_zero(self):
        self.big_list.pop(0)


if __name__ == "__main__":
    BenchList().bench()
```

![Boxplot Visualization](https://raw.githubusercontent.com/smurak/easybench/main/images/visualization_boxplot.png)

### `BoxPlotFormatter`の主なオプション

- `showfliers`：外れ値を表示するかどうか（デフォルト：`True`）
- `log_scale`：対数スケールを使用するかどうか（デフォルト：`False`）
- `figsize`：図のサイズ（デフォルト：`(10, 6)`）
- `engine`：プロットエンジン（`"matplotlib"`または`"seaborn"`）
- `orientation`：ボックスプロットの向き（`"vertical"`または`"horizontal"`）
- `sns_theme`：`sns.set_theme()`に渡されるseabornテーマパラメータの辞書（例：`{"style": "darkgrid", "palette": "Set2", "context": "notebook"}`）

### `PlotReporter`のオプション

- `formatter`：使用するプロットフォーマッタ（例：`BoxPlotFormatter`）
- `show`：プロットを画面に表示するかどうか（デフォルト：`True`）
- `save_path`：プロットを保存するファイルパス
- `dpi`：画像解像度（デフォルト：`100`）

!!! note
    ボックスプロットを使用するには、`matplotlib`をインストールする必要があります：
    ```bash
    pip install matplotlib
    ```
    
    seabornエンジンを使用したい場合は、`seaborn`もインストールしてください：
    ```bash
    pip install seaborn
    ```
