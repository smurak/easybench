# 出力のカスタマイズ

## `Formatter`と`Reporter`

EasyBenchでは、**レポーター**（Reporter）という仕組みを使用してベンチマーク結果の出力を行います。デフォルトでは`ConsoleReporter`が使用されます。

`ConsoleReporter`はコンソール画面にデータを出力するレポーターで、デフォルトでは表形式（`TableFormatter`）でデータを整形して表示します。EasyBenchでは、この`Formatter`（出力形式）と`Reporter`（出力方法）を変更することで、様々な形式での出力が可能になります。

### Reporterの使用

レポーターを使用するには、ベンチマーク設定（`BenchConfig`または`@bench.config`）の`reporters`パラメータにリストとして設定します。`reporters`はリスト形式であり、複数の出力方法を同時に指定できます。

Reporterを指定する方法は3種類あります：

1. **文字列で指定**：レポーター名を文字列として指定する

    - `"console"`: 標準的なテーブル形式のコンソール出力
    - `"simple"`: シンプルなコンソール出力
    - `"boxplot"`: ボックスプロットによる可視化
    - `"violinplot"`: バイオリンプロットによる可視化
    - `"lineplot"`: ラインプロットによる可視化
    - `"histplot"`: ヒストプロット(ヒストグラム)による可視化
    - `"barplot"`: バープロットによる可視化
    - `"boxplot-sns"`: seabornスタイルのボックスプロットによる可視化
    - `"violinplot-sns"`: seabornスタイルのバイオリンプロットによる可視化
    - `"lineplot-sns"`: seabornスタイルのラインプロットによる可視化
    - `"histplot-sns"`: seabornスタイルのヒストプロットによる可視化
    - `"barplot-sns"`: seabornスタイルのバープロットによる可視化
    - `"*.csv"` または `"*.json"`: ファイル出力

2. **引数付きで指定**：`(reporter_name, parameter_dict)`の形式で指定  
   (例: `("boxplot", {"log_scale": False})`)

3. **Reporterオブジェクトで指定**：Reporterクラスのインスタンスを直接指定  
   (例: `FileReporter("results.json")`)


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

高度なユースケースでは、レポーターを自作できます。  
詳細は [ReporterやFormatterを自作する](custom-reporters.md)を参照してください。

### カスタムReporterの登録

カスタムレポーターを名前をつけて登録すると、文字列で指定して使用できるようになります：

```python
from easybench import BenchConfig, set_reporter
from easybench.reporters import ConsoleReporter, SimpleFormatter
from easybench.visualization import PlotReporter, LinePlotFormatter

# [方法1] 関数呼び出しを使用

# 1. レポーターオブジェクトを返す関数を作成
def create_log_plot(**kwargs):
    return PlotReporter(LinePlotFormatter(log_scale=True, **kwargs))

# 2. その関数を名前をつけて登録
set_reporter("log-lineplot", create_log_plot)  # "log-lineplot"として登録


# [方法2] デコレーター構文を使用

# 1. レポータオブジェクトを返す関数にデコレータを適用
@set_reporter("custom-simple")  # "custom-simple"として登録
def create_simple_reporter(**kwargs):
    return ConsoleReporter(SimpleFormatter(**kwargs))


# 登録したレポーターを使用
bench_config = BenchConfig(
    reporters=[
        "console",
        "log-lineplot",
        ("custom-simple", {"metric": "min"}),
    ]
)
```

