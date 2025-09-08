# ReporterやFormatterを自作する

EasyBenchでは、ベンチマーク結果の出力は次の2つのコンポーネントによって処理されます:

1. **Formatter**: ベンチマーク結果を特定の形式(テキスト表、CSV、JSONなど)に変換します
2. **Reporter**: フォーマットされたデータを特定の宛先(コンソール、ファイル、APIなど)に送信します

この2種類のコンポーネントにより、「何を出力するか」と「どこに出力するか」を個別にカスタマイズできます。

## Formatter

Formatterは、ベンチマーク結果を特定の形式に変換するコンポーネントです。
例えば、テーブル形式、CSV、JSON、DataFrameなどの出力を行います。

### 既存のFormatter一覧

EasyBenchには以下の標準Formatterがあります:

- `TableFormatter`: テキストテーブル形式（デフォルト）
- `SimpleFormatter`: 簡潔なテキスト出力
- `CSVFormatter`: CSV形式
- `JSONFormatter`: JSON形式
- `DataFrameFormatter`: pandas DataFrame形式
- 各種プロットフォーマッター: `BoxPlotFormatter`, `LinePlotFormatter`など

### Formatterの実装方法

カスタムFormatterを作成するには、`Formatter`クラスを継承して、`format`メソッドを実装します。

```python
from easybench.reporters import Formatted, Formatter
from easybench.core import BenchConfig
from easybench.utils import ResultsType, StatsType

class CustomFormatter(Formatter):
    """カスタムFormatter."""
    
    def format(
        self,
        results: ResultsType,  # ベンチマーク結果データ
        stats: StatsType,      # 計算済み統計情報
        config: BenchConfig,   # ベンチマーク設定
    ) -> Formatted:            # フォーマットされた出力
        """結果を独自の形式にフォーマットします."""
        # ここに独自のフォーマットロジックを実装
        formatted_output = "Your custom formatting here"
        return formatted_output
```

#### `format`メソッドの引数

##### 1. `results`（ResultsType）

`results`引数の値は、実行したベンチマーク関数ごとの生データを含む辞書です：

```python
{
    "bench_function_a": {
        "times": [0.001, 0.0012, 0.0011],       # 各実行の所要時間（秒）
        "memory": [1024, 1028, 1022],           # 各実行のメモリ使用量（バイト）
        "output": ["result1", "result1", "result1"]  # 各実行の戻り値
    },
    "bench_function_b": {
        "times": [0.002, 0.0019, 0.0021],
        "memory": [2048, 2050, 2045],
        "output": [42, 42, 42]
    }
}
```

!!! warning "注意"
    各キー(`times`, `memory`, `output`)はベンチマークの設定により存在しない場合があります。例えば、メモリ取得オプションがオフのベンチマーク結果には`memory`キーの値が存在しません。

##### 2. `stats`（StatsType）

`stats`引数の値は、ベンチマーク結果から自動計算された統計情報の辞書です：

```python
{
    "bench_function_a": {
        "avg": 0.0011,           # 平均実行時間（秒）
        "min": 0.001,            # 最小実行時間
        "max": 0.0012,           # 最大実行時間
        "avg_memory": 1024.67,   # 平均メモリ使用量（バイト）
        "max_memory": 1028       # 最大メモリ使用量
    },
    "bench_function_b": {
        "avg": 0.002,
        "min": 0.0019,
        "max": 0.0021,
        "avg_memory": 2047.67,
        "max_memory": 2050
    }
}
```

!!! info "ヒント"
    統計情報は複数のフォーマッターで重複して計算しないよう、事前に計算されて渡されます。  
    独自の統計計算を行いたい場合は、`results`引数の生データを用いて計算してください。

##### 3. `config`（BenchConfig）

`config`引数の値は、ベンチマーク設定を含む`BenchConfig`オブジェクトです：

```python
BenchConfig(
    trials=10,              # 実行回数
    warmups=2,              # ウォームアップ回数
    time="ms",              # 時間計測オプション (True/False/"m"/"s"/"ms"/"us"/"ns")
    memory="KB",            # メモリ計測オプション (True/False/"B"/"KB"/"MB"/"GB")
    sort_by="avg",          # 結果のソート基準
    reverse=False,          # ソート順を逆にするか
    show_output=False,      # 出力値を表示するか
    color=True              # 色付き出力を使用するか
)
```

#### `format`メソッドの戻り値 (`Formatted`)

`format`メソッドは次のような型の値を戻り値とします：

- `str`: テキスト形式の出力 (テーブル、CSV、JSONなど)
- `pd.DataFrame`: pandas DataFrame形式
- `matplotlib.figure.Figure`: グラフ形式

### 実装例: XMLFormatter

以下はXML形式で出力するFormatterの実装例です：

```python
from easybench.reporters import Formatter, TimeUnit, MemoryUnit
from easybench.core import BenchConfig
from easybench.utils import ResultsType, StatsType

class XMLFormatter(Formatter):
    """ベンチマーク結果をXML形式でフォーマット"""
    
    def format(
        self,
        results: ResultsType,
        stats: StatsType,
        config: BenchConfig,
    ) -> str:
        """XML形式に変換します"""
        time_unit = TimeUnit.from_config(config)
        memory_unit = MemoryUnit.from_config(config)
        
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append(f'<benchmark trials="{config.trials}">')
        
        # ソート済み関数名のリストを取得
        methods = self.sort_keys(stats, config)
        
        for method_name in methods:
            stat = stats[method_name]
            lines.append(f'  <function name="{method_name}">')
            
            # 時間計測結果
            if config.time:
                avg_time = time_unit.convert_seconds(stat["avg"])
                min_time = time_unit.convert_seconds(stat["min"])
                max_time = time_unit.convert_seconds(stat["max"])
                
                lines.append(f'    <time unit="{time_unit}">')
                lines.append(f'      <average>{avg_time:.6f}</average>')
                lines.append(f'      <minimum>{min_time:.6f}</minimum>')
                lines.append(f'      <maximum>{max_time:.6f}</maximum>')
                lines.append('    </time>')
            
            # メモリ計測結果
            if config.memory:
                avg_mem = memory_unit.convert_bytes(stat["avg_memory"])
                max_mem = memory_unit.convert_bytes(stat["max_memory"])
                
                lines.append(f'    <memory unit="{memory_unit}">')
                lines.append(f'      <average>{avg_mem:.2f}</average>')
                lines.append(f'      <maximum>{max_mem:.2f}</maximum>')
                lines.append('    </memory>')
            
            lines.append('  </function>')
        
        lines.append('</benchmark>')
        return '\n'.join(lines)
```

## Reporter

Reporterは、指定したFormatterで変換されたデータを特定の宛先（コンソール、ファイル、APIなど）に送信するコンポーネントです。

### 既存のReporter一覧

EasyBenchには以下の標準Reporterがあります：

- `ConsoleReporter`: コンソール出力（デフォルト）
- `FileReporter`: ファイル出力
- `CallbackReporter`: コールバック関数への出力
- `SimpleConsoleReporter`: 簡潔なコンソール出力
- `PlotReporter`: グラフ出力

### Reporterの実装方法

カスタムReporterを作成するには、`Reporter`クラスを継承して、`report_formatted`メソッドを実装します。

```python
from easybench.reporters import Reporter, Formatted

class MyCustomReporter(Reporter):
    """カスタムレポーター."""
    
    def report_formatted(self, formatted_output: Formatted) -> None:
        """フォーマット済み出力をレポートする."""
        ...
```

また、`formatter`属性に`Formatter`オブジェクトを設定します。  
デフォルトでは、以下のように初期化時に第1引数に入力する仕組みになっています。

```python
class Reporter:
    def __init__(self, formatter: Formatter) -> None:
        self.formatter = formatter
```

#### `formatted_output` 引数（Formatted）

`formatted_output`は`Formatter`の`format`メソッドの戻り値と同じ以下のような形式です：

1. **文字列** (`str`): テキスト形式の出力
2. **DataFrame** (`pd.DataFrame`): 表形式データ
3. **Figure** (`matplotlib.figure.Figure`): グラフ画像

### 実装例: SlackReporter

以下はSlackにベンチマーク結果を送信するReporterの実装例です：

```python
from easybench.reporters import Reporter, TableFormatter

class SlackReporter(Reporter):
    """ベンチマーク結果をSlackに送信するレポーター"""
    
    def __init__(self, webhook_url, channel="#benchmarks", formatter=None):
        # デフォルトではTableFormatterを使用
        super().__init__(formatter or TableFormatter())
        self.webhook_url = webhook_url
        self.channel = channel
    
    def report_formatted(self, formatted_output: str) -> None:
        """
        フォーマット済みの出力をSlackに送信します.
        
        Args:
            formatted_output: Formatterから出力されたデータ(文字列のみに対応)
        """
        import requests
        
        payload = {
            "channel": self.channel,
            "text": "ベンチマーク結果",
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "```\n" + formatted_output + "\n```"}
                }
            ]
        }
        requests.post(self.webhook_url, json=payload)
```

## FormatterとReporterの連携

カスタム作成した`Formatter`と`Reporter`は、以下のように連携して使用できます：

```python
from easybench import BenchConfig

# カスタムFormatterとReporterを使用したベンチマーク設定
config = BenchConfig(
    trials=50,
    reporters=[
        # 独自のレポーターを使用
        SlackReporter(
            webhook_url="https://hooks.slack.com/services/XXX/YYY/ZZZ",
            formatter=XMLFormatter()  # フォーマッターのインスタンスを指定
        ),
        # 標準のコンソール出力も併用
        "console"
    ]
)
```

カスタムレポーターの登録方法については、[出力のカスタマイズ](customize-output.md)を参照してください。
