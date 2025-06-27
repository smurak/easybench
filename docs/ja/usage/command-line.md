## コマンドラインインターフェース（`easybench`コマンド）

複数のベンチマークを一度に実行するには、`easybench`コマンドを使用します：

```bash
easybench [オプション] [パス]
```

* デフォルトでは、`benchmarks`ディレクトリ内の`bench_*.py`という名前のファイルを実行します
* ベンチマークファイルを含むディレクトリまたは特定のベンチマークファイルを指定できます
* ベンチマークスクリプトは以下の規則に従う必要があります：
    * クラスベースのベンチマーク：`EasyBench`基本クラスを継承するクラスは自動的に検出され、含まれます
    * 関数ベースのベンチマーク：`bench_`で始まる関数名のものがベンチマーク関数として認識されます

### コマンドオプション

```bash
easybench [--trials N] [--loops-per-trial N] [--warmups N] [--memory] [--memory-unit UNIT] [--sort-by METRIC] [--reverse] [--no-color] [--show-output] [--time-unit UNIT] [--no-progress] [--progress] [--include PATTERN] [--exclude PATTERN] [--include-files PATTERN] [--exclude-files PATTERN] [--no-time] [--clip-outliers VALUE] [パス]
```

- `--trials N`：試行数（デフォルト：5）
- `--loops-per-trial N`：精度向上のために試行ごとに実行するループ数
- `--warmups N`：計測前に実行するウォームアップ試行数
- `--memory`：メモリ測定を有効化
- `--memory-unit UNIT`：結果表示用のメモリ単位（B/KB/MB/GB）
- `--sort-by METRIC`：並べ替え基準（def/avg/min/max/avg_memory/max_memory）
- `--reverse`：結果を降順に並べ替え
- `--no-color`：カラー出力を無効化
- `--show-output`：関数の戻り値を表示
- `--time-unit UNIT`：結果表示用の時間単位（s/ms/us/ns/m）
- `--no-progress`：ベンチマーク中のプログレスバーを無効化
- `--progress`：ベンチマーク中のプログレスバーを有効化
- `--include PATTERN`：一致するベンチマーク関数のみを含める正規表現パターン
- `--exclude PATTERN`：一致するベンチマーク関数を除外する正規表現パターン
- `--include-files PATTERN`：一致するベンチマークファイルのみを含める正規表現パターン
- `--exclude-files PATTERN`：一致するベンチマークファイルを除外する正規表現パターン
- `--no-time`：時間測定レポートを無効化
- `--clip-outliers VALUE`: 指定した割合の両端（最小・最大側）の値を切り詰める （0より大きく0.5未満）
- `パス`：ベンチマークファイルを含むディレクトリまたは特定のベンチマークファイル（デフォルト："benchmarks"）

### 関数ベースのベンチマーク例

コマンドラインから実行する関数ベースのベンチマークの例：

```python
# ファイル名: benchmarks/bench_list_operations.py
from easybench import fixture

@fixture(scope="trial")
def big_list():
    return list(range(1_000_000))

def bench_insert_first(big_list):
    """リストの先頭に要素を挿入"""
    big_list.insert(0, 123)

def bench_pop_first(big_list):
    """リストの先頭から要素を削除"""
    big_list.pop(0)
```

このファイルを`benchmarks`フォルダに保存し、`easybench`コマンドを実行して両方の関数をベンチマークし、結果を比較します：

```bash
easybench --trials 10 --memory
```

特定のベンチマークファイルを直接実行することもできます：

```bash
easybench --trials 10 --memory benchmarks/bench_list_operations.py
```
