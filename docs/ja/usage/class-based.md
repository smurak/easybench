## クラスベースのベンチマーク（`EasyBench`クラス）

複数のベンチマークを比較したり、より複雑なセットアップを行ったりする場合、クラスベースのアプローチが便利です：

```python
from easybench import EasyBench, BenchConfig

class BenchListOperation(EasyBench):

    # ベンチマーク設定
    bench_config = BenchConfig(
        trials=10,     # 試行数
        memory=True,   # メモリ使用量を測定
        sort_by="avg"  # 平均時間で並べ替え
    )

    # 各試行の前に実行
    def setup_trial(self):
        self.big_list = list(range(10_000_000))

    # ベンチマークメソッド（bench_で始まる必要があります）
    def bench_insert_first(self):
        self.big_list.insert(0, 123)


    def bench_pop_first(self):
        self.big_list.pop(0)

if __name__ == "__main__":
    BenchListOperation().bench()
```

クラスベースのアプローチの使い方：

1. `EasyBench`を継承するクラスを作成
2. `bench_config`クラス変数でベンチマーク設定を構成
3. `setup_trial`メソッドで各試行の準備を行う
4. `bench_`で始まるメソッドがベンチマークの対象になる
5. ベンチマークを実行するために`bench()`メソッドを呼び出す
   * `bench()`は結果を画面に表示し、測定値の辞書を返す

### ライフサイクルメソッド

クラスベースのベンチマークでは、以下のライフサイクルメソッドを使用できます：

```python
class BenchExample(EasyBench):
    def setup_class(self):
        # クラス内のすべてのベンチマークの前に1回実行
        ...
        
    def teardown_class(self):
        # クラス内のすべてのベンチマークの後に1回実行
        ...
        
    def setup_function(self):
        # 各ベンチマーク関数の前に実行
        ...
        
    def teardown_function(self):
        # 各ベンチマーク関数の後に実行
        ...
        
    def setup_trial(self):
        # 各試行の前に実行
        ...
        
    def teardown_trial(self):
        # 各試行の後に実行
        ...
```

#### パラメータ化されたベンチマーク（`parametrize`デコレータ）

`parametrize`デコレータを使用して、異なるパラメータセットで同じベンチマークメソッドを実行できます：

```python
from easybench import BenchParams, EasyBench, parametrize

class BenchListOperations(EasyBench):
    # BenchParamsでパラメータセットを定義
    small_params = BenchParams(
        name="Small List",
        params={"size": 10_000}
    )
    
    large_params = BenchParams(
        name="Large List",
        params={"size": 1_000_000}
    )
    
    # parametrizeデコレータにパラメータセットのリストを渡して適用
    @parametrize([small_params, large_params])
    def bench_create_list(self, size):
        return list(range(size))

if __name__ == "__main__":
    BenchListOperations().bench()
```

これにより、各パラメータセットでベンチマークが実行され、結果にパラメータセット名が含まれます：

```
Benchmark Results (5 trials):

Function                         Avg Time (s) Min Time (s) Max Time (s)
--------------------------------------------------------------------
bench_create_list (Small List)   0.000442     0.000309     0.000855    
bench_create_list (Large List)   0.092680     0.062617     0.129535    
```

#### パラメータセットの組み合わせ（`parametrize.grid`）

より複雑なパラメータ化テストのために、`parametrize.grid`は複数のパラメータリストのすべての組み合わせを作成します：

```python
from easybench import BenchParams, EasyBench, parametrize

class BenchContainerOperations(EasyBench):
    # 2つのパラメータセットを定義
    sizes = [
        BenchParams(name="Small", params={"size": 100}),
        BenchParams(name="Large", params={"size": 10000}),
    ]
    
    operations = [
        BenchParams(name="Append", fn_params={"op": lambda x: x.append(0)}),
        BenchParams(name="Pop", fn_params={"op": lambda x: x.pop()}),
    ]
    
    # すべてのパラメータ組み合わせを作成
    @parametrize.grid([sizes, operations])
    def bench_operation(self, size, op):
        lst = list(range(size))
        op(lst)

if __name__ == "__main__":
    BenchContainerOperations().bench()
```

これにより、4つのパラメータ組み合わせが作成されます：

```
Benchmark Results (5 trials):

Function                            Avg Time (s)  Min Time (s)  Max Time (s)
----------------------------------------------------------------------------
bench_operation (Small x Append)        0.000260      0.000004      0.001239
bench_operation (Small x Pop)           0.000005      0.000004      0.000008
bench_operation (Large x Append)        0.001642      0.000277      0.006804
bench_operation (Large x Pop)           0.000204      0.000183      0.000247
```

### フィクスチャ（`fixture`デコレータ）

共通のテストデータを提供するために、pytestスタイルのフィクスチャを使用できます：

```python
from easybench import EasyBench, fixture

# フィクスチャを定義
@fixture(scope="trial")
def big_list():
    return list(range(10_000_000))

class BenchListOperation(EasyBench):
    # 引数としてフィクスチャを受け取る
    def bench_insert_first(self, big_list):
        big_list.insert(0, -1)

    def bench_pop_first(self, big_list):
        big_list.pop(0)

if __name__ == "__main__":
    BenchListOperation().bench()
```

`fixture`デコレータの`scope`パラメータはフィクスチャの生存期間を指定します：

- `"trial"`：各試行ごとに作成（デフォルト）
- `"function"`：ベンチマーク関数ごとに1回作成
- `"class"`：ベンチマーククラスごとに1回作成


<a id="設定オプション"></a>
### 設定オプション

`BenchConfig`クラスで以下の設定が利用可能です：

```python
from easybench import BenchConfig, EasyBench, customize

class MyBenchmark(EasyBench):
    bench_config = BenchConfig(
        trials=5,               # 試行数
        warmups=2,              # 実際の測定前のウォームアップ試行数
        sort_by="avg",          # 並べ替え基準
        reverse=False,          # 並べ替え順序（False=昇順、True=降順）
        memory=True,            # メモリ測定を有効化（TrueまたはB/KB/MB/GB）
        color=True,             # 結果でカラー出力を使用
        show_output=False,      # 関数の戻り値を表示
        loops_per_trial=1,      # 試行ごとの関数実行回数（下記説明参照）
        reporters=["console"],  # カスタムレポーター（「高度な使用方法」参照）
        progress=True,          # tqdmによる進捗表示を有効化
        include=None,           # 一致するベンチマークのみを含める正規表現パターン
        exclude=None,           # 一致するベンチマークを除外する正規表現パターン
        clip_outliers=None,     # 指定した割合の最大側の値を切り詰める （0以上1未満）
    )

    # 個別のメソッドに対して設定をカスタマイズすることもできます
    @customize(loops_per_trial=1000, name="Pass")
    def bench_fast_operation(self):
        # このメソッドは1試行あたり1000回実行されます
        # そして結果には"Pass"という名前で表示されます
        pass
```

並べ替えオプション（`sort_by`）：

- `"def"`：定義順（デフォルト）
- `"avg"`：平均実行時間
- `"min"`：最小実行時間
- `"max"`：最大実行時間
- `"avg_memory"`：平均メモリ使用量（`memory=True`の場合）
- `"max_memory"`：最大メモリ使用量（`memory=True`の場合）

メモリ測定オプション（`memory`）：

- `False`：メモリ測定を無効化（デフォルト）
- `True`：メモリ測定を有効化し、キロバイト単位で表示
- `"B"`：メモリ使用量をバイト単位で表示
- `"KB"`：メモリ使用量をキロバイト単位で表示
- `"MB"`：メモリ使用量をメガバイト単位で表示
- `"GB"`：メモリ使用量をギガバイト単位で表示

時間測定オプション（`time`）：

- `False`：時間測定レポートを無効化
- `True`：時間測定レポートを秒単位で有効化
- `"s"`：時間を秒単位で表示（デフォルト）
- `"ms"`：時間をミリ秒単位で表示
- `"μs"`または`"us"`：時間をマイクロ秒単位で表示
- `"ns"`：時間をナノ秒単位で表示
- `"m"`：時間を分単位で表示

進捗表示オプション（`progress`）：

- `False`：進捗表示を無効化（デフォルト）
- `True`：tqdmを使用した進捗表示を有効化
- カスタム関数：tqdmインターフェースに従うカスタム進捗表示関数を使用

ベンチマーク選択オプション：

- `include`：指定した正規表現パターンに一致するベンチマーク関数のみを実行
- `exclude`：指定した正規表現パターンに一致するベンチマーク関数を実行から除外

  - パラメータ化されたベンチマークでは、これらのオプションはフルネーム（例："bench_func (param_name)"）に対してマッチングが行われます
  - 両方のオプションが指定された場合、`exclude`が優先されます

#### `warmups`による測定精度の向上

ベンチマークを行う際、最初の実行はコードのコンパイル、キャッシュのウォームアップ、またはその他のシステム効果によって影響を受ける可能性があります。
より安定した正確な測定値を得るために、`warmups`パラメータを使用して、実際の測定開始前に何回の試行を行うかを指定できます：

```python
@bench
@bench.config(trials=5, warmups=3, time="ms")
def my_function():
    # この関数はウォームアップとして3回実行され（結果は破棄）、
    # その後、測定対象の5回の実際の試行が実行されます
    # ...
```

`warmups`の仕組み：

- 実際の測定が始まる前に、関数は`warmups`回実行されます
- 各ウォームアップは、 `setup_trial` / `teardown_trial` を含む完全な試行実行です
- ウォームアップ試行の結果は破棄され、測定結果に含まれません
- ウォームアップが完了すると、結果が記録される通常の試行が始まります

使用すべき場面：

- 最適なパフォーマンスに到達するためにJITコンパイルが必要な関数の場合
- システムがキャッシュをウォームアップしたり、安定状態に達する時間が必要な場合
- 最初の数回の実行が一貫して異なるパフォーマンス特性を示す場合

#### `loops_per_trial`によるタイマー精度の向上

タイマー解像度が低い環境（例えば、特定の仮想マシンや`time.perf_counter()`の精度が限られているシステム）では、意味のある計測結果を得るために関数を複数回実行する必要がある場合があります。

また、非常に高速な操作（数マイクロ秒以下）をベンチマークする場合、時間計測(タイマー呼び出し)自体のオーバーヘッドが測定結果に大きな影響を与える可能性があります。  
そのような場合、`loops_per_trial`パラメータを使用することで、タイマー呼び出しのオーバーヘッドを分散させ、より正確な測定を行うことができます。

`loops_per_trial`パラメータは、単一の実行時間測定（試行）で関数を何回実行するかを指定します：

```python
# 要素数100のリストに対して、1を10000回追加する処理の平均時間を計測する
# （10000回の追加処理中、同じリストインスタンスを使い続ける点に注意）
# この処理を500回繰り返してベンチマークを計測する
@bench(small_list=lambda: list(range(100)))
@bench.config(trials=500, loops_per_trial=10000, time="us")
def append_item(small_list):
    small_list.append(1)
```

`loops_per_trial`の仕組み：

- 単一の実行時間測定（試行）内で関数が`loops_per_trial`回ループで実行されます
- 総実行時間を`loops_per_trial`で割って、実行あたりの平均時間を算出します
- これにより、個別の時間測定がタイマーの解像度制限の影響を受ける非常に高速な操作に対して、より正確な測定が可能になります

使用すべき場面：

- 非常に高速な操作（マイクロ秒またはナノ秒）の場合
- タイマーの精度が低い環境の場合
- 単純な操作で計測結果の変動が大きい場合

<a id="メモリ測定の制限"></a>
### メモリ測定の制限

!!! note
    EasyBenchはメモリ使用量を測定するためにPythonの組み込み`tracemalloc`モジュールを使用しています。  
    これにはいくつかの重要な制限があります：
   
    - `tracemalloc`はPythonのメモリマネージャを通じて行われるメモリ割り当てのみを追跡します
    - C拡張（NumPy、Pandas、その他のネイティブライブラリなど）によって割り当てられるメモリは多くの場合Pythonのメモリマネージャをバイパスし、正確に測定されません
    - 報告されるメモリ使用量はPythonオブジェクトのみを反映し、プロセスの総メモリ消費量ではありません
   
    C拡張を多用するアプリケーションでは、より正確な測定のために`memory_profiler`やシステムモニタリングツールなどの外部プロファイラーの使用を検討してください。
