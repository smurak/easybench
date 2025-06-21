## デコレータベースのベンチマーク（`@bench`デコレータ）

### **基本的な使い方**

`@bench`デコレータは関数をベンチマークするための最もシンプルな方法を提供します：

```python
from easybench import bench

# 関数パラメータと共に@benchを追加
@bench(item=123, big_list=list(range(1_000_000)))
def insert_first(item, big_list):
    big_list.insert(0, item)
```

### **毎試行で入力値を生成する**

上記の例では、`big_list`は一度だけ作成され、すべての試行で同じリストが使用されます。  
各試行で新しいデータが必要な場合は、関数やラムダを使用してオンデマンドで新しいデータを生成します：

```python
from easybench import bench

# 各試行で新しいリストを作成
@bench(item=-1, big_list=lambda: list(range(1_000_000)))
def insert_first(item, big_list):
    big_list.insert(0, item)
```

### **関数パラメータ**（`@bench.fn_params`）

パラメータとして関数を使用したい場合があります。  
そのような場合は、`@bench.fn_params`デコレータを使用します：

```python
def pop_first(some_list):
    """リストから最初の要素を削除"""
    some_list.pop(0)

@bench(big_list=list(range(1_000_000)))
@bench.fn_params(func=pop_first)
def apply_function(big_list, func):
    func(big_list)
```

### **設定**（`@bench.config`）

ベンチマーク設定をカスタマイズするには、`@bench.config`デコレータを使用します：

```python
@bench(big_list=list(range(10_000_000)))
@bench.config(trials=10, memory=True)
def pop_first(big_list):
    big_list.pop(0)
```

!!! tip
    `@bench.config`デコレータは他のbenchデコレータの前（下）に配置してください。  


主な設定オプション：

* `trials`：試行数（デフォルト：`5`）
* `memory`：メモリ使用量も測定（デフォルト：`False`）
    * `False`：メモリ測定を無効化
    * `True`：メモリをキロバイト単位で表示
    * `"B"`、`"KB"`、`"MB"`、`"GB"`：メモリをバイト、キロバイト、メガバイト、またはギガバイト単位で表示
* `time`：時間測定単位を指定（デフォルト：`"s"`）
    * `"s"`：秒単位で時間を表示
    * `"ms"`：ミリ秒単位で時間を表示
    * `"μs"`または`"us"`：マイクロ秒単位で時間を表示
    * `"ns"`：ナノ秒単位で時間を表示
    * `"m"`：分単位で時間を表示
    * `False`：時間測定レポートを無効化
* その他のオプションについては、[設定オプション](./class-based.md#設定オプション)を参照してください

### **複数のパラメータセット**（`BenchParams`）

関数を複数のパラメータセットでベンチマークしたい場合、
`BenchParams`で作成したパラメータセットのリストを`@bench`デコレータに渡すことができます：

```python
from easybench import bench, BenchParams

# パラメータセットを定義
small = BenchParams(
    name="Small",                                 # パラメータセット名
    params={"lst": lambda: list(range(10_000))},  # @benchのパラメータ
)
large = BenchParams(
    name="Large",
    params={"lst": lambda: list(range(1_000_000))}
)

# 複数のパラメータセットでベンチマーク
@bench([small, large])
def pop_first(lst):
    return lst.pop(0)
```

#### **パラメータセットの組み合わせ**（`bench.grid`）

異なるパラメータセットのすべての組み合わせで関数をベンチマークするには、`bench.grid`を使用できます：

```python
from easybench import bench, BenchParams

# サイズパラメータセットを定義
small = BenchParams(name="Small", params={"size": 10})
large = BenchParams(name="Large", params={"size": 100})

# 操作パラメータセットを定義
append = BenchParams(name="Append", fn_params={"op": lambda x: x.append(0)})
pop = BenchParams(name="Pop", fn_params={"op": lambda x: x.pop()})

# すべてのパラメータの組み合わせの直積を作成
@bench.grid([[small, large], [append, pop]])
def operation(size, op):
    lst = list(range(size))
    op(lst)
```

これにより、すべてのパラメータの組み合わせが作成され、ベンチマークされます：

- Small × Append
- Small × Pop
- Large × Append
- Large × Pop

### **オンデマンドベンチマーク**

関数を実行しながら同時にそのパフォーマンスを測定したい場合は、`.bench()`メソッドを使用します：

```python
@bench
def insert_first(item, big_list):
    big_list.insert(0, item)
    return len(big_list)

# 通常の関数として実行（ベンチマークなし）
result = insert_first(3, list(range(1_000_000)))

# ベンチマークと共に実行
result = insert_first.bench(3, list(range(1_000_000)))
print(result)  # 1000001
```

* デフォルトでは、試行回数は `1` 回です。
* 複数の試行を実行するには、`bench_trials`パラメータを指定します：
  ```python
  result = insert_first.bench(3, list(range(1_000_000)), bench_trials=10)
  ```
* 複数の試行を実行する場合、`.bench()`メソッドは初回試行時の戻り値を返します。
  ```python
  result = insert_first.bench(3, list(range(100_000)), bench_trials=10)
  print(result)  # 100001
  ```
