## Jupyter Notebook マジックコマンド（`%%easybench`）

### 基本的な使い方

`%%easybench`マジックコマンドを使うと、Jupyter Notebookのセル内のコードを直接ベンチマークできます：

```python
%%easybench --trials=3 --memory
# この下にベンチマークしたいコードを記述
result = []
for i in range(1_000_000):
    result.append(i)
```

### セットアップ

マジックコマンドを使用するには、最初に拡張機能を読み込む必要があります：

```python
%load_ext easybench
```

### オプション

`%%easybench`マジックコマンドは、以下のオプションをサポートしています：

- `--trials=N`：実行する試行回数（デフォルト：1）
- `--memory`：メモリ測定を有効化
- `--memory-unit=UNIT`：メモリ単位（B/KB/MB/GB）
- `--warmups=N`：ウォームアップ実行回数（デフォルト：0）
- `--loops-per-trial=N`：試行ごとのループ回数（デフォルト：1）
- `--clip-outliers=FLOAT`：外れ値のクリッピング率（0.0〜1.0）
- `--time-unit=UNIT`：時間単位（s/ms/us/ns/m）
- `--no-time`：時間測定を無効化
- `--reporters REPORTER [REPORTER ...]`：使用するレポーター（複数指定可能）。例：console、simple、boxplot、violinplot、lineplot、histplot、barplot、results.csv、results.json

### 詳細な使用例

複数のオプションを組み合わせて使用することも可能です：

```python
%%easybench --trials=10 --memory --memory-unit=MB --warmups=2 --time-unit=ms --reporters lineplot console
# リストの作成
data = [i for i in range(100_000)]

# データに対して操作を実行
sorted_data = sorted(data, reverse=True)
```
