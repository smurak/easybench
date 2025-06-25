# クイックスタート

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/smurak/easybench/blob/main/notebooks/easybench_quickstart.ipynb)

`easybench`でベンチマークを行う方法は3種類あります：

1. `@bench`デコレータ

    ```python
    from easybench import bench
    
    # @benchを関数パラメータと共に追加
    @bench(item=123, big_list=lambda: list(range(1_000_000)))
    def insert_first(item, big_list):
        big_list.insert(0, item)
    ```

    !!! tip
    
        各試行ごとに新しいデータが必要な場合は、そのデータを返す関数やラムダをパラメータで指定します。  
        （上記の例では`lambda: list(range(1_000_000))`のように）

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
    
        # 複数のベンチマークメソッドを定義できます
        def bench_pop_first(self):
            self.big_list.pop(0)
    
    if __name__ == "__main__":
        # ベンチマークを実行
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
        
        # 複数のベンチマーク関数を定義できます
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
