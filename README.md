# Python Oni

ランダム生成された壁付きステージ上で2D鬼ごっこを行うための実験用コードです。
`stage_generator.py` による迷路生成に加え、pygame を用いた可視化および `gym`
ベースの環境 `MultiTagEnv` を用意しており、鬼も逃げも強化学習の対象となります。
壁はセルの集合ではなくポリゴンとして保持するようになり、衝突判定や描画は
`shapely` を利用したポリゴン同士の交差判定で行います。

## ステージ生成

```bash
python3 stage_generator.py --width 31 --height 21
```

実行すると固定サイズ（例: 31x21）のステージが標準出力に表示されます。
`generate_stage` 関数に幅・高さを指定することで別サイズのステージも生成可能です。
ステージには行き止まりが存在せず、孤立したエリアも生じないよう接続性を保ったまま生成されます。道幅はランダムで広げられます。
壁密度を高めたい場合は `--extra-wall-prob` オプションで値を指定します。デフォルトは
`0.1` です。

## 鬼ごっこ環境

以下のスクリプトで単純な鬼ごっこを実行できます。

```bash
py tag_game.py
```
実行すると、鬼から逃げまでの経路が `shortest_path_vectors` を用いて計算され、
壁を回避した最短経路が緑色の線で表示されます。ステージはポリゴン障害物として
描画され、
ステージサイズはデフォルトで
31x21 ですが、`--width-range` と `--height-range` を指定するとその範囲から
ランダムに奇数が選ばれます。

また、強化学習向けには `gym_tag_env.py` に `MultiTagEnv` クラスを実装しています。`reset()` でステージとエージェントを再初期化し、`step()` では鬼と逃げのアクションをタプルで与え、観測と報酬も `(鬼, 逃げ)` のタプルで返されます。初期位置は毎回ランダムに選ばれ、必要に応じて `start_distance_range` で互いの距離を制約できます。逃げ側の報酬は捕まったら `-1`、時間いっぱい逃げ切ったら `+1` です。現在の実装では、直線距離ではなく最短経路長の変化を用いて追加報酬を与えます。

`StageMap` クラスには壁を考慮した最短経路探索 `shortest_path` が用意されています。
その経路を構成する方向ベクトル列を取得するには `shortest_path_vectors` を利用します。

```python
stage = StageMap(31, 21)
start = pygame.Vector2(1, 1)
goal = pygame.Vector2(10, 10)
vectors = stage.shortest_path_vectors(start, goal)
```

## 学習

以下のコマンドで必要なライブラリをインストールしてください。

```bash
pip install -r requirements.txt
```

`train_sac.py` は Soft Actor-Critic (SAC) を用いて鬼と逃げを同時に学習するスクリプトです。
学習中にマップと各エージェントの状態を表示したい場合は `--render` オプションを指定してください。

```bash
py train_sac.py --episodes 1000 --render
```

描画更新間隔は `--render-interval` で指定できます (デフォルト1ステップごと)。
学習時間を秒単位で制限したい場合は `--duration` を用います。速度倍率や描画速度は `--speed-multiplier` と `--render-speed` で調整可能です。
保存したモデルは `evaluate_sac.py` から評価できます。

学習後、鬼側モデルと逃げ側モデルは `out/YYYYMMDD_HHMMSS/oni_sac.pth` と
`out/YYYYMMDD_HHMMSS/nige_sac.pth` に保存されます。

同じディレクトリにはエピソードごとの報酬を記録した `rewards.csv` と、
それを基に描画した学習曲線 `learning_curve.png` も生成されます。
これらを参照することで学習の進捗を確認できます。


以前は `train_selfplay.py` を用いて同時学習を行っていましたが、現在は `train_sac.py` を使用します。

### `tag_game.py`

| オプション | 説明 | デフォルト |
|------------|------|-----------|
| `--duration <秒>` | 1ゲームの制限時間 | 10.0 |
| `--width-range a,b` | 幅を[a,b]からランダムに奇数選択 | - |
| `--height-range a,b` | 高さを[a,b]からランダムに奇数選択 | - |
| `--games <int>` | 連続対戦数 | 1 |
| `--oni <path>` | 鬼側モデルのパス（指定すると逃げはプレイヤー操作） | - |
| `--nige <path>` | 逃げ側モデルのパス（指定すると鬼はプレイヤー操作） | - |

### `train_sac.py`

| オプション | 説明 | デフォルト |
|------------|------|-----------|
| `--oni-model <path>` | 鬼モデルの保存/読み込みパス | `oni_sac.pth` |
| `--nige-model <path>` | 逃げモデルの保存/読み込みパス | `nige_sac.pth` |
| `--checkpoint-freq <int>` | 指定間隔でチェックポイント保存 | 0 |
| `--render` | 学習中に画面を描画 | - |
| `--render-interval <int>` | 描画間隔(ステップ数) | 1 |
| `--duration <秒>` | 各エピソードの学習時間（環境時間） | 10 |
| `--episodes <int>` | 総エピソード数 | 10 |
| `--speed-multiplier <float>` | 環境の処理速度倍率（タイマーも連動） | 1.0 |
| `--render-speed <float>` | 描画FPSの倍率 | 1.0 |
| `--gamma <float>` | 割引率 | 0.99 |
| `--lr <float>` | 学習率 | 3e-4 |
| `--g` | GPU を利用する(利用可能な場合) | - |

`train_sac.py` で学習したモデルは、それぞれ `out/YYYYMMDD_HHMMSS/oni_sac.pth` と
`out/YYYYMMDD_HHMMSS/nige_sac.pth` に保存されます。

### `evaluate_sac.py`

| オプション | 説明 | デフォルト |
|------------|------|-----------|
| `--oni-model <path>` | 評価用鬼モデルのパス | `oni_sac.pth` |
| `--nige-model <path>` | 評価用逃げモデルのパス | `nige_sac.pth` |
| `--episodes <int>` | 評価エピソード数 | 10 |
| `--render` | 描画を有効化 | - |
| `--speed-multiplier <float>` | 環境の処理速度倍率 | 1.0 |
| `--render-speed <float>` | 描画FPSの倍率 | 1.0 |
| `--g` | GPU を利用する(利用可能な場合) | - |
| `--output-dir <path>` | 評価結果を保存する基点ディレクトリ | `eval` |

指定エピソードの評価が終了すると、`<output-dir>/YYYYMMDD_HHMMSS` 以下に
`rewards.csv` と `evaluation_curve.png` が生成されます。標準出力には
平均報酬とその標準偏差が表示されるため、学習時と同様に性能を定量的に
確認できます。

## ライセンス

このプロジェクトは [MIT License](LICENSE) のもとで公開されています。
