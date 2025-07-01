# Python Oni

ランダム生成された壁付きステージ上で2D鬼ごっこを行うための実験用コードです。
`stage_generator.py` による迷路生成に加え、pygame を用いた可視化および `gym`
ベースの環境 `MultiTagEnv` を用意しており、鬼も逃げも強化学習の対象となります。

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
壁を回避した最短経路が緑色の線で表示されます。

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

`train.py` は 鬼と逃げを同時に学習する自作ポリシー勾配方式です。
学習中にマップと各エージェントの状態を表示したい場合は `--render` オプションを指定してください。
描画時にはステップ数の代わりに残り時間や実行回数、鬼と逃げの累積報酬が画面上部に表示されます。
`train.py` では各エピソード開始時にこれらの値を自動設定するため、表示が常に最新の状態に保たれます。

```bash
py train.py --episodes 1000 --render
```

描画更新間隔は `--render-interval` で指定できます (デフォルト1ステップごと)。
学習時間を秒単位で制限したい場合は `--duration` を用います。物理更新の倍率は `--speed-multiplier`、描画フレームレートは `--render-speed` でそれぞれ制御できます。`--speed-multiplier` を大きくすると 1 ステップあたりの内部更新回数が増え、計算負荷によっては指定倍率通りにならないことがあります。`--duration` で指定した値は環境時間なので、`--speed-multiplier` が 2 の場合、実際の経過時間は `duration / speed_multiplier` となります。
学習処理は `run_selfplay` 関数として実装されており、保存したモデルは `evaluate.py` を通じて同じネットワーク構造で評価できます。

学習後、鬼側モデルは `oni_selfplay.pth`、逃げ側モデルは `nige_selfplay.pth` として保存されます。

## 旧 self-play スクリプト

以前は `train_selfplay.py` を用いて同時学習を行っていましたが、`train.py` に統合したため基本的には使用不要です。残してありますが機能は同等です。

## コマンドラインオプション一覧

主要スクリプトで利用できる主なオプションを以下にまとめます。

### `tag_game.py`

| オプション | 説明 | デフォルト |
|------------|------|-----------|
| `--duration <秒>` | 1ゲームの制限時間 | 10.0 |

### `train.py`

| オプション | 説明 | デフォルト |
|------------|------|-----------|
| `--timesteps <int>` | 各エピソードの学習ステップ数 | 10000 |
| `--oni-model <path>` | 鬼モデルの保存/読み込みパス | `oni_policy.zip` |
| `--nige-model <path>` | 逃げモデルの保存/読み込みパス | `nige_policy.zip` |
| `--checkpoint-freq <int>` | 指定間隔でチェックポイント保存 | 0 |
| `--render` | 学習中に画面を描画 | - |
| `--render-interval <int>` | 描画間隔(ステップ数) | 1 |
| `--duration <秒>` | 各エピソードの学習時間（環境時間） | 10 |
| `--episodes <int>` | 総エピソード数 | 10 |
| `--speed-multiplier <float>` | 環境の処理速度倍率（タイマーも連動） | 1.0 |
| `--render-speed <float>` | 描画FPSの倍率 | 1.0 |
| `--gamma <float>` | 自作ポリシー勾配用の割引率 | 0.99 |
| `--lr <float>` | 自作ポリシー勾配用の学習率 | 3e-4 |
| `--g` | GPU を利用する(利用可能な場合) | - |

学習済み `.pth` ファイルを読み込み、`train.py` と同じ `Policy` ネットワークで行動を計算します。

### `evaluate.py`

| オプション | 説明 | デフォルト |
|------------|------|-----------|
| `--oni-model <path>` | 評価用鬼モデルのパス | `oni_selfplay.pth` |
| `--nige-model <path>` | 評価用逃げモデルのパス | `nige_selfplay.pth` |
| `--episodes <int>` | 評価エピソード数 | 10 |
| `--render` | 描画を有効化 | - |
| `--speed-multiplier <float>` | 環境の処理速度倍率 | 1.0 |
| `--render-speed <float>` | 描画FPSの倍率 | 1.0 |
| `--g` | GPU を利用する(利用可能な場合) | - |

## ライセンス

このプロジェクトは [MIT License](LICENSE) のもとで公開されています。
