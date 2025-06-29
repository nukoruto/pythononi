# Python Oni

ランダム生成された壁付きステージ上で2D鬼ごっこを行うための実験用コードです。
`stage_generator.py` による迷路生成に加え、pygame を用いた可視化および `gym`
ベースの環境 `TagEnv` を用意しています。

## ステージ生成

```bash
python3 stage_generator.py
```

実行すると固定サイズ（例: 31x21）のステージが標準出力に表示されます。
`generate_stage` 関数に幅・高さを指定することで別サイズのステージも生成可能です。
ステージには行き止まりが存在しないよう調整されており、道幅はランダムで広げられます。

## 鬼ごっこ環境

以下のスクリプトで単純な鬼ごっこを実行できます。

```bash
python3 tag_game.py
```

また、強化学習向けには `gym_tag_env.py` に `TagEnv` クラスを実装しています。
`reset()` でステージとエージェントを再初期化し、`step(action)` では加速度ベースで
移動させながら報酬と終了判定を返します。行動・観測はいずれも `spaces.Box` により
連続値として定義しています。`render()` を呼び出すと pygame で状態を描画します。

## 学習

以下のコマンドで必要なライブラリをインストールしてください。

```bash
pip install -r requirements.txt
```

`train.py` では Stable-Baselines3 の PPO を用いた学習が行えます。
学習中にマップと各エージェントの視野を表示したい場合は `--render` オプションを
指定してください。

```bash
python3 train.py --timesteps 50000 --render
```

描画更新間隔は `--render-interval` で指定できます (デフォルト1ステップごと)。
