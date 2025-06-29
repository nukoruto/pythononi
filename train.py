import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from gym_tag_env import TagEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent for TagEnv")
    parser.add_argument("--timesteps", type=int, default=10000, help="Training steps per run")
    parser.add_argument("--model", type=str, default="ppo_tag.zip", help="Path to save/load model")
    parser.add_argument("--checkpoint-freq", type=int, default=0, help="Save checkpoints every N steps")
    parser.add_argument("--render", action="store_true", help="Render environment during training")
    parser.add_argument("--render-interval", type=int, default=1, help="Render every N steps when --render is set")
    parser.add_argument("--duration", type=int, default=10, help="Training duration per run in seconds")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs to execute")
    parser.add_argument("--parallel", type=int, default=1, help="Number of concurrent runs")
    parser.add_argument("--speed-multiplier", type=float, default=1.0, help="Environment speed multiplier")
    return parser.parse_args()


class RenderCallback(BaseCallback):
    """Render environment during training if ``--render`` is specified."""

    def __init__(self, env: TagEnv, render_interval: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.env = env
        self.render_interval = max(1, render_interval)

    def _on_step(self) -> bool:  # type: ignore[override]
        if self.n_calls % self.render_interval == 0:
            self.env.render()
        return True


def run_single(run_idx: int, args: argparse.Namespace) -> None:
    env = TagEnv(speed_multiplier=args.speed_multiplier)
    if os.path.exists(args.model) and run_idx == 0:
        model = PPO.load(args.model, env=env)
        print(f"Loaded model from {args.model}")
    else:
        model = PPO("MlpPolicy", env, verbose=1)

    callbacks = []
    if args.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=".",
                name_prefix=f"ppo_checkpoint_{run_idx}"
            )
        )
    if args.render and args.parallel == 1:
        callbacks.append(RenderCallback(env, render_interval=args.render_interval))

    import time
    start = time.time()
    while time.time() - start < args.duration:
        model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False, callback=callbacks)
    model.save(args.model.replace(".zip", f"_{run_idx}.zip"))
    env.close()


def main():
    args = parse_args()

    if args.parallel > 1:
        from multiprocessing import Process
        processes = []
        for i in range(args.runs):
            p = Process(target=run_single, args=(i, args))
            p.start()
            processes.append(p)
            if len(processes) >= args.parallel:
                for pr in processes:
                    pr.join()
                processes = []
        for pr in processes:
            pr.join()
    else:
        for i in range(args.runs):
            run_single(i, args)



if __name__ == "__main__":
    main()
