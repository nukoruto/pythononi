import argparse
import os

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from episode_swap_env import EpisodeSwapEnv
from stable_baselines3.common.env_util import make_vec_env


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent for TagEnv")
    parser.add_argument("--timesteps", type=int, default=10000, help="Training steps per episode")
    parser.add_argument("--oni-model", type=str, default="oni_policy.zip", help="Path to save/load oni model")
    parser.add_argument("--nige-model", type=str, default="nige_policy.zip", help="Path to save/load nige model")
    parser.add_argument("--checkpoint-freq", type=int, default=0, help="Save checkpoints every N steps")
    parser.add_argument("--render", action="store_true", help="Render environment during training")
    parser.add_argument("--render-interval", type=int, default=1, help="Render every N steps when --render is set")
    parser.add_argument("--duration", type=int, default=10, help="Training duration per episode in seconds")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--speed-multiplier", type=float, default=1.0, help="Environment speed multiplier")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    return parser.parse_args()


class RenderCallback(BaseCallback):
    """Render environment during training if ``--render`` is specified."""

    def __init__(self, env: gym.Env, render_interval: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.env = env
        self.render_interval = max(1, render_interval)

    def _on_step(self) -> bool:  # type: ignore[override]
        if hasattr(self.env, "training_end_time"):
            import time
            self.env.remaining_time = max(0.0, self.env.training_end_time - time.time())
        if self.n_calls % self.render_interval == 0:
            self.env.render()
        return True


def _create_env(args: argparse.Namespace):
    if args.num_envs > 1:
        if args.render:
            print("--render は --num-envs が1のときのみ有効です")
        return make_vec_env(
            lambda: EpisodeSwapEnv(speed_multiplier=args.speed_multiplier),
            n_envs=args.num_envs,
        )
    return EpisodeSwapEnv(speed_multiplier=args.speed_multiplier)


def run_single(run_idx: int, args: argparse.Namespace) -> None:
    """Alternate training between oni and nige each episode."""

    env = _create_env(args)
    oni_model_path = args.oni_model.replace(".zip", f"_{run_idx}.zip")
    nige_model_path = args.nige_model.replace(".zip", f"_{run_idx}.zip")

    if os.path.exists(args.oni_model) and run_idx == 0:
        oni_model = PPO.load(args.oni_model, env=env)
        print(f"Loaded oni model from {args.oni_model}")
    else:
        oni_model = PPO("MlpPolicy", env, verbose=1)

    if os.path.exists(args.nige_model) and run_idx == 0:
        nige_model = PPO.load(args.nige_model, env=env)
        print(f"Loaded nige model from {args.nige_model}")
    else:
        nige_model = PPO("MlpPolicy", env, verbose=1)

    if hasattr(env, "envs"):
        for e in env.envs:
            e.oni_model = oni_model
            e.nige_model = nige_model
    else:
        env.oni_model = oni_model
        env.nige_model = nige_model

    for ep in range(args.episodes):
        if hasattr(env, "envs"):
            for e in env.envs:
                e.base_env.current_run = ep + 1
                e.base_env.total_runs = args.episodes
        else:
            env.base_env.current_run = ep + 1
            env.base_env.total_runs = args.episodes

        train_oni = env.training_agent == "oni"

        callbacks: list[BaseCallback] = []
        if args.checkpoint_freq > 0:
            prefix = "oni" if train_oni else "nige"
            callbacks.append(
                CheckpointCallback(
                    save_freq=args.checkpoint_freq,
                    save_path=".",
                    name_prefix=f"{prefix}_checkpoint_{run_idx}_{ep}"
                )
            )
        if args.render and args.num_envs == 1:
            callbacks.append(RenderCallback(env, render_interval=args.render_interval))

        import time
        start = time.time()
        if hasattr(env, "envs"):
            for e in env.envs:
                e.training_end_time = start + args.duration
        else:
            env.training_end_time = start + args.duration
        model = oni_model if train_oni else nige_model
        while time.time() - start < args.duration:
            model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False, callback=callbacks)

        # Start new episode and swap training agent automatically
        if hasattr(env, "envs"):
            for e in env.envs:
                e.reset()
        else:
            env.reset()

    oni_model.save(oni_model_path)
    nige_model.save(nige_model_path)
    env.close()


def main():
    args = parse_args()
    run_single(0, args)



if __name__ == "__main__":
    main()
