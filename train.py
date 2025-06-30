import argparse
import os

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from gym_tag_env import TagEnv, MultiTagEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent for TagEnv")
    parser.add_argument("--timesteps", type=int, default=10000, help="Training steps per run")
    parser.add_argument("--oni-model", type=str, default="oni_policy.zip", help="Path to save/load oni model")
    parser.add_argument("--nige-model", type=str, default="nige_policy.zip", help="Path to save/load nige model")
    parser.add_argument("--checkpoint-freq", type=int, default=0, help="Save checkpoints every N steps")
    parser.add_argument("--render", action="store_true", help="Render environment during training")
    parser.add_argument("--render-interval", type=int, default=1, help="Render every N steps when --render is set")
    parser.add_argument("--duration", type=int, default=10, help="Training duration per run in seconds")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs to execute")
    parser.add_argument("--parallel", type=int, default=1, help="Number of concurrent runs")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel envs per run")
    parser.add_argument("--speed-multiplier", type=float, default=1.0, help="Environment speed multiplier")
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


def run_single(run_idx: int, args: argparse.Namespace) -> None:
    """Train oni first, then nige using the trained oni model."""

    # --- train oni ---
    def make_oni_env() -> TagEnv:
        env = TagEnv(speed_multiplier=args.speed_multiplier)
        env.current_run = run_idx + 1
        env.total_runs = args.runs
        return env

    if args.num_envs > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        oni_env = SubprocVecEnv([make_oni_env for _ in range(args.num_envs)])
    else:
        oni_env = make_oni_env()

    oni_model_path = args.oni_model.replace(".zip", f"_{run_idx}.zip")
    if os.path.exists(args.oni_model) and run_idx == 0:
        oni_model = PPO.load(args.oni_model, env=oni_env)
        print(f"Loaded oni model from {args.oni_model}")
    else:
        oni_model = PPO("MlpPolicy", oni_env, verbose=1)

    if args.render and args.num_envs > 1:
        raise ValueError("--render is only supported when --num-envs=1")

    callbacks: list[BaseCallback] = []
    if args.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=".",
                name_prefix=f"oni_checkpoint_{run_idx}"
            )
        )
    if args.render and args.parallel == 1:
        callbacks.append(RenderCallback(oni_env, render_interval=args.render_interval))

    import time
    start = time.time()
    if args.num_envs == 1:
        oni_env.training_end_time = start + args.duration
    else:
        oni_env.set_attr("training_end_time", start + args.duration)
    while time.time() - start < args.duration:
        oni_model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False, callback=callbacks)
    oni_model.save(oni_model_path)
    oni_env.close()

    # --- train nige using fixed oni model ---
    fixed_oni = PPO.load(oni_model_path, env=TagEnv())

    class NigeEnv(gym.Env):
        def __init__(self, model: PPO):
            super().__init__()
            self.env = MultiTagEnv(speed_multiplier=args.speed_multiplier)
            self.model = model
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
            self._oni_obs = None

        def reset(self, *, seed: int | None = None, options=None):
            obs, _ = self.env.reset(seed=seed)
            self._oni_obs = obs[0]
            return obs[1], {}

        def step(self, action):
            oni_action, _ = self.model.predict(self._oni_obs, deterministic=True)
            (oni_obs, nige_obs), (_, nige_reward), term, trunc, info = self.env.step((oni_action, action))
            self._oni_obs = oni_obs
            return nige_obs, nige_reward, term, trunc, info

        def render(self):
            self.env.render()

        def close(self):
            self.env.close()

    def make_nige_env() -> gym.Env:
        env = NigeEnv(fixed_oni)
        env.env.current_run = run_idx + 1
        env.env.total_runs = args.runs
        return env

    if args.num_envs > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        nige_env = SubprocVecEnv([make_nige_env for _ in range(args.num_envs)])
    else:
        nige_env = make_nige_env()
    nige_model_path = args.nige_model.replace(".zip", f"_{run_idx}.zip")
    if os.path.exists(args.nige_model) and run_idx == 0:
        nige_model = PPO.load(args.nige_model, env=nige_env)
        print(f"Loaded nige model from {args.nige_model}")
    else:
        nige_model = PPO("MlpPolicy", nige_env, verbose=1)

    callbacks = []
    if args.render and args.num_envs > 1:
        raise ValueError("--render is only supported when --num-envs=1")
    if args.render and args.parallel == 1:
        callbacks.append(RenderCallback(nige_env, render_interval=args.render_interval))

    start = time.time()
    if args.num_envs == 1:
        nige_env.env.training_end_time = start + args.duration
    while time.time() - start < args.duration:
        nige_model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False, callback=callbacks)
    nige_model.save(nige_model_path)
    nige_env.close()


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
