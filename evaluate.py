import argparse
import os
from typing import List, Tuple

import numpy as np
from stable_baselines3 import PPO

from gym_tag_env import MultiTagEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained agents")
    parser.add_argument("--oni-model", type=str, default="oni_policy.zip", help="Oni model path")
    parser.add_argument("--nige-model", type=str, default="nige_policy.zip", help="Nige model path")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument("--speed-multiplier", type=float, default=1.0, help="Environment speed multiplier")
    return parser.parse_args()


def run_episode(env: MultiTagEnv, oni_model: PPO, nige_model: PPO, render: bool) -> Tuple[float, float]:
    obs, _ = env.reset()
    oni_obs, nige_obs = obs
    done = False
    total_rewards = [0.0, 0.0]
    while not done:
        oni_action, _ = oni_model.predict(oni_obs, deterministic=True)
        nige_action, _ = nige_model.predict(nige_obs, deterministic=True)
        (oni_obs, nige_obs), (r_on, r_ni), terminated, truncated, _ = env.step((oni_action, nige_action))
        done = terminated or truncated
        total_rewards[0] += r_on
        total_rewards[1] += r_ni
        if render:
            env.render()
    return total_rewards[0], total_rewards[1]


def main():
    args = parse_args()
    if not os.path.exists(args.oni_model):
        raise FileNotFoundError(f"Model not found: {args.oni_model}")
    if not os.path.exists(args.nige_model):
        raise FileNotFoundError(f"Model not found: {args.nige_model}")
    env = MultiTagEnv(speed_multiplier=args.speed_multiplier)
    oni_model = PPO.load(args.oni_model, env=env)
    nige_model = PPO.load(args.nige_model, env=env)

    rewards: List[Tuple[float, float]] = []
    for _ in range(args.episodes):
        rewards.append(run_episode(env, oni_model, nige_model, args.render))

    env.close()
    avg_oni = np.mean([r[0] for r in rewards])
    avg_nige = np.mean([r[1] for r in rewards])
    print(f"Average rewards over {args.episodes} episodes -> oni: {avg_oni:.2f}, nige: {avg_nige:.2f}")


if __name__ == "__main__":
    main()
