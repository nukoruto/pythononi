import argparse
import os
from typing import List

import numpy as np
from stable_baselines3 import PPO

from gym_tag_env import TagEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained agent")
    parser.add_argument("--model", type=str, default="ppo_tag.zip", help="Model path")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Render environment")
    return parser.parse_args()


def run_episode(env: TagEnv, model: PPO, render: bool) -> float:
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        if render:
            env.render()
    return total_reward


def main():
    args = parse_args()
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    env = TagEnv()
    model = PPO.load(args.model, env=env)

    rewards: List[float] = []
    for _ in range(args.episodes):
        rewards.append(run_episode(env, model, args.render))

    env.close()
    print(f"Average reward over {args.episodes} episodes: {np.mean(rewards):.2f}")


if __name__ == "__main__":
    main()
