import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
from train import Policy

from gym_tag_env import MultiTagEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained agents")
    parser.add_argument("--oni-model", type=str, default="oni_selfplay.pth", help="Oni model path")
    parser.add_argument("--nige-model", type=str, default="nige_selfplay.pth", help="Nige model path")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument("--speed-multiplier", type=float, default=1.0, help="Environment speed multiplier")
    parser.add_argument("--g", action="store_true", help="Use GPU if available")
    return parser.parse_args()


def _select_action(policy: Policy, obs: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        mean, _ = policy(torch.tensor(obs, dtype=torch.float32, device=device))
    return mean.cpu().numpy()


def run_episode(
    env: MultiTagEnv,
    oni_model: Policy,
    nige_model: Policy,
    device: torch.device,
    render: bool,
) -> Tuple[float, float]:
    obs, _ = env.reset()
    oni_obs, nige_obs = obs
    done = False
    total_rewards = [0.0, 0.0]
    while not done:
        oni_action = _select_action(oni_model, oni_obs, device)
        nige_action = _select_action(nige_model, nige_obs, device)
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
    device = torch.device("cuda" if args.g and torch.cuda.is_available() else "cpu")
    if args.g and device.type != "cuda":
        print("GPU is not available. Falling back to CPU.")
    print(f"Using device: {device}")

    env = MultiTagEnv(speed_multiplier=args.speed_multiplier)
    input_dim = env.observation_space.shape[0]
    oni_model = Policy(input_dim=input_dim).to(device)
    oni_model.load_state_dict(torch.load(args.oni_model, map_location=device))
    oni_model.eval()
    nige_model = Policy(input_dim=input_dim).to(device)
    nige_model.load_state_dict(torch.load(args.nige_model, map_location=device))
    nige_model.eval()

    rewards: List[Tuple[float, float]] = []
    for i in range(args.episodes):
        env.set_run_info(i + 1, args.episodes)
        env.set_training_end_time(None)
        rewards.append(run_episode(env, oni_model, nige_model, device, args.render))

    env.close()
    avg_oni = np.mean([r[0] for r in rewards])
    avg_nige = np.mean([r[1] for r in rewards])
    print(f"Average rewards over {args.episodes} episodes -> oni: {avg_oni:.2f}, nige: {avg_nige:.2f}")


if __name__ == "__main__":
    main()
