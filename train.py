import argparse
import os
import re
import time

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from typing import Any, Optional

from gym_tag_env import MultiTagEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train agents for MultiTagEnv")
    parser.add_argument("--oni-model", type=str, default="oni_policy.zip", help="Path to save/load oni model")
    parser.add_argument("--nige-model", type=str, default="nige_policy.zip", help="Path to save/load nige model")
    parser.add_argument("--checkpoint-freq", type=int, default=0, help="Save checkpoints every N steps")
    parser.add_argument("--render", action="store_true", help="Render environment during training")
    parser.add_argument("--render-interval", type=int, default=1, help="Render every N steps when --render is set")
    parser.add_argument("--duration", type=int, default=10, help="Training duration per episode in seconds")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--speed-multiplier", type=float, default=1.0, help="Environment speed multiplier")
    parser.add_argument("--render-speed", type=float, default=1.0, help="Rendering speed multiplier")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for self-play")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for self-play")
    parser.add_argument("--entropy-coeff", type=float, default=0.01, help="Entropy regularization coefficient")
    parser.add_argument("--g", action="store_true", help="Use GPU if available")
    return parser.parse_args()


def _create_env(args: argparse.Namespace) -> MultiTagEnv:
    """Create :class:`MultiTagEnv` with the specified speed."""
    return MultiTagEnv(
        speed_multiplier=args.speed_multiplier,
        render_speed=args.render_speed,
    )
class Policy(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return self.mean(h), self.log_std.exp()

    def act(self, obs: torch.Tensor):
        mean, std = self(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy


def compute_returns(rewards, gamma: float):
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def _next_output_path(base_dir: str, prefix: str) -> str:
    """Return next sequential path like ``prefix_N.pth`` under ``base_dir``."""
    os.makedirs(base_dir, exist_ok=True)
    max_idx = 0
    for name in os.listdir(base_dir):
        if name.startswith(f"{prefix}_") and name.endswith(".pth"):
            m = re.search(rf"{prefix}_(\d+)\.pth", name)
            if m:
                try:
                    idx = int(m.group(1))
                    max_idx = max(max_idx, idx)
                except ValueError:
                    continue
    return os.path.join(base_dir, f"{prefix}_{max_idx + 1}.pth")


def run_selfplay(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if args.g and torch.cuda.is_available() else "cpu")
    if args.g and device.type != "cuda":
        print("GPU is not available. Falling back to CPU.")
    print(f"Using device: {device}")

    env = _create_env(args)
    input_dim = env.observation_space.shape[0]
    oni_policy = Policy(input_dim=input_dim).to(device)
    nige_policy = Policy(input_dim=input_dim).to(device)
    oni_optim = optim.Adam(oni_policy.parameters(), lr=args.lr)
    nige_optim = optim.Adam(nige_policy.parameters(), lr=args.lr)

    oni_episode_rewards = []
    nige_episode_rewards = []

    for ep in range(1, args.episodes + 1):
        env.set_run_info(ep, args.episodes)
        env.set_training_end_time(
            time.time() + args.duration / args.speed_multiplier
        )
        obs, _ = env.reset()
        oni_obs, nige_obs = obs
        oni_log_probs = []
        oni_entropies = []
        nige_log_probs = []
        nige_entropies = []
        oni_rewards = []
        nige_rewards = []
        done = False
        while not done:
            oni_action, oni_logp, oni_ent = oni_policy.act(
                torch.tensor(oni_obs, dtype=torch.float32, device=device)
            )
            nige_action, nige_logp, nige_ent = nige_policy.act(
                torch.tensor(nige_obs, dtype=torch.float32, device=device)
            )
            (oni_obs, nige_obs), (r_o, r_n), terminated, truncated, _ = env.step((
                oni_action.detach().cpu().numpy(),
                nige_action.detach().cpu().numpy(),
            ))
            if args.render:
                env.render()
            oni_log_probs.append(oni_logp)
            oni_entropies.append(oni_ent)
            nige_log_probs.append(nige_logp)
            nige_entropies.append(nige_ent)
            oni_rewards.append(r_o)
            nige_rewards.append(r_n)
            done = terminated or truncated

        oni_returns = torch.tensor(
            compute_returns(oni_rewards, args.gamma), dtype=torch.float32, device=device
        )
        nige_returns = torch.tensor(
            compute_returns(nige_rewards, args.gamma), dtype=torch.float32, device=device
        )
        oni_entropy = torch.stack(oni_entropies)
        nige_entropy = torch.stack(nige_entropies)
        oni_loss = (
            -(torch.stack(oni_log_probs) * oni_returns).sum()
            - args.entropy_coeff * oni_entropy.sum()
        )
        nige_loss = (
            -(torch.stack(nige_log_probs) * nige_returns).sum()
            - args.entropy_coeff * nige_entropy.sum()
        )

        oni_optim.zero_grad()
        oni_loss.backward()
        oni_optim.step()

        nige_optim.zero_grad()
        nige_loss.backward()
        nige_optim.step()

        total_oni = sum(oni_rewards)
        total_nige = sum(nige_rewards)
        oni_episode_rewards.append(total_oni)
        nige_episode_rewards.append(total_nige)

        if ep % 10 == 0:
            avg_oni = sum(oni_episode_rewards[-10:]) / 10
            avg_nige = sum(nige_episode_rewards[-10:]) / 10
            print(
                f"episode {ep}: average oniR={avg_oni:.2f} average nigeR={avg_nige:.2f}"
            )

    oni_path = _next_output_path("out/oni", "out")
    nige_path = _next_output_path("out/nige", "nige")
    torch.save(oni_policy.state_dict(), oni_path)
    torch.save(nige_policy.state_dict(), nige_path)
    env.close()

def main():
    args = parse_args()
    run_selfplay(args)



if __name__ == "__main__":
    main()
