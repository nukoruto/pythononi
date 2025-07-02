import argparse
import os
import re
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from typing import Any, Optional

from gym_tag_env import MultiTagEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train agents for TagEnv")
    parser.add_argument("--timesteps", type=int, default=10000, help="Training steps per episode")
    parser.add_argument("--oni-model", type=str, default="oni_policy.zip", help="Path to save/load oni model")
    parser.add_argument("--nige-model", type=str, default="nige_policy.zip", help="Path to save/load nige model")
    parser.add_argument("--checkpoint-freq", type=int, default=0, help="Save checkpoints every N steps")
    parser.add_argument("--render", action="store_true", help="Render environment during training")
    parser.add_argument("--render-interval", type=int, default=1, help="Render every N steps when --render is set")
    parser.add_argument("--duration", type=int, default=10, help="Training duration per episode in seconds")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
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


def _make_vec_env(args: argparse.Namespace) -> gym.vector.VectorEnv:
    """Factory for vectorized environments."""

    def _factory() -> MultiTagEnv:
        return _create_env(args)

    env_fns = [_factory for _ in range(args.num_envs)]
    if args.num_envs > 1:
        return gym.vector.AsyncVectorEnv(env_fns)
    else:
        return gym.vector.SyncVectorEnv(env_fns)
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


def compute_returns_batch(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Compute discounted returns for batched rewards."""
    returns = np.zeros_like(rewards)
    R = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R
        returns[t] = R
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

    env = _make_vec_env(args)
    input_dim = env.single_observation_space[0].shape[0]
    oni_policy = Policy(input_dim=input_dim).to(device)
    nige_policy = Policy(input_dim=input_dim).to(device)
    oni_optim = optim.Adam(oni_policy.parameters(), lr=args.lr)
    nige_optim = optim.Adam(nige_policy.parameters(), lr=args.lr)

    oni_episode_rewards = []
    nige_episode_rewards = []

    for ep in range(1, args.episodes + 1):
        env.env_method("set_run_info", ep, args.episodes)
        env.env_method(
            "set_training_end_time",
            time.time() + args.duration / args.speed_multiplier,
        )
        obs, _ = env.reset()
        oni_obs, nige_obs = obs
        oni_log_probs: list[torch.Tensor] = []
        oni_entropies: list[torch.Tensor] = []
        nige_log_probs: list[torch.Tensor] = []
        nige_entropies: list[torch.Tensor] = []
        oni_rewards: list[np.ndarray] = []
        nige_rewards: list[np.ndarray] = []
        done = np.zeros(args.num_envs, dtype=bool)
        while not done.all():
            oni_action, oni_logp, oni_ent = oni_policy.act(
                torch.tensor(oni_obs, dtype=torch.float32, device=device)
            )
            nige_action, nige_logp, nige_ent = nige_policy.act(
                torch.tensor(nige_obs, dtype=torch.float32, device=device)
            )
            (oni_obs, nige_obs), (r_o, r_n), terminated, truncated, _ = env.step(
                (
                    oni_action.detach().cpu().numpy(),
                    nige_action.detach().cpu().numpy(),
                )
            )
            if args.render:
                env.envs[0].render()
            oni_log_probs.append(oni_logp)
            oni_entropies.append(oni_ent)
            nige_log_probs.append(nige_logp)
            nige_entropies.append(nige_ent)
            oni_rewards.append(r_o)
            nige_rewards.append(r_n)
            done = np.logical_or(terminated, truncated)

        oni_rewards_np = np.stack(oni_rewards)
        nige_rewards_np = np.stack(nige_rewards)
        oni_returns = torch.tensor(
            compute_returns_batch(oni_rewards_np, args.gamma),
            dtype=torch.float32,
            device=device,
        )
        nige_returns = torch.tensor(
            compute_returns_batch(nige_rewards_np, args.gamma),
            dtype=torch.float32,
            device=device,
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
