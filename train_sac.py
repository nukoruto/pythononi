import argparse
import os
from typing import Tuple

import gymnasium as gym
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gym_tag_env import MultiTagEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Soft Actor-Critic training for MultiTagEnv"
    )
    parser.add_argument("--oni-model", type=str, default="oni_sac.pth", help="Path to save oni model")
    parser.add_argument("--nige-model", type=str, default="nige_sac.pth", help="Path to save nige model")
    parser.add_argument("--checkpoint-freq", type=int, default=0, help="Save checkpoints every N steps")
    parser.add_argument("--render", action="store_true", help="Render environment during training")
    parser.add_argument("--render-interval", type=int, default=1, help="Render every N steps")
    parser.add_argument("--duration", type=int, default=10, help="Episode length in seconds")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--speed-multiplier", type=float, default=1.0, help="Environment speed multiplier")
    parser.add_argument("--render-speed", type=float, default=1.0, help="Rendering speed multiplier")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini batch size")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--tau", type=float, default=0.005, help="Target update coefficient")
    parser.add_argument("--g", action="store_true", help="Use GPU if available")
    return parser.parse_args()


def _create_env(args: argparse.Namespace) -> MultiTagEnv:
    return MultiTagEnv(
        speed_multiplier=args.speed_multiplier,
        render_speed=args.render_speed,
    )


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.obs[idxs]),
            torch.tensor(self.actions[idxs]),
            torch.tensor(self.rewards[idxs]),
            torch.tensor(self.next_obs[idxs]),
            torch.tensor(self.dones[idxs]),
        )

    def __len__(self):
        return self.size


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mean(h), self.log_std(h)

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return y_t, log_prob

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action, _ = self.sample(obs_t.unsqueeze(0))
        return action.squeeze(0).cpu().numpy()


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


class SACAgent:
    def __init__(self, obs_dim: int, action_dim: int, args: argparse.Namespace, device: torch.device):
        self.device = device
        self.gamma = args.gamma
        self.tau = args.tau
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.q1 = Critic(obs_dim, action_dim).to(device)
        self.q2 = Critic(obs_dim, action_dim).to(device)
        self.q1_target = Critic(obs_dim, action_dim).to(device)
        self.q2_target = Critic(obs_dim, action_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=args.lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=args.lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=args.lr)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=args.lr)
        self.target_entropy = -float(action_dim)
        self.alpha = self.log_alpha.exp().item()

    def update(self, buffer: ReplayBuffer, batch_size: int) -> None:
        obs, actions, rewards, next_obs, dones = buffer.sample(batch_size)
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            q1_next = self.q1_target(next_obs, next_action)
            q2_next = self.q2_target(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target_q = rewards + (1 - dones) * self.gamma * q_next

        q1 = self.q1(obs, actions)
        q2 = self.q2(obs, actions)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()
        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        new_actions, log_prob = self.actor.sample(obs)
        q1_new = self.q1(obs, new_actions)
        q2_new = self.q2(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        soft_update(self.q1_target, self.q1, self.tau)
        soft_update(self.q2_target, self.q2, self.tau)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.actor.act(obs)

    def save(self, path: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }, path)



def run_training(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if args.g and torch.cuda.is_available() else "cpu")
    if args.g and device.type != "cuda":
        print("GPU is not available. Falling back to CPU.")
    print(f"Using device: {device}")

    env = _create_env(args)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    oni = SACAgent(obs_dim, action_dim, args, device)
    nige = SACAgent(obs_dim, action_dim, args, device)
    oni_buf = ReplayBuffer(args.buffer_size, obs_dim, action_dim)
    nige_buf = ReplayBuffer(args.buffer_size, obs_dim, action_dim)

    total_steps = 0
    for ep in range(1, args.episodes + 1):
        env.set_run_info(ep, args.episodes)
        env.set_training_end_time(time.time() + args.duration / args.speed_multiplier)
        obs, _ = env.reset()
        oni_obs, nige_obs = obs
        done = False
        while not done:
            oni_action = oni.act(oni_obs)
            nige_action = nige.act(nige_obs)
            (next_oni_obs, next_nige_obs), (r_o, r_n), terminated, truncated, _ = env.step((oni_action, nige_action))
            done = terminated or truncated
            oni_buf.add(oni_obs, oni_action, r_o, next_oni_obs, float(done))
            nige_buf.add(nige_obs, nige_action, r_n, next_nige_obs, float(done))
            oni_obs = next_oni_obs
            nige_obs = next_nige_obs
            total_steps += 1
            if args.render and total_steps % args.render_interval == 0:
                env.render()
            if len(oni_buf) >= args.batch_size:
                oni.update(oni_buf, args.batch_size)
            if len(nige_buf) >= args.batch_size:
                nige.update(nige_buf, args.batch_size)
            if args.checkpoint_freq > 0 and total_steps % args.checkpoint_freq == 0:
                oni.save(args.oni_model)
                nige.save(args.nige_model)
        if args.render:
            env.render()
        print(f"Episode {ep} finished")

    oni.save(args.oni_model)
    nige.save(args.nige_model)
    env.close()


def main():
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()

