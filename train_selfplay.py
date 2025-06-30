import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from gym_tag_env import MultiTagEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Self-play training for oni and nige")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--speed-multiplier", type=float, default=1.0)
    return parser.parse_args()


class Policy(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 2):
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
        return action, log_prob


def compute_returns(rewards, gamma):
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def train():
    args = parse_args()
    env = MultiTagEnv(speed_multiplier=args.speed_multiplier)
    oni_policy = Policy()
    nige_policy = Policy()
    oni_optim = optim.Adam(oni_policy.parameters(), lr=args.lr)
    nige_optim = optim.Adam(nige_policy.parameters(), lr=args.lr)

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        oni_obs, nige_obs = obs
        oni_log_probs = []
        nige_log_probs = []
        oni_rewards = []
        nige_rewards = []
        done = False
        while not done:
            oni_action, oni_logp = oni_policy.act(torch.tensor(oni_obs, dtype=torch.float32))
            nige_action, nige_logp = nige_policy.act(torch.tensor(nige_obs, dtype=torch.float32))
            (oni_obs, nige_obs), (r_o, r_n), terminated, truncated, _ = env.step((oni_action.detach().numpy(), nige_action.detach().numpy()))
            if args.render:
                env.render()
            oni_log_probs.append(oni_logp)
            nige_log_probs.append(nige_logp)
            oni_rewards.append(r_o)
            nige_rewards.append(r_n)
            done = terminated or truncated

        oni_returns = torch.tensor(compute_returns(oni_rewards, args.gamma), dtype=torch.float32)
        nige_returns = torch.tensor(compute_returns(nige_rewards, args.gamma), dtype=torch.float32)
        oni_loss = -(torch.stack(oni_log_probs) * oni_returns).sum()
        nige_loss = -(torch.stack(nige_log_probs) * nige_returns).sum()

        oni_optim.zero_grad()
        oni_loss.backward()
        oni_optim.step()

        nige_optim.zero_grad()
        nige_loss.backward()
        nige_optim.step()

        if ep % 10 == 0:
            print(f"episode {ep}: oniR={sum(oni_rewards):.2f} nigeR={sum(nige_rewards):.2f}")

    torch.save(oni_policy.state_dict(), "oni_selfplay.pth")
    torch.save(nige_policy.state_dict(), "nige_selfplay.pth")
    env.close()


if __name__ == "__main__":
    train()
