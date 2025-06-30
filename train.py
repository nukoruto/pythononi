import argparse
import os

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from gym_tag_env import MultiTagEnv


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
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for self-play")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for self-play")
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
            self.env.remaining_time = max(
                0.0,
                (self.env.training_end_time - time.time()) * getattr(self.env, "speed_multiplier", 1.0),
            )
        if self.n_calls % self.render_interval == 0:
            self.env.render()
        return True





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


def compute_returns(rewards, gamma: float):
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def run_selfplay(args: argparse.Namespace) -> None:
    env = MultiTagEnv(speed_multiplier=args.speed_multiplier)
    oni_policy = Policy()
    nige_policy = Policy()
    oni_optim = optim.Adam(oni_policy.parameters(), lr=args.lr)
    nige_optim = optim.Adam(nige_policy.parameters(), lr=args.lr)

    for ep in range(1, args.episodes + 1):
        env.set_run_info(ep, args.episodes)
        import time
        scaled_duration = args.duration / args.speed_multiplier
        env.set_training_end_time(time.time() + scaled_duration)
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
            (oni_obs, nige_obs), (r_o, r_n), terminated, truncated, _ = env.step((
                oni_action.detach().numpy(),
                nige_action.detach().numpy(),
            ))
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

    torch.save(oni_policy.state_dict(), args.oni_model.replace('.zip', '_selfplay.pth'))
    torch.save(nige_policy.state_dict(), args.nige_model.replace('.zip', '_selfplay.pth'))
    env.close()


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

    if isinstance(env, VecEnv):
        env.set_attr("oni_model", oni_model)
        env.set_attr("nige_model", nige_model)
    else:
        env.oni_model = oni_model
        env.nige_model = nige_model

    for ep in range(args.episodes):
        if isinstance(env, VecEnv):
            env.env_method("set_run_info", ep + 1, args.episodes)
            training_agents = env.get_attr("training_agent")
            train_oni = training_agents[0] == "oni"
        else:
            env.set_run_info(ep + 1, args.episodes)
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
        scaled_duration = args.duration / args.speed_multiplier
        if isinstance(env, VecEnv):
            env.env_method("set_training_end_time", start + scaled_duration)
        else:
            env.set_training_end_time(start + scaled_duration)
        model = oni_model if train_oni else nige_model
        while time.time() - start < scaled_duration:
            model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False, callback=callbacks)

        # Start new episode and swap training agent automatically
        env.reset()

    oni_model.save(oni_model_path)
    nige_model.save(nige_model_path)
    env.close()




def main():
    args = parse_args()
    run_selfplay(args)



if __name__ == "__main__":
    main()
