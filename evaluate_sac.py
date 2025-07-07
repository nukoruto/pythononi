import argparse
import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from train_sac import Actor, CNNActor
from gym_tag_env import MultiTagEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SAC trained agents")
    parser.add_argument(
        "--oni",
        "-O",
        type=str,
        default="oni_sac.pth",
        help="Oni model path",
    )
    parser.add_argument(
        "--nige",
        "-N",
        type=str,
        default="nige_sac.pth",
        help="Nige model path",
    )
    parser.add_argument("--eps", "--episodes", dest="episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--draw", "--render", dest="render", action="store_true", help="Render environment")
    parser.add_argument("--speed", "--speed-multiplier", dest="speed_multiplier", type=float, default=1.0, help="Environment speed multiplier")
    parser.add_argument("--draw-speed", "--render-speed", dest="render_speed", type=float, default=1.0, help="Rendering speed multiplier")
    parser.add_argument("--gpu", "--g", dest="g", action="store_true", help="Use GPU if available")
    parser.add_argument(
        "--out",
        "--output-dir",
        dest="output_dir",
        type=str,
        default="eval",
        help="Base directory to store evaluation logs",
    )
    parser.add_argument("--cnn", "--use-cnn", dest="use_cnn", action="store_true", help="Use CNN-based models")
    return parser.parse_args()


def _timestamp_output_dir(base_dir: str) -> str:
    """Create and return ``base_dir/YYYYMMDD_HHMMSS`` directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, ts)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def run_episode(
    env: MultiTagEnv,
    oni_actor: Actor,
    nige_actor: Actor,
    render: bool,
    use_cnn: bool,
) -> Tuple[float, float]:
    obs, info = env.reset()
    oni_obs, nige_obs = obs
    oni_tensor = info["oni_tensor"]
    nige_tensor = info["nige_tensor"]
    done = False
    total_rewards = [0.0, 0.0]
    while not done:
        with torch.no_grad():
            if use_cnn:
                oni_action = oni_actor.act(oni_obs, oni_tensor)
                nige_action = nige_actor.act(nige_obs, nige_tensor)
            else:
                oni_action = oni_actor.act(oni_obs)
                nige_action = nige_actor.act(nige_obs)
        (oni_obs, nige_obs), (r_on, r_ni), terminated, truncated, info = env.step((oni_action, nige_action))
        oni_tensor = info["oni_tensor"]
        nige_tensor = info["nige_tensor"]
        done = terminated or truncated
        total_rewards[0] += r_on
        total_rewards[1] += r_ni
        if render:
            env.render()
    return total_rewards[0], total_rewards[1]


def main():
    args = parse_args()
    if not os.path.exists(args.oni):
        raise FileNotFoundError(f"Model not found: {args.oni}")
    if not os.path.exists(args.nige):
        raise FileNotFoundError(f"Model not found: {args.nige}")
    device = torch.device("cuda" if args.g and torch.cuda.is_available() else "cpu")
    if args.g and device.type != "cuda":
        print("GPU is not available. Falling back to CPU.")
    print(f"Using device: {device}")

    env = MultiTagEnv(
        speed_multiplier=args.speed_multiplier,
        render_speed=args.render_speed,
    )
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor_cls = CNNActor if args.use_cnn else Actor
    oni_actor = actor_cls(obs_dim, action_dim).to(device)
    oni_state = torch.load(args.oni, map_location=device)
    oni_actor.load_state_dict(oni_state["actor"])
    oni_actor.eval()

    nige_actor = actor_cls(obs_dim, action_dim).to(device)
    nige_state = torch.load(args.nige, map_location=device)
    nige_actor.load_state_dict(nige_state["actor"])
    nige_actor.eval()

    output_dir = _timestamp_output_dir(args.output_dir)
    rewards: List[Tuple[float, float]] = []
    for i in range(args.episodes):
        env.set_run_info(i + 1, args.episodes)
        env.set_training_end_time(None)
        rewards.append(run_episode(env, oni_actor, nige_actor, args.render, args.use_cnn))

    env.close()

    rewards_df = pd.DataFrame({
        "episode": range(1, args.episodes + 1),
        "oni_reward": [r[0] for r in rewards],
        "nige_reward": [r[1] for r in rewards],
    })
    rewards_df.to_csv(os.path.join(output_dir, "rewards.csv"), index=False)

    plt.figure()
    plt.plot(rewards_df["episode"], rewards_df["oni_reward"], label="oni")
    plt.plot(rewards_df["episode"], rewards_df["nige_reward"], label="nige")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation_curve.png"))
    plt.close()

    avg_oni = rewards_df["oni_reward"].mean()
    std_oni = rewards_df["oni_reward"].std()
    avg_nige = rewards_df["nige_reward"].mean()
    std_nige = rewards_df["nige_reward"].std()
    print(
        f"Evaluation over {args.episodes} episodes -> "
        f"oni: {avg_oni:.2f} ± {std_oni:.2f}, "
        f"nige: {avg_nige:.2f} ± {std_nige:.2f}"
    )


if __name__ == "__main__":
    main()
