import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from gym_tag_env import TagEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent for TagEnv")
    parser.add_argument("--timesteps", type=int, default=10000, help="Training steps per run")
    parser.add_argument("--model", type=str, default="ppo_tag.zip", help="Path to save/load model")
    parser.add_argument("--checkpoint-freq", type=int, default=0, help="Save checkpoints every N steps")
    return parser.parse_args()


def main():
    args = parse_args()
    env = TagEnv()

    if os.path.exists(args.model):
        model = PPO.load(args.model, env=env)
        print(f"Loaded model from {args.model}")
    else:
        model = PPO("MlpPolicy", env, verbose=1)

    callbacks = []
    if args.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(save_freq=args.checkpoint_freq, save_path=".", name_prefix="ppo_checkpoint")
        )

    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False, callback=callbacks)
    model.save(args.model)
    print(f"Model saved to {args.model}")


if __name__ == "__main__":
    main()
