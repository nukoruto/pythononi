import random
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

from gym_tag_env import MultiTagEnv
from stable_baselines3 import PPO


class EpisodeSwapEnv(gym.Env):
    """Wrapper that trains oni and nige alternately per episode."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        width: int = 31,
        height: int = 21,
        max_steps: int = 500,
        extra_wall_prob: float = 0.0,
        speed_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_env = MultiTagEnv(
            width=width,
            height=height,
            max_steps=max_steps,
            extra_wall_prob=extra_wall_prob,
            speed_multiplier=speed_multiplier,
        )
        self.training_agent = "oni"
        self.oni_model: Optional[PPO] = None
        self.nige_model: Optional[PPO] = None
        self.observation_space = self.base_env.action_space  # dummy, will reset in reset()
        self.action_space = self.base_env.action_space
        self._last_obs: Tuple[np.ndarray, np.ndarray] | None = None

    def set_training_agent(self, agent: str) -> None:
        assert agent in ("oni", "nige")
        self.training_agent = agent

    # delegate unknown attributes to base_env
    def __getattr__(self, name):
        return getattr(self.base_env, name)

    def reset(self, *, seed: int | None = None, options=None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        self._last_obs = obs
        if self.training_agent == "oni":
            self.observation_space = self.base_env.observation_space
            return obs[0], info
        else:
            self.observation_space = self.base_env.observation_space
            return obs[1], info

    def step(self, action: np.ndarray):
        assert self._last_obs is not None
        oni_action: np.ndarray
        nige_action: np.ndarray
        if self.training_agent == "oni":
            oni_action = action
            if self.nige_model is not None:
                nige_action, _ = self.nige_model.predict(self._last_obs[1], deterministic=True)
            else:
                nige_action = self.base_env.action_space.sample()
        else:
            nige_action = action
            if self.oni_model is not None:
                oni_action, _ = self.oni_model.predict(self._last_obs[0], deterministic=True)
            else:
                oni_action = self.base_env.action_space.sample()
        obs, rewards, terminated, truncated, info = self.base_env.step((oni_action, nige_action))
        self._last_obs = obs
        if self.training_agent == "oni":
            return obs[0], rewards[0], terminated, truncated, info
        else:
            return obs[1], rewards[1], terminated, truncated, info

    def render(self):
        self.base_env.render()

    def close(self):
        self.base_env.close()
