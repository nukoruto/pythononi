# coding: utf-8
"""Gym environment for the tag game."""
import math
import random
from typing import Tuple, List

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from stage_generator import generate_stage
from tag_game import StageMap, Agent, CELL_SIZE


class TagEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        width: int = 31,
        height: int = 21,
        max_steps: int = 500,
        extra_wall_prob: float = 0.0,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.extra_wall_prob = extra_wall_prob
        self.stage: StageMap | None = None
        self.oni: Agent | None = None
        self.nige: Agent | None = None
        low = np.array([-width, -height, 0], dtype=np.float32)
        high = np.array([width, height, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.step_count = 0
        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.stage = StageMap(
            self.width,
            self.height,
            extra_wall_prob=self.extra_wall_prob,
            rng=self.np_random,
        )
        self.oni = Agent(1.5, 1.5, (255, 0, 0))
        self.nige = Agent(self.width - 2, self.height - 2, (0, 100, 255))
        self.step_count = 0
        return np.array(self.oni.observe(self.nige), dtype=np.float32), {}

    def step(self, action: np.ndarray):
        assert self.oni and self.nige and self.stage
        self.step_count += 1
        dx, dy = float(action[0]), float(action[1])
        self.oni.set_direction(dx, dy)
        # random policy for escapee
        rnd = self.np_random.uniform(-1, 1, size=2)
        self.nige.set_direction(float(rnd[0]), float(rnd[1]))
        self.oni.update(self.stage)
        self.nige.update(self.stage)

        obs = np.array(self.oni.observe(self.nige), dtype=np.float32)
        terminated = self.oni.collides_with(self.nige)
        truncated = self.step_count >= self.max_steps
        reward = 1.0 if terminated else -0.01
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width * CELL_SIZE, self.height * CELL_SIZE))
            self.clock = pygame.time.Clock()
        assert self.stage and self.oni and self.nige
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.screen = None
                return
        self.screen.fill((0, 0, 0))
        self.stage.draw(self.screen)
        self.oni.draw(self.screen)
        self.nige.draw(self.screen)
        if self.oni.can_see(self.nige):
            pygame.draw.line(
                self.screen,
                (255, 0, 0),
                (
                    int(self.oni.pos.x * CELL_SIZE + CELL_SIZE / 2),
                    int(self.oni.pos.y * CELL_SIZE + CELL_SIZE / 2),
                ),
                (
                    int(self.nige.pos.x * CELL_SIZE + CELL_SIZE / 2),
                    int(self.nige.pos.y * CELL_SIZE + CELL_SIZE / 2),
                ),
                2,
            )
        pygame.display.flip()
        if self.clock:
            self.clock.tick(60)

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None
            self.clock = None
