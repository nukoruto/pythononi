# coding: utf-8
"""Gym environment for the tag game."""
import random
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from stage_generator import generate_stage
from tag_game import StageMap, Agent, CELL_SIZE

INFO_PANEL_HEIGHT = 40


class MultiTagEnv(gym.Env):
    """Two-agent tag environment.

    ``step`` expects actions for both agents and returns observations and
    rewards as tuples ``(oni, nige)``. 逃げ側も強化学習の対象となります。
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        width: int = 31,
        height: int = 21,
        max_steps: int = 500,
        extra_wall_prob: float = 0.0,
        speed_multiplier: float = 1.0,
        render_speed: float = 1.0,
        start_distance_range: tuple[int, int] | None = None,
        width_range: tuple[int, int] | None = None,
        height_range: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.extra_wall_prob = extra_wall_prob
        self.stage: StageMap | None = None
        self.oni: Agent | None = None
        self.nige: Agent | None = None
        low = np.array([-1.0, -1.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.step_count = 0
        self.physical_step_count = 0
        self.speed_multiplier = max(0.1, speed_multiplier)
        self.render_speed = max(0.1, render_speed)
        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.cumulative_rewards: list[float] = [0.0, 0.0]
        self.last_rewards: tuple[float, float] = (0.0, 0.0)
        self.remaining_time: float = 0.0
        self.current_run: int = 0
        self.total_runs: int = 1
        self.training_end_time: float | None = None
        self.start_distance_range = start_distance_range
        self.width_range = width_range
        self.height_range = height_range

    def set_run_info(self, current_run: int, total_runs: int) -> None:
        """Set current episode index and total runs for rendering."""
        self.current_run = current_run
        self.total_runs = total_runs

    def set_training_end_time(self, end_time: float | None) -> None:
        """Set training end time used by the renderer."""
        self.training_end_time = end_time

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[tuple[np.ndarray, np.ndarray], dict]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.stage = StageMap(
            self.width,
            self.height,
            extra_wall_prob=self.extra_wall_prob,
            width_range=self.width_range,
            height_range=self.height_range,
            rng=self.np_random,
        )
        self.width = self.stage.width
        self.height = self.stage.height
        if self.screen is not None:
            self.screen = pygame.display.set_mode(
                (self.width * CELL_SIZE, self.height * CELL_SIZE + INFO_PANEL_HEIGHT)
            )
        oni_pos = self.stage.random_open_position()
        nige_pos = self.stage.random_open_position()
        if self.start_distance_range is not None:
            min_d, max_d = self.start_distance_range
            max_d = max_d or (self.width + self.height)
            _, d = self.stage.shortest_path_info(oni_pos, nige_pos)
            tries = 0
            while not (min_d <= d <= max_d) and tries < 100:
                nige_pos = self.stage.random_open_position()
                _, d = self.stage.shortest_path_info(oni_pos, nige_pos)
                tries += 1
        self.oni = Agent(oni_pos.x, oni_pos.y, (255, 0, 0))
        self.nige = Agent(nige_pos.x, nige_pos.y, (0, 100, 255))
        self.step_count = 0
        self.physical_step_count = 0
        self.cumulative_rewards = [0.0, 0.0]
        self.last_rewards = (0.0, 0.0)
        _, self.prev_distance = self.stage.shortest_path_info(
            self.oni.pos, self.nige.pos
        )
        obs = (
            np.array(self.oni.observe(self.nige, self.stage), dtype=np.float32),
            np.array(self.nige.observe(self.oni, self.stage), dtype=np.float32),
        )
        return obs, {}

    def step(
        self, actions: tuple[np.ndarray, np.ndarray]
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[float, float], bool, bool, dict]:
        assert self.oni and self.nige and self.stage
        action_oni, action_nige = actions
        self.step_count += 1
        truncated_by_time = False
        if self.training_end_time is not None:
            now = time.time()
            self.remaining_time = max(
                0.0,
                (self.training_end_time - now) * self.speed_multiplier,
            )
            truncated_by_time = now >= self.training_end_time

        odx, ody = float(action_oni[0]), float(action_oni[1])
        ndx, ndy = float(action_nige[0]), float(action_nige[1])
        self.oni.set_direction(odx, ody)
        self.nige.set_direction(ndx, ndy)

        _, prev_dist = self.stage.shortest_path_info(self.oni.pos, self.nige.pos)

        updates = max(1, int(round(self.speed_multiplier)))
        for _ in range(updates):
            self.oni.update(self.stage)
            self.nige.update(self.stage)
        self.physical_step_count += updates

        _, new_dist = self.stage.shortest_path_info(self.oni.pos, self.nige.pos)
        self.prev_distance = new_dist
        dist_delta = prev_dist - new_dist

        oni_obs = np.array(
            self.oni.observe(self.nige, self.stage), dtype=np.float32
        )
        nige_obs = np.array(
            self.nige.observe(self.oni, self.stage), dtype=np.float32
        )

        terminated = self.oni.collides_with(self.nige)
        use_step_limit = self.training_end_time is None
        truncated_by_steps = use_step_limit and self.physical_step_count >= self.max_steps
        truncated = truncated_by_steps or truncated_by_time

        if terminated:
            remain_ratio = (
                self.max_steps - self.physical_step_count
            ) / self.max_steps
            oni_reward = 2.0 + remain_ratio
            nige_reward = -2.0 * (
                1.0 - self.physical_step_count / self.max_steps
            )

        else:
            oni_reward = -0.005 * updates + 0.01 * dist_delta
            if truncated:
                nige_reward = 1.0
            else:
                nige_reward = 0.0
            nige_reward += 0.01 * (-dist_delta) + 0.002 * updates

        # reward is computed for ``updates`` physical steps so that the
        # total reward does not shrink when ``speed_multiplier`` is large

        self.last_rewards = (oni_reward, nige_reward)
        self.cumulative_rewards[0] += oni_reward
        self.cumulative_rewards[1] += nige_reward

        info = {}
        return (oni_obs, nige_obs), (oni_reward, nige_reward), terminated, truncated, info

    def render(self) -> None:
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.width * CELL_SIZE, self.height * CELL_SIZE + INFO_PANEL_HEIGHT)
            )
            self.clock = pygame.time.Clock()
        assert self.stage and self.oni and self.nige
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.screen = None
                return
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(
            self.screen,
            (255, 255, 255),
            pygame.Rect(0, 0, self.width * CELL_SIZE, INFO_PANEL_HEIGHT),
        )
        offset = (0, INFO_PANEL_HEIGHT)
        self.stage.draw(self.screen, offset)
        self.oni.draw(self.screen, offset)
        self.nige.draw(self.screen, offset)
        self.stage.draw_shortest_path_vectors(
            self.screen,
            self.oni.pos,
            self.nige.pos,
            offset=offset,
        )
        font = pygame.font.SysFont(None, 24)
        txt_time = font.render(
            f"Time:{self.remaining_time:.2f}s", True, (0, 0, 0)
        )
        txt_run = font.render(
            f"Run:{self.current_run}/{self.total_runs}", True, (0, 0, 0)
        )
        txt_reward = font.render(
            f"O:{self.cumulative_rewards[0]:.2f} N:{self.cumulative_rewards[1]:.2f}",
            True,
            (0, 0, 0),
        )
        self.screen.blit(txt_time, (10, 5))
        self.screen.blit(txt_run, (160, 5))
        self.screen.blit(txt_reward, (10, 25))
        pygame.display.flip()
        if self.clock:
            self.clock.tick(60 * self.render_speed)



