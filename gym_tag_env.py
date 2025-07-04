# coding: utf-8
"""Gym environment for the tag game."""
import random
import time
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import torch

from stage_generator import generate_stage
from tag_game import (
    StageMap,
    Agent,
    CELL_SIZE,
    MAX_SPEED_BOOST,
    NIGE_MAX_SPEED,
    NIGE_ACCEL_STEPS,
    ONI_MAX_SPEED,
    ONI_ACCEL_STEPS,
)

INFO_PANEL_HEIGHT = 40


class MultiTagEnv(gym.Env):
    """Two-agent tag environment.

    ``step`` expects actions for both agents and returns vector observations and
    rewards as tuples ``(oni, nige)``. CNN observation tensors are provided via
    the info dictionary returned by :meth:`reset` and :meth:`step`. 逃げ側も強化
    学習の対象となります。
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        width: int = 31,
        height: int = 21,
        max_steps: int = 500,
        extra_wall_prob: float = 0.1,
        speed_multiplier: float = 1.0,
        render_speed: float = 1.0,
        start_distance_range: tuple[int, int] | None = None,
        width_range: tuple[int, int] | None = None,
        height_range: tuple[int, int] | None = None,
        fov_deg: float | None = 120.0,
    ) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.extra_wall_prob = extra_wall_prob
        self.fov_deg = fov_deg
        self.stage: StageMap | None = None
        self.oni: Agent | None = None
        self.nige: Agent | None = None
        low = np.array(
            [
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                0.0,
                -1.0,
                -1.0,
                0.0,
                0.0,
                -1.0,
                -1.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        high = np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )
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
        self.oni_history: list[tuple[float, float]] = []
        self.nige_history: list[tuple[float, float]] = []
        self.prev_pred_distance: float = 0.0

    def _is_visible(self, agent: Agent, opponent: Agent) -> bool:
        """Return True if ``opponent`` is within ``agent``'s FOV."""
        if self.fov_deg is None or self.fov_deg >= 360:
            return True
        vec = opponent.pos - agent.pos
        if vec.length_squared() == 0:
            return True
        facing = agent.facing if agent.facing.length_squared() > 0 else pygame.Vector2(1, 0)
        cos_angle = max(-1.0, min(1.0, facing.normalize().dot(vec.normalize())))
        angle = math.degrees(math.acos(cos_angle))
        return angle <= self.fov_deg / 2

    def _make_obs(
        self,
        agent: Agent,
        opponent: Agent,
        collided: bool,
    ) -> np.ndarray:
        """Return normalized observation vector for ``agent``."""
        assert self.stage
        # position normalized to [-1, 1]
        px = agent.pos.x / self.stage.width * 2.0 - 1.0
        py = agent.pos.y / self.stage.height * 2.0 - 1.0

        # velocity normalized to [-1, 1]
        v_scale = agent.max_speed + MAX_SPEED_BOOST
        vx = np.clip(agent.vel.x / v_scale, -1.0, 1.0)
        vy = np.clip(agent.vel.y / v_scale, -1.0, 1.0)

        collision = 1.0 if collided else 0.0

        visible = self._is_visible(agent, opponent)
        direction, dist = self.stage.shortest_path_info(agent.pos, opponent.pos)
        max_dist = self.stage.width + self.stage.height
        if visible:
            dir_x = direction.x
            dir_y = direction.y
            dist_norm = min(dist / max_dist, 1.0)
        else:
            dir_x = 0.0
            dir_y = 0.0
            dist_norm = 1.0
        capture_ease = 1.0 - dist_norm

        ov_scale = opponent.max_speed + MAX_SPEED_BOOST
        ovx = np.clip(opponent.vel.x / ov_scale, -1.0, 1.0)
        ovy = np.clip(opponent.vel.y / ov_scale, -1.0, 1.0)

        remain_ratio = max(
            0.0, 1.0 - self.physical_step_count / float(self.max_steps)
        )

        visible_flag = 1.0 if visible else 0.0

        return np.array(
            [
                px,
                py,
                vx,
                vy,
                collision,
                dir_x,
                dir_y,
                dist_norm,
                capture_ease,
                ovx,
                ovy,
                remain_ratio,
                visible_flag,
            ],
            dtype=np.float32,
        )

    def build_obs_tensor(
        self,
        agent: Agent,
        opponent: Agent,
        collided: bool,
        history: list[tuple[float, float]],
        remain_ratio: float,
    ) -> torch.Tensor:
        """Return multi-channel observation tensor for CNN input."""
        assert self.stage
        h, w = self.stage.height, self.stage.width
        tensor = np.zeros((17, h, w), dtype=np.float32)

        # channel 0: walls
        tensor[0] = np.array(self.stage.grid, dtype=np.float32)

        ax, ay = int(agent.pos.x), int(agent.pos.y)
        ox, oy = int(opponent.pos.x), int(opponent.pos.y)
        visible = self._is_visible(agent, opponent)

        # channel 1: agent position
        if 0 <= ax < w and 0 <= ay < h:
            tensor[1, ay, ax] = 1.0

        v_scale = agent.max_speed + MAX_SPEED_BOOST
        vx = np.clip(agent.vel.x / v_scale, -1.0, 1.0)
        vy = np.clip(agent.vel.y / v_scale, -1.0, 1.0)
        tensor[2].fill(vx)
        tensor[3].fill(vy)

        # channel 4: opponent position
        if visible and 0 <= ox < w and 0 <= oy < h:
            tensor[4, oy, ox] = 1.0

        ov_scale = opponent.max_speed + MAX_SPEED_BOOST
        ovx = np.clip(opponent.vel.x / ov_scale, -1.0, 1.0)
        ovy = np.clip(opponent.vel.y / ov_scale, -1.0, 1.0)
        tensor[5].fill(ovx)
        tensor[6].fill(ovy)

        # channel 7: collision flag
        if collided and 0 <= ax < w and 0 <= ay < h:
            tensor[7, ay, ax] = 1.0

        # channel 8-9: contact vector (approximate using -dir)
        if collided:
            tensor[8].fill(-agent.dir.x)
            tensor[9].fill(-agent.dir.y)

        # channel 10: remaining time ratio
        tensor[10].fill(remain_ratio)

        max_range = self.stage.width + self.stage.height
        if visible:
            tensor[11].fill((opponent.pos.x - agent.pos.x) / max_range)
            tensor[12].fill((opponent.pos.y - agent.pos.y) / max_range)
        else:
            tensor[11].fill(0.0)
            tensor[12].fill(0.0)

        # channel 13: predicted opponent position after two steps
        pred_x = int(opponent.pos.x + opponent.vel.x * 2)
        pred_y = int(opponent.pos.y + opponent.vel.y * 2)
        if visible and 0 <= pred_x < w and 0 <= pred_y < h:
            tensor[13, pred_y, pred_x] = 1.0

        # channel 14: movement history
        for idx, (hx, hy) in enumerate(reversed(history[-5:])):
            weight = (5 - idx) / 5.0
            hx_i, hy_i = int(hx), int(hy)
            if 0 <= hx_i < w and 0 <= hy_i < h:
                tensor[14, hy_i, hx_i] = weight

        # channel 15: capture ease
        direction, dist = self.stage.shortest_path_info(agent.pos, opponent.pos)
        max_range = self.stage.width + self.stage.height
        if visible:
            dist_norm = min(dist / max_range, 1.0)
        else:
            dist_norm = 1.0
        capture_ease = 1.0 - dist_norm
        tensor[15].fill(capture_ease)

        # channel 16: visibility flag
        tensor[16].fill(1.0 if visible else 0.0)

        return torch.tensor(tensor, dtype=torch.float32)

    def _get_obs(self, oni_collided: bool, nige_collided: bool):
        """Return vector and tensor observations for both agents."""
        assert self.oni and self.nige and self.stage
        remain_ratio = max(
            0.0, 1.0 - self.physical_step_count / float(self.max_steps)
        )
        oni_vec = self._make_obs(self.oni, self.nige, oni_collided)
        nige_vec = self._make_obs(self.nige, self.oni, nige_collided)
        oni_tensor = self.build_obs_tensor(
            self.oni,
            self.nige,
            oni_collided,
            self.oni_history,
            remain_ratio,
        )
        nige_tensor = self.build_obs_tensor(
            self.nige,
            self.oni,
            nige_collided,
            self.nige_history,
            remain_ratio,
        )
        return (oni_vec, oni_tensor), (nige_vec, nige_tensor)

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
        """Reset environment and return vector observations for both agents.

        The returned info dictionary contains CNN observation tensors under
        ``"oni_tensor"`` and ``"nige_tensor"``.
        """

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
        self.nige = Agent(
            nige_pos.x,
            nige_pos.y,
            (0, 100, 255),
            max_speed=NIGE_MAX_SPEED,
            accel_steps=NIGE_ACCEL_STEPS,
        )
        self.oni = Agent(
            oni_pos.x,
            oni_pos.y,
            (255, 0, 0),
            max_speed=ONI_MAX_SPEED,
            accel_steps=ONI_ACCEL_STEPS,
        )
        self.oni_history = [(self.oni.pos.x, self.oni.pos.y)]
        self.nige_history = [(self.nige.pos.x, self.nige.pos.y)]
        self.step_count = 0
        self.physical_step_count = 0
        self.cumulative_rewards = [0.0, 0.0]
        self.last_rewards = (0.0, 0.0)
        _, self.prev_distance = self.stage.shortest_path_info(
            self.oni.pos, self.nige.pos
        )
        pred_pos = pygame.Vector2(
            min(max(self.nige.pos.x + self.nige.vel.x * 2, 0), self.stage.width - 1),
            min(max(self.nige.pos.y + self.nige.vel.y * 2, 0), self.stage.height - 1),
        )
        _, self.prev_pred_distance = self.stage.shortest_path_info(self.oni.pos, pred_pos)
        obs = self._get_obs(False, False)
        obs_vec = (obs[0][0], obs[1][0])
        return obs_vec, {
            "oni_tensor": obs[0][1],
            "nige_tensor": obs[1][1],
        }

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

        prev_oni_pos = self.oni.pos.copy()
        prev_nige_pos = self.nige.pos.copy()
        _, prev_dist = self.stage.shortest_path_info(prev_oni_pos, prev_nige_pos)

        updates = max(1, int(round(self.speed_multiplier)))
        oni_collisions = 0
        nige_collisions = 0
        for _ in range(updates):
            if self.oni.update(self.stage):
                oni_collisions += 1
            if self.nige.update(self.stage):
                nige_collisions += 1
        self.physical_step_count += updates

        _, new_dist = self.stage.shortest_path_info(self.oni.pos, self.nige.pos)
        self.prev_distance = new_dist
        dist_delta = prev_dist - new_dist

        pred_pos = pygame.Vector2(
            min(max(self.nige.pos.x + self.nige.vel.x * 2, 0), self.stage.width - 1),
            min(max(self.nige.pos.y + self.nige.vel.y * 2, 0), self.stage.height - 1),
        )
        _, pred_dist = self.stage.shortest_path_info(self.oni.pos, pred_pos)
        pred_dist_delta = self.prev_pred_distance - pred_dist
        self.prev_pred_distance = pred_dist

        oni_move = self.oni.pos - prev_oni_pos
        nige_move = self.nige.pos - prev_nige_pos
        dir_on_to_ni, _ = self.stage.shortest_path_info(prev_oni_pos, prev_nige_pos)
        dir_ni_to_on, _ = self.stage.shortest_path_info(prev_nige_pos, prev_oni_pos)
        oni_align = 0.0
        nige_align = 0.0
        if oni_move.length_squared() > 0 and dir_on_to_ni.length_squared() > 0:
            oni_align = oni_move.normalize().dot(dir_on_to_ni)
        if nige_move.length_squared() > 0 and dir_ni_to_on.length_squared() > 0:
            nige_align = nige_move.normalize().dot(dir_ni_to_on)

        self.oni_history.append((self.oni.pos.x, self.oni.pos.y))
        self.nige_history.append((self.nige.pos.x, self.nige.pos.y))
        self.oni_history = self.oni_history[-5:]
        self.nige_history = self.nige_history[-5:]

        (oni_obs, oni_tensor), (nige_obs, nige_tensor) = self._get_obs(
            oni_collisions > 0,
            nige_collisions > 0,
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
            oni_reward += 0.01 * pred_dist_delta + 0.02 * oni_align
            if truncated:
                nige_reward = 2.0
                oni_reward -= 1.0
            else:
                nige_reward = 0.0
            nige_reward += 0.01 * (-dist_delta)
            nige_reward += -0.02 * nige_align + 0.002 * updates

        # penalty for wall collisions grows exponentially with the number of
        # hits in this step
        oni_penalty = 0.001 * (1.5 ** oni_collisions - 1.0)
        nige_penalty = 0.001 * (1.5 ** nige_collisions - 1.0)
        oni_reward -= oni_penalty
        nige_reward -= nige_penalty

        # reward is computed for ``updates`` physical steps so that the
        # total reward does not shrink when ``speed_multiplier`` is large

        self.last_rewards = (oni_reward, nige_reward)
        self.cumulative_rewards[0] += oni_reward
        self.cumulative_rewards[1] += nige_reward

        info = {
            "oni_tensor": oni_tensor,
            "nige_tensor": nige_tensor,
        }
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



