import math
from typing import Tuple, List

import argparse
import time

import pygame

from stage_generator import generate_stage, Stage
import numpy as np


CELL_SIZE = 20
EXTRA_WALL_PROB = 0.1
INFO_PANEL_HEIGHT = 40
DEFAULT_DURATION = 10.0

# 加速に関するマジックナンバー
CONTINUOUS_ACCEL = 0.01
MAX_SPEED_BOOST = 0.3


class StageMap:
    def __init__(
        self,
        width: int,
        height: int,
        extra_wall_prob: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        self.rng = rng or np.random.default_rng()
        self.grid = generate_stage(width, height, extra_wall_prob=extra_wall_prob, rng=self.rng)
        self.width = width
        self.height = height

    def is_wall(self, x: int, y: int) -> bool:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True
        return self.grid[y][x] == 1

    def collides_circle(self, x: float, y: float, radius: float) -> bool:
        """Return True if a circle (in grid units) intersects a wall."""
        left = int(math.floor(x - radius))
        right = int(math.floor(x + radius))
        top = int(math.floor(y - radius))
        bottom = int(math.floor(y + radius))
        for gy in range(top, bottom + 1):
            for gx in range(left, right + 1):
                if self.is_wall(gx, gy):
                    closest_x = max(gx, min(x, gx + 1))
                    closest_y = max(gy, min(y, gy + 1))
                    if (x - closest_x) ** 2 + (y - closest_y) ** 2 < radius ** 2:
                        return True
        return False

    def draw(self, screen: pygame.Surface, offset: Tuple[int, int] = (0, 0)) -> None:
        wall_color = (40, 40, 40)
        floor_color = (200, 200, 200)
        off_x, off_y = offset
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                color = wall_color if cell == 1 else floor_color
                pygame.draw.rect(
                    screen,
                    color,
                    pygame.Rect(off_x + x * CELL_SIZE, off_y + y * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                )


class Agent:
    def __init__(
        self,
        x: float,
        y: float,
        color: Tuple[int, int, int],
        max_speed: float = 0.2,
        accel_steps: int = 4,
    ):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(0, 0)
        self.dir = pygame.Vector2(0, 0)
        self.facing = pygame.Vector2(1, 0)
        self.color = color
        self.max_speed = max_speed
        self.accel = max_speed / float(accel_steps)
        self.radius = CELL_SIZE // 3
        self.speed_boost = 0.0


    def set_direction(self, dx: float, dy: float) -> None:
        v = pygame.Vector2(dx, dy)
        if v.length_squared() > 0:
            v_norm = v.normalize()
            if v_norm.dot(self.facing) > 0.99:
                self.speed_boost = min(self.speed_boost + CONTINUOUS_ACCEL, MAX_SPEED_BOOST)
            else:
                self.speed_boost = 0.0
            self.dir = v_norm
            self.facing = self.dir
        else:
            self.dir = pygame.Vector2(0, 0)
            self.speed_boost = 0.0

    def update(self, stage: StageMap) -> None:
        if self.dir.length_squared() > 0:
            accel = self.accel + self.speed_boost
            self.vel += self.dir * accel
            max_speed = self.max_speed + self.speed_boost
            if self.vel.length() > max_speed:
                self.vel.scale_to_length(max_speed)
        else:
            self.vel = pygame.Vector2(0, 0)
            self.speed_boost = 0.0
        radius = self.radius / CELL_SIZE

        new_x = self.pos.x + self.vel.x
        if stage.collides_circle(new_x, self.pos.y, radius):
            new_x = self.pos.x
            self.vel.x = 0
            self.speed_boost = 0.0

        new_y = self.pos.y + self.vel.y
        if stage.collides_circle(new_x, new_y, radius):
            new_y = self.pos.y
            self.vel.y = 0
            self.speed_boost = 0.0

        self.pos.update(new_x, new_y)

    def draw(self, screen: pygame.Surface, offset: Tuple[int, int] = (0, 0)) -> None:
        off_x, off_y = offset
        pygame.draw.circle(
            screen,
            self.color,
            (
                int(off_x + self.pos.x * CELL_SIZE + CELL_SIZE / 2),
                int(off_y + self.pos.y * CELL_SIZE + CELL_SIZE / 2),
            ),
            self.radius,
        )



    def collides_with(self, other: "Agent") -> bool:
        center_self = pygame.Vector2(
            self.pos.x * CELL_SIZE + CELL_SIZE / 2,
            self.pos.y * CELL_SIZE + CELL_SIZE / 2,
        )
        center_other = pygame.Vector2(
            other.pos.x * CELL_SIZE + CELL_SIZE / 2,
            other.pos.y * CELL_SIZE + CELL_SIZE / 2,
        )
        return center_self.distance_to(center_other) < self.radius + other.radius

    def observe(self, other: "Agent") -> List[float]:
        """Return relative position (dx, dy) of ``other``."""

        diff = other.pos - self.pos
        return [diff.x, diff.y]


def get_state(oni: Agent, nige: Agent) -> Tuple[List[float], List[float]]:
    """Return observation vectors for both agents."""

    return oni.observe(nige), nige.observe(oni)


def main():
    parser = argparse.ArgumentParser(description="2D鬼ごっこデモ")
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help="ゲームの制限時間（秒）",
    )
    args = parser.parse_args()

    pygame.init()
    width, height = 31, 21
    stage = StageMap(width, height, extra_wall_prob=EXTRA_WALL_PROB)

    screen = pygame.display.set_mode(
        (width * CELL_SIZE, height * CELL_SIZE + INFO_PANEL_HEIGHT)
    )
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    start = time.time()
    end_time = start + args.duration

    oni = Agent(1.5, 1.5, (255, 0, 0))
    nige = Agent(width - 2, height - 2, (0, 100, 255))

    running = True
    while running:
        remaining = max(0.0, end_time - time.time())
        if remaining <= 0:
            break
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        # arrow keys control escapee
        dx = keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]
        dy = keys[pygame.K_DOWN] - keys[pygame.K_UP]
        nige.set_direction(dx, dy)
        # wasd for oni
        odx = keys[pygame.K_d] - keys[pygame.K_a]
        ody = keys[pygame.K_s] - keys[pygame.K_w]
        oni.set_direction(odx, ody)
        oni.update(stage)
        nige.update(stage)

        screen.fill((0, 0, 0))
        pygame.draw.rect(
            screen,
            (255, 255, 255),
            pygame.Rect(0, 0, width * CELL_SIZE, INFO_PANEL_HEIGHT),
        )
        offset = (0, INFO_PANEL_HEIGHT)
        stage.draw(screen, offset)
        oni.draw(screen, offset)
        nige.draw(screen, offset)
        pygame.draw.line(
            screen,
            (255, 0, 0),
            (
                int(oni.pos.x * CELL_SIZE + CELL_SIZE / 2) + offset[0],
                int(oni.pos.y * CELL_SIZE + CELL_SIZE / 2) + offset[1],
            ),
            (
                int(nige.pos.x * CELL_SIZE + CELL_SIZE / 2) + offset[0],
                int(nige.pos.y * CELL_SIZE + CELL_SIZE / 2) + offset[1],
            ),
            2,
        )
        txt = font.render(f"残り{remaining:.1f}秒", True, (0, 0, 0))
        screen.blit(txt, (10, 5))
        if oni.collides_with(nige):
            font = pygame.font.SysFont(None, 48)
            text = font.render("Caught!", True, (255, 0, 0))
            rect = text.get_rect(
                center=(
                    width * CELL_SIZE // 2,
                    height * CELL_SIZE // 2 + INFO_PANEL_HEIGHT // 2,
                )
            )
            screen.blit(text, rect)
            pygame.display.flip()
            pygame.time.wait(1000)
            running = False
            continue
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
