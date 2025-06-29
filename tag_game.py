import math
import random
from typing import Tuple, List

import pygame

from stage_generator import generate_stage, Stage


CELL_SIZE = 20
FOV_DEG = 120
FOV_DIST = 5


class StageMap:
    def __init__(self, width: int, height: int):
        self.grid = generate_stage(width, height)
        self.width = width
        self.height = height

    def is_wall(self, x: int, y: int) -> bool:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True
        return self.grid[y][x] == 1

    def draw(self, screen: pygame.Surface) -> None:
        wall_color = (40, 40, 40)
        floor_color = (200, 200, 200)
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                color = wall_color if cell == 1 else floor_color
                pygame.draw.rect(
                    screen,
                    color,
                    pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE),
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


    def set_direction(self, dx: float, dy: float) -> None:
        v = pygame.Vector2(dx, dy)
        if v.length_squared() > 0:
            self.dir = v.normalize()
            self.facing = self.dir
        else:
            self.dir = pygame.Vector2(0, 0)

    def update(self, stage: StageMap) -> None:
        if self.dir.length_squared() > 0:
            self.vel += self.dir * self.accel
            if self.vel.length() > self.max_speed:
                self.vel.scale_to_length(self.max_speed)
        else:
            self.vel = pygame.Vector2(0, 0)

        new_pos = self.pos + self.vel
        # x axis collision
        next_x = int(new_pos.x)
        if stage.is_wall(next_x, int(self.pos.y)):
            new_pos.x = self.pos.x
            self.vel.x = 0
        # y axis collision
        next_y = int(new_pos.y)
        if stage.is_wall(int(new_pos.x), next_y):
            new_pos.y = self.pos.y
            self.vel.y = 0

        self.pos = new_pos

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.circle(
            screen,
            self.color,
            (int(self.pos.x * CELL_SIZE + CELL_SIZE / 2), int(self.pos.y * CELL_SIZE + CELL_SIZE / 2)),
            self.radius,
        )
        self.draw_fov(screen)

    def draw_fov(self, screen: pygame.Surface) -> None:
        start_angle = math.atan2(self.facing.y, self.facing.x) - math.radians(FOV_DEG) / 2
        end_angle = start_angle + math.radians(FOV_DEG)
        center = (
            int(self.pos.x * CELL_SIZE + CELL_SIZE / 2),
            int(self.pos.y * CELL_SIZE + CELL_SIZE / 2),
        )
        pygame.draw.arc(
            screen,
            (150, 150, 255),
            pygame.Rect(
                center[0] - FOV_DIST * CELL_SIZE,
                center[1] - FOV_DIST * CELL_SIZE,
                2 * FOV_DIST * CELL_SIZE,
                2 * FOV_DIST * CELL_SIZE,
            ),
            start_angle,
            end_angle,
            1,
        )

    def can_see(self, other: "Agent") -> bool:
        diff = other.pos - self.pos
        if diff.length() > FOV_DIST:
            return False
        if diff.length_squared() == 0:
            return True
        ang = math.degrees(math.acos(self.facing.normalize().dot(diff.normalize())))
        return ang <= FOV_DEG / 2

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


def main():
    pygame.init()
    width, height = 31, 21
    stage = StageMap(width, height)

    screen = pygame.display.set_mode((width * CELL_SIZE, height * CELL_SIZE))
    clock = pygame.time.Clock()

    oni = Agent(1.5, 1.5, (255, 0, 0))
    nige = Agent(width - 2, height - 2, (0, 100, 255))

    running = True
    while running:
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
        stage.draw(screen)
        oni.draw(screen)
        nige.draw(screen)
        if oni.can_see(nige):
            pygame.draw.line(
                screen,
                (255, 0, 0),
                (
                    int(oni.pos.x * CELL_SIZE + CELL_SIZE / 2),
                    int(oni.pos.y * CELL_SIZE + CELL_SIZE / 2),
                ),
                (
                    int(nige.pos.x * CELL_SIZE + CELL_SIZE / 2),
                    int(nige.pos.y * CELL_SIZE + CELL_SIZE / 2),
                ),
                2,
            )
        if oni.collides_with(nige):
            font = pygame.font.SysFont(None, 48)
            text = font.render("Caught!", True, (255, 0, 0))
            rect = text.get_rect(center=(width * CELL_SIZE // 2, height * CELL_SIZE // 2))
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
