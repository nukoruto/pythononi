import math
from typing import Tuple, List

import argparse
import time

import pygame

from stage_generator import generate_stage, Stage
import numpy as np
import torch


class Actor(torch.nn.Module):
    """Simple actor network compatible with SAC models."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.mean = torch.nn.Linear(hidden_dim, action_dim)
        self.log_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mean(h), self.log_std(h)

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return y_t, log_prob

    def act(self, obs: np.ndarray) -> np.ndarray:
        device = next(self.parameters()).device
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action, _ = self.sample(obs_t.unsqueeze(0))
        return action.squeeze(0).cpu().numpy()


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
        width_range: Tuple[int, int] | None = None,
        height_range: Tuple[int, int] | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.rng = rng or np.random.default_rng()

        if width_range is not None:
            w_min, w_max = width_range
            if w_min % 2 == 0:
                w_min += 1
            if w_max % 2 == 0:
                w_max -= 1
            if w_min > w_max:
                raise ValueError("width_range must include at least one odd number")
            num = (w_max - w_min) // 2 + 1
            width = int(w_min + 2 * int(self.rng.integers(0, num)))

        if height_range is not None:
            h_min, h_max = height_range
            if h_min % 2 == 0:
                h_min += 1
            if h_max % 2 == 0:
                h_max -= 1
            if h_min > h_max:
                raise ValueError("height_range must include at least one odd number")
            num = (h_max - h_min) // 2 + 1
            height = int(h_min + 2 * int(self.rng.integers(0, num)))

        self.grid = generate_stage(width, height, extra_wall_prob=extra_wall_prob, rng=self.rng)
        self.width = width
        self.height = height

    def is_wall(self, x: int, y: int) -> bool:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True
        return self.grid[y][x] == 1

    def random_open_position(self) -> pygame.Vector2:
        """Return a random free cell center as ``pygame.Vector2``."""
        cells: list[tuple[int, int]] = []
        for y in range(self.height):
            for x in range(self.width):
                if not self.is_wall(x, y):
                    cells.append((x, y))
        if not cells:
            raise ValueError("stage has no open cells")
        cx, cy = cells[int(self.rng.integers(0, len(cells)))]
        return pygame.Vector2(cx + 0.5, cy + 0.5)

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

    def shortest_path(
        self, start: pygame.Vector2, goal: pygame.Vector2
    ) -> list[tuple[int, int]]:
        """Return cell coordinates from ``start`` to ``goal`` along the shortest
        path.

        Parameters
        ----------
        start : pygame.Vector2
            開始位置（セル座標）
        goal : pygame.Vector2
            目的地（セル座標）

        Returns
        -------
        list[tuple[int, int]]
            ``start`` から ``goal`` までのセル座標列。経路が存在しない
            場合は空リストを返す。
        """

        start_cell = (int(start.x), int(start.y))
        goal_cell = (int(goal.x), int(goal.y))
        if start_cell == goal_cell:
            return [start_cell]

        from collections import deque

        queue: deque[tuple[int, int]] = deque([start_cell])
        parents: dict[tuple[int, int], tuple[int, int] | None] = {start_cell: None}

        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) == goal_cell:
                break
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = cx + dx, cy + dy
                if (
                    0 <= nx < self.width
                    and 0 <= ny < self.height
                    and not self.is_wall(nx, ny)
                    and (nx, ny) not in parents
                ):
                    parents[(nx, ny)] = (cx, cy)
                    queue.append((nx, ny))

        if goal_cell not in parents:
            return []

        # reconstruct path
        path = [goal_cell]
        cur = goal_cell
        while parents[cur] is not None:
            cur = parents[cur]
            path.append(cur)
        path.reverse()
        return path

    def shortest_path_direction(
        self, start: pygame.Vector2, goal: pygame.Vector2
    ) -> pygame.Vector2:
        """Return normalized direction of the first step on the shortest path.

        ``shortest_path`` を利用して ``start`` から ``goal`` までの経路を
        求め、その最初の1歩分の方向ベクトルを返す。経路が存在しない
        場合は長さ0のベクトルとなる。
        """

        path = self.shortest_path(start, goal)
        if len(path) < 2:
            return pygame.Vector2(0, 0)

        dx = path[1][0] - path[0][0]
        dy = path[1][1] - path[0][1]
        direction = pygame.Vector2(dx, dy)
        if direction.length_squared() > 0:
            direction = direction.normalize()
        return direction

    def shortest_path_info(
        self, start: pygame.Vector2, goal: pygame.Vector2
    ) -> tuple[pygame.Vector2, int]:
        """Return direction and path length from ``start`` to ``goal``.

        経路が存在しない場合は方向ベクトル(0,0)と距離0を返す。"""

        path = self.shortest_path(start, goal)
        if not path:
            return pygame.Vector2(0, 0), 0

        if len(path) < 2:
            direction = pygame.Vector2(0, 0)
        else:
            dx = path[1][0] - path[0][0]
            dy = path[1][1] - path[0][1]
            direction = pygame.Vector2(dx, dy)
            if direction.length_squared() > 0:
                direction = direction.normalize()
        distance = len(path) - 1
        return direction, distance

    def shortest_path_vectors(
        self, start: pygame.Vector2, goal: pygame.Vector2
    ) -> list[pygame.Vector2]:
        """Return step vectors along the shortest path.

        ``shortest_path`` で得られるセル座標列から、各ステップの方向ベクトル
        (長さ1) を作成し、順番に並べたリストを返す。経路が存在しない場合は
        空リストを返す。
        """

        path = self.shortest_path(start, goal)
        if len(path) < 2:
            return []

        vectors: list[pygame.Vector2] = []
        for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
            vec = pygame.Vector2(x1 - x0, y1 - y0)
            if vec.length_squared() > 0:
                vec = vec.normalize()
            vectors.append(vec)
        return vectors

    def draw_shortest_path_vectors(
        self,
        screen: pygame.Surface,
        start: pygame.Vector2,
        goal: pygame.Vector2,
        *,
        color: Tuple[int, int, int] = (0, 255, 0),
        offset: Tuple[int, int] = (0, 0),
    ) -> None:
        """Draw the shortest path from ``start`` to ``goal``.

        Parameters
        ----------
        screen : pygame.Surface
            Surface to draw on.
        start : pygame.Vector2
            Starting position (world coordinates).
        goal : pygame.Vector2
            Goal position (world coordinates).
        color : Tuple[int, int, int], optional
            Line color, by default green.
        offset : Tuple[int, int], optional
            Pixel offset for drawing, by default ``(0, 0)``.
        """

        vectors = self.shortest_path_vectors(start, goal)
        if not vectors:
            return

        off_x, off_y = offset
        pos = pygame.Vector2(int(start.x) + 0.5, int(start.y) + 0.5)
        prev_px = (
            off_x + pos.x * CELL_SIZE,
            off_y + pos.y * CELL_SIZE,
        )
        for vec in vectors:
            pos += vec
            next_px = (
                off_x + pos.x * CELL_SIZE,
                off_y + pos.y * CELL_SIZE,
            )
            pygame.draw.line(screen, color, prev_px, next_px, 2)
            prev_px = next_px

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

    def update(self, stage: StageMap) -> bool:
        """Update agent position and return True if it collided with a wall."""
        collided = False
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
        new_y = self.pos.y + self.vel.y

        if stage.collides_circle(new_x, new_y, radius):
            collided = True
            # 角にぶつかった場合のスライド処理
            if not stage.collides_circle(new_x, self.pos.y, radius):
                # X 方向のみ移動可能
                self.pos.x = new_x
                self.vel.y = 0
                self.speed_boost = 0.0
            elif not stage.collides_circle(self.pos.x, new_y, radius):
                # Y 方向のみ移動可能
                self.pos.y = new_y
                self.vel.x = 0
                self.speed_boost = 0.0
            else:
                # 両方向とも衝突
                self.vel.x = 0
                self.vel.y = 0
                self.speed_boost = 0.0
        else:
            self.pos.update(new_x, new_y)
        return collided
    def draw(self, screen: pygame.Surface, offset: Tuple[int, int] = (0, 0)) -> None:
        off_x, off_y = offset
        pygame.draw.circle(
            screen,
            self.color,
            (
                int(off_x + self.pos.x * CELL_SIZE),
                int(off_y + self.pos.y * CELL_SIZE),
            ),
            self.radius,
        )



    def collides_with(self, other: "Agent") -> bool:
        center_self = pygame.Vector2(
            self.pos.x * CELL_SIZE,
            self.pos.y * CELL_SIZE,
        )
        center_other = pygame.Vector2(
            other.pos.x * CELL_SIZE,
            other.pos.y * CELL_SIZE,
        )
        return center_self.distance_to(center_other) < self.radius + other.radius

    def observe(
        self, other: "Agent", stage: StageMap | None = None
    ) -> List[float]:
        """Return vector toward ``other``.

        壁を考慮した最短経路方向を求める場合は ``stage`` を渡す。
        ``stage`` が ``None`` の場合は従来の相対座標を返す。
        """

        if stage is None:
            diff = other.pos - self.pos
            distance = diff.length()
            return [diff.x, diff.y, distance]

        direction, length = stage.shortest_path_info(self.pos, other.pos)
        max_dist = stage.width + stage.height
        dist_norm = min(length / max_dist, 1.0)
        return [direction.x, direction.y, dist_norm]


def get_state(
    oni: Agent, nige: Agent, stage: StageMap | None = None
) -> Tuple[List[float], List[float]]:
    """Return observation vectors for both agents."""

    if stage is None:
        return oni.observe(nige), nige.observe(oni)
    return oni.observe(nige, stage), nige.observe(oni, stage)


def main():
    parser = argparse.ArgumentParser(description="2D鬼ごっこデモ")
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help="ゲームの制限時間（秒）",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1,
        help="連続対戦数",
    )
    parser.add_argument(
        "--width-range",
        type=str,
        default=None,
        help="ステージ幅の最小値と最大値をカンマ区切りで指定",
    )
    parser.add_argument(
        "--height-range",
        type=str,
        default=None,
        help="ステージ高さの最小値と最大値をカンマ区切りで指定",
    )
    parser.add_argument(
        "--oni",
        type=str,
        default=None,
        help="鬼側モデルのパス（指定すると逃げはプレイヤー操作）",
    )
    parser.add_argument(
        "--nige",
        type=str,
        default=None,
        help="逃げ側モデルのパス（指定すると鬼はプレイヤー操作）",
    )
    args = parser.parse_args()

    if (args.oni is None) == (args.nige is None):
        raise SystemExit("--oni または --nige のどちらか一方を指定してください")

    pygame.init()
    width, height = 73, 51
    width_range = None
    height_range = None
    if args.width_range:
        try:
            w_min, w_max = map(int, args.width_range.split(","))
            width_range = (w_min, w_max)
        except Exception as e:
            raise SystemExit(f"invalid --width-range: {e}")
    if args.height_range:
        try:
            h_min, h_max = map(int, args.height_range.split(","))
            height_range = (h_min, h_max)
        except Exception as e:
            raise SystemExit(f"invalid --height-range: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oni_actor = None
    nige_actor = None
    if args.oni is not None:
        oni_actor = Actor(3, 2).to(device)
        state = torch.load(args.oni, map_location=device)
        oni_actor.load_state_dict(state["actor"] if "actor" in state else state)
        oni_actor.eval()
    if args.nige is not None:
        nige_actor = Actor(3, 2).to(device)
        state = torch.load(args.nige, map_location=device)
        nige_actor.load_state_dict(state["actor"] if "actor" in state else state)
        nige_actor.eval()

    for game in range(args.games):
        stage = StageMap(
            width,
            height,
            extra_wall_prob=EXTRA_WALL_PROB,
            width_range=width_range,
            height_range=height_range,
        )
        width, height = stage.width, stage.height

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
        result = "timeout"
        while running:
            remaining = max(0.0, end_time - time.time())
            if remaining <= 0:
                result = "timeout"
                break
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            keys = pygame.key.get_pressed()
            pdx = keys[pygame.K_d] - keys[pygame.K_a]
            pdy = keys[pygame.K_s] - keys[pygame.K_w]

            oni_obs, nige_obs = get_state(oni, nige, stage)
            if oni_actor is not None:
                action = oni_actor.act(np.array(oni_obs, dtype=np.float32))
                oni.set_direction(float(action[0]), float(action[1]))
                nige.set_direction(pdx, pdy)
            else:
                oni.set_direction(pdx, pdy)
                if nige_actor is not None:
                    action = nige_actor.act(np.array(nige_obs, dtype=np.float32))
                    nige.set_direction(float(action[0]), float(action[1]))

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
            stage.draw_shortest_path_vectors(
                screen, oni.pos, nige.pos, offset=offset
            )
            info = f"残り{remaining:.1f}秒  ({game + 1}/{args.games})"
            txt = font.render(info, True, (0, 0, 0))
            screen.blit(txt, (10, 5))
            if oni.collides_with(nige):
                result = "caught"
                font_big = pygame.font.SysFont(None, 48)
                text = font_big.render("Caught!", True, (255, 0, 0))
                rect = text.get_rect(
                    center=(
                        width * CELL_SIZE // 2,
                        height * CELL_SIZE // 2 + INFO_PANEL_HEIGHT // 2,
                    )
                )
                screen.blit(text, rect)
                pygame.display.flip()
                pygame.time.wait(1000)
                break
            pygame.display.flip()
            clock.tick(60)

        print(f"Game {game + 1}/{args.games}: {result}")

    pygame.quit()


if __name__ == "__main__":
    main()
