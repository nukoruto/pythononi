"""Stage generator for tag game.
"""
from typing import List, Tuple, Iterable
import numpy as np

Cell = int
Stage = List[List[Cell]]  # 0: path, 1: wall

def generate_maze(width: int, height: int, rng: np.random.Generator) -> Stage:
    """Generate a perfect maze using depth-first search."""
    # cells at odd indices are open spaces; even indices are walls
    grid = [[1 for _ in range(width)] for _ in range(height)]

    def in_bounds(x: int, y: int) -> bool:
        return 0 <= x < width and 0 <= y < height

    dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    stack = [(1, 1)]
    grid[1][1] = 0
    while stack:
        x, y = stack[-1]
        ordered_dirs = [dirs[i] for i in rng.permutation(len(dirs))]
        carved = False
        for dx, dy in ordered_dirs:
            nx, ny = x + dx, y + dy
            if in_bounds(nx, ny) and grid[ny][nx] == 1:
                wx, wy = x + dx // 2, y + dy // 2
                grid[wy][wx] = 0
                grid[ny][nx] = 0
                stack.append((nx, ny))
                carved = True
                break
        if not carved:
            stack.pop()
    return grid

def remove_dead_ends(stage: Stage, rng: np.random.Generator) -> Stage:
    """Remove dead ends by opening additional walls until every path cell has at
    least two open neighbors."""
    h, w = len(stage), len(stage[0])
    changed = True
    while changed:
        changed = False
        for y in range(1, h-1):
            for x in range(1, w-1):
                if stage[y][x] == 0:
                    neighbors = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
                    open_neighbors = [n for n in neighbors if stage[n[1]][n[0]] == 0]
                    if len(open_neighbors) <= 1:
                        walls = [n for n in neighbors if stage[n[1]][n[0]] == 1]
                        if walls:
                            nx, ny = walls[int(rng.integers(0, len(walls)))]
                            stage[ny][nx] = 0
                            changed = True
    return stage

def widen_paths(stage: Stage, width_range: Tuple[int, int], rng: np.random.Generator) -> Stage:
    """Randomly widen paths to create varying corridor widths."""
    h, w = len(stage), len(stage[0])
    min_w, max_w = width_range
    for y in range(1, h-1):
        for x in range(1, w-1):
            if stage[y][x] == 0 and rng.random() < 0.2:
                widen = int(rng.integers(min_w, max_w + 1))
                for dy in range(-widen, widen+1):
                    for dx in range(-widen, widen+1):
                        nx, ny = x + dx, y + dy
                        if 0 < nx < w-1 and 0 < ny < h-1:
                            stage[ny][nx] = 0
    return stage

def _neighbors(x: int, y: int) -> Iterable[Tuple[int, int]]:
    """Yield 4-neighborhood coordinates."""
    yield x + 1, y
    yield x - 1, y
    yield x, y + 1
    yield x, y - 1

def is_stage_connected(stage: Stage) -> bool:
    """Return True if all open cells are mutually reachable."""
    h, w = len(stage), len(stage[0])
    start = (1, 1)
    if stage[start[1]][start[0]] == 1:
        return False
    stack = [start]
    visited = {start}
    while stack:
        x, y = stack.pop()
        for nx, ny in _neighbors(x, y):
            if 0 <= nx < w and 0 <= ny < h and stage[ny][nx] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                stack.append((nx, ny))
    open_cells = sum(row.count(0) for row in stage)
    return len(visited) == open_cells

def add_random_walls(stage: Stage, prob: float, rng: np.random.Generator) -> Stage:
    """Randomly convert path cells back into walls while keeping connectivity."""
    if prob <= 0:
        return stage
    h, w = len(stage), len(stage[0])
    cells = [(x, y) for y in range(1, h - 1) for x in range(1, w - 1) if stage[y][x] == 0]
    for idx in rng.permutation(len(cells)):
        x, y = cells[int(idx)]
        if rng.random() < prob:
            stage[y][x] = 1
            if not is_stage_connected(stage):
                stage[y][x] = 0
    # ensure start and goal remain open
    stage[1][1] = 0
    stage[h - 2][w - 2] = 0
    return stage

def generate_stage(
    width: int,
    height: int,
    path_width: Tuple[int, int] = (1, 2),
    extra_wall_prob: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Stage:
    if rng is None:
        rng = np.random.default_rng()
    if width % 2 == 0 or height % 2 == 0:
        raise ValueError("width and height must be odd to generate maze")
    stage = generate_maze(width, height, rng)
    stage = remove_dead_ends(stage, rng)
    stage = widen_paths(stage, path_width, rng)
    stage = add_random_walls(stage, extra_wall_prob, rng)
    stage = remove_dead_ends(stage, rng)
    return stage

def print_stage(stage: Stage) -> None:
    chars = {0: ' ', 1: '#'}
    for row in stage:
        print(''.join(chars[c] for c in row))

if __name__ == "__main__":
    s = generate_stage(31, 21)
    print_stage(s)
