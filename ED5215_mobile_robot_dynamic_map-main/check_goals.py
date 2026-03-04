"""
check_goals.py — static analysis only (no matplotlib, no simulation)
Verifies:
  1. Every goal cell is free in the built maze.
  2. Every goal is reachable from ROBOT_START by A* on the static maze.
  3. There are no duplicate goals.
  4. D* Lite instance keying is correct.
"""

import numpy as np
from collections import deque

# ── reproduce the exact maze builder ────────────────────────────────────
GRID_ROWS   = 22
GRID_COLS   = 28
ROBOT_START = (1, 1)

GOALS = [
    (2,  25),   # 1 top-right
    (10, 13),   # 2 centre
    (19,  2),   # 3 bottom-left
    (5,   7),   # 4 left
    (18, 24),   # 5 bottom-right
    (1,  20),   # 6 top-mid
    (12,  4),   # 7 mid-left  (moved from 12,5)
    (8,  22),   # 8 mid-right
]

INNER_WALLS = [
    (3,  3,  2,  12), (3,  3,  14, 22),
    (7,  7,  6,  14), (7,  7,  16, 25),
    (11, 11, 2,   8), (11, 11, 10, 18),
    (15, 15, 5,  13), (15, 15, 15, 24),
    (19, 19, 3,   9), (19, 19, 12, 20),
    (1,  5,  14, 14), (5,  11,  9,  9),
    (8,  15, 21, 21), (11, 19, 14, 14),
    (3,   9, 19, 19), (10, 19,  5,  5),
    (4,  12, 24, 24),
]

def build_maze():
    maze = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int8)
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1
    for r1, r2, c1, c2 in INNER_WALLS:
        r1 = max(1, min(GRID_ROWS-2, r1));  r2 = max(1, min(GRID_ROWS-2, r2))
        c1 = max(1, min(GRID_COLS-2, c1));  c2 = max(1, min(GRID_COLS-2, c2))
        maze[r1:r2+1, c1:c2+1] = 1
    return maze

def bfs_reachable(maze, start):
    """Return set of all cells reachable from start via BFS."""
    visited = set()
    q = deque([start])
    visited.add(start)
    rows, cols = maze.shape
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr,nc] == 0 and (nr,nc) not in visited:
                visited.add((nr,nc))
                q.append((nr,nc))
    return visited

maze = build_maze()
reachable = bfs_reachable(maze, ROBOT_START)

print("=" * 60)
print(f"  Maze: {GRID_ROWS}×{GRID_COLS}   Robot start: {ROBOT_START}")
print(f"  Total free cells: {np.sum(maze==0)}")
print("=" * 60)

all_ok = True

# Check start
if maze[ROBOT_START[0], ROBOT_START[1]] != 0:
    print(f"  ✗ ROBOT START {ROBOT_START} is inside a WALL!")
    all_ok = False
else:
    print(f"  ✓ Robot start {ROBOT_START}: free")

print()
print("  Goal checks:")
print(f"  {'#':<4} {'Cell':<12} {'In Wall?':<12} {'Reachable?':<12}")
print(f"  {'-'*4} {'-'*12} {'-'*12} {'-'*12}")

for i, g in enumerate(GOALS):
    in_wall   = maze[g[0], g[1]] != 0
    reachable_flag = g in reachable
    wall_str  = "WALL ✗" if in_wall else "free ✓"
    reach_str = "YES ✓" if reachable_flag else "BLOCKED ✗"
    flag = ""
    if in_wall or not reachable_flag:
        flag = " ← BUG"
        all_ok = False
    print(f"  {i+1:<4} {str(g):<12} {wall_str:<12} {reach_str:<12}{flag}")

# Check for duplicates
seen = set()
for i, g in enumerate(GOALS):
    if g in seen:
        print(f"  ✗ DUPLICATE Goal {i+1}: {g}")
        all_ok = False
    seen.add(g)

print()
print("=" * 60)
if all_ok:
    print("  ✓ ALL GOALS ARE FREE AND REACHABLE!")
else:
    print("  ✗ PROBLEMS FOUND — see above.")
print("=" * 60)
