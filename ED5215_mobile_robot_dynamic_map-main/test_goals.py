"""
test_goals.py
=============
Self-contained headless goal-collection test.
NO matplotlib imported at all - pure numpy + nav_planner only.

Run with:  python test_goals.py
"""

import sys
import os
import random
import numpy as np
from collections import deque

# Prevent matplotlib import inside grid_nav_viz from crashing
os.environ.setdefault('MPLBACKEND', 'Agg')

# ─── Inline all needed constants (mirrors grid_nav_viz.py exactly) ────────
GRID_ROWS   = 22
GRID_COLS   = 28
ROBOT_START = (1, 1)

GOALS = [
    (2,  25),   # 1
    (10, 13),   # 2
    (19,  2),   # 3
    (5,   7),   # 4
    (18, 24),   # 5
    (1,  20),   # 6
    (12,  4),   # 7
    (8,  22),   # 8
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

NUM_DYN_OBS    = 5
OBS_MOVE_EVERY = 12
OBS_SPREAD     = 1
COLLECT_DIST   = 1
REPLAN_ON_BLOCK = True
MAX_STEPS      = 6000    # hard safety cap per algorithm
SEED           = 99

# ─── Import only the planning functions ───────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from nav_planner import astar, dijkstra, bfs, DStarLite


# ─── Maze builder ─────────────────────────────────────────────────────────
def build_maze():
    maze = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int8)
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1
    for r1, r2, c1, c2 in INNER_WALLS:
        r1 = max(1, min(GRID_ROWS-2, r1)); r2 = max(1, min(GRID_ROWS-2, r2))
        c1 = max(1, min(GRID_COLS-2, c1)); c2 = max(1, min(GRID_COLS-2, c2))
        maze[r1:r2+1, c1:c2+1] = 1
    return maze


# ─── BFS reachability check ───────────────────────────────────────────────
def bfs_reachable(maze, start):
    vis = {start}; q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = (r+dr, c+dc)
            if 0<=nb[0]<GRID_ROWS and 0<=nb[1]<GRID_COLS and maze[nb[0],nb[1]]==0 and nb not in vis:
                vis.add(nb); q.append(nb)
    return vis


# ─── Minimal environment ─────────────────────────────────────────────────
class Env:
    def __init__(self, algo_fn, seed=SEED):
        random.seed(seed)
        self.base    = build_maze()
        self.robot   = list(ROBOT_START)
        self.goals   = [g for g in GOALS if self.base[g[0], g[1]] == 0]
        self.goal_idx = 0
        self.collected = set()
        self.done    = False
        self.planned = []
        self.frame_n = 0
        self.algo_fn = algo_fn          # callable(grid, start, goal) -> (path, _)

        # spawn dynamic obstacles
        safety = {tuple(self.robot)} | {tuple(g) for g in self.goals}
        self.dyn_obs, tries = [], 0
        while len(self.dyn_obs) < NUM_DYN_OBS and tries < 2000:
            r = random.randint(1, GRID_ROWS-2)
            c = random.randint(1, GRID_COLS-2)
            if self.base[r, c] == 0 and (r, c) not in safety:
                self.dyn_obs.append([r, c]); safety.add((r, c))
            tries += 1

        self._replan()

    # ── internal helpers ──────────────────────────────────────────────────
    def _inflate(self):
        cells = set()
        for o in self.dyn_obs:
            for dr in range(-OBS_SPREAD, OBS_SPREAD+1):
                for dc in range(-OBS_SPREAD, OBS_SPREAD+1):
                    nr, nc = o[0]+dr, o[1]+dc
                    if 0<=nr<GRID_ROWS and 0<=nc<GRID_COLS:
                        cells.add((nr, nc))
        return cells

    def _pgrid(self):
        g  = self.base.copy()
        gc = tuple(self.goals[self.goal_idx]) if self.goal_idx < len(self.goals) else None
        rc = tuple(self.robot)
        for r, c in self._inflate():
            if (r, c) not in (rc, gc):
                g[r, c] = 1
        return g

    def _move_obs(self):
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        occ = {(o[0], o[1]) for o in self.dyn_obs}
        rc  = tuple(self.robot)
        for obs in self.dyn_obs:
            random.shuffle(dirs)
            for dr, dc in dirs:
                nr, nc = obs[0]+dr, obs[1]+dc
                if (1<=nr<GRID_ROWS-1 and 1<=nc<GRID_COLS-1
                        and self.base[nr,nc]==0
                        and (nr,nc) not in occ and (nr,nc)!=rc):
                    occ.discard((obs[0],obs[1]))
                    obs[0], obs[1] = nr, nc
                    occ.add((nr,nc))
                    break

    def _advance(self):
        while self.goal_idx < len(self.goals) and self.goal_idx in self.collected:
            self.goal_idx += 1

    def _replan(self):
        self._advance()
        if self.goal_idx >= len(self.goals):
            uncol = [i for i in range(len(self.goals)) if i not in self.collected]
            if not uncol:
                self.planned = []; self.done = True
            else:
                self.goal_idx = uncol[0]; self._replan()
            return

        curr = tuple(self.goals[self.goal_idx])
        rob  = tuple(self.robot)
        if abs(rob[0]-curr[0]) + abs(rob[1]-curr[1]) <= COLLECT_DIST:
            self.collected.add(self.goal_idx)
            self.goal_idx += 1
            self._replan()
            return

        path, _ = self.algo_fn(self._pgrid(), rob, curr)
        self.planned = list(path[1:]) if len(path) > 1 else []

    # ── one simulation step ───────────────────────────────────────────────
    def step(self):
        if self.done: return
        self.frame_n += 1

        if self.frame_n % OBS_MOVE_EVERY == 0:
            self._move_obs()
            if REPLAN_ON_BLOCK and self.planned:
                if any(c in self._inflate() for c in self.planned[:4]):
                    self._replan()

        if not self.planned:
            self._replan()
        if not self.planned:
            return  # temporarily blocked; will retry next frame

        nxt = self.planned.pop(0)
        self.robot = list(nxt)

        # collection check
        rob = tuple(self.robot)
        hit = False
        for i, g in enumerate(self.goals):
            if i not in self.collected:
                if abs(rob[0]-g[0]) + abs(rob[1]-g[1]) <= COLLECT_DIST:
                    self.collected.add(i); hit = True

        if hit:
            self.planned = []
            self._replan()


# ─── D* Lite wrapper (separate instance per algo slot) ───────────────────
def make_dstar(key='dstar'):
    inst = DStarLite((GRID_ROWS, GRID_COLS))
    def _plan(grid, start, goal):
        return inst.plan(grid, start, goal)
    return _plan


# ─── Run all algorithms ───────────────────────────────────────────────────
ALGOS = [
    ('A*',        astar),
    ('Dijkstra',  dijkstra),
    ('BFS',       bfs),
    ('D*',        make_dstar('dstar')),
    ('D* Lite',   make_dstar('dstarlite')),
]

print("=" * 62)
print(f"  GOAL COLLECTION TEST | {GRID_ROWS}×{GRID_COLS} grid | {len(GOALS)} goals")
print(f"  Max steps = {MAX_STEPS} | Seed = {SEED}")
print("=" * 62)

all_pass = True
for name, fn in ALGOS:
    env = Env(fn)
    for _ in range(MAX_STEPS):
        env.step()
        if env.done:
            break

    n = len(env.collected)
    t = len(env.goals)
    ok = n == t
    if not ok:
        all_pass = False
        miss = [env.goals[i] for i in range(t) if i not in env.collected]
    else:
        miss = []

    tag   = "✓ PASS" if ok else "✗ FAIL"
    stuck = "(TIMED OUT)" if not env.done else ""
    print(f"  {tag}  {name:<12}  {n}/{t} goals  frame={env.frame_n}  {stuck}")
    if miss:
        print(f"         Missing: {miss}")

print("=" * 62)
if all_pass:
    print("  ✓ ALL ALGORITHMS REACH ALL 8 GOALS\n")
else:
    print("  ✗ Some algorithms did NOT finish — see above\n")

sys.exit(0 if all_pass else 1)
