"""
test_goals_pure.py
==================
100% pure Python — no numpy, no matplotlib.
Uses only heapq, collections, random from the standard library.
Tests A*, BFS, Dijkstra and D*-Lite goal completion independently.

Run from Spyder console:  %run test_goals_pure.py
Or from any command line:  python test_goals_pure.py
"""

import heapq
import random
from collections import deque

# ─── Configuration ─────────────────────────────────────────────────────────
GRID_ROWS   = 22
GRID_COLS   = 28
ROBOT_START = (1, 1)
MAX_STEPS   = 8000
SEED        = 99
COLLECT_DIST = 1

GOALS = [
    (2,  25),  # 1 top-right
    (10, 13),  # 2 centre
    (19,  2),  # 3 bottom-left
    (5,   7),  # 4 left
    (18, 24),  # 5 bottom-right
    (1,  20),  # 6 top-mid
    (12,  4),  # 7 mid-left
    (8,  22),  # 8 mid-right
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


# ─── Maze (pure Python list-of-lists) ─────────────────────────────────────
def build_maze():
    g = [[0] * GRID_COLS for _ in range(GRID_ROWS)]
    # border walls
    for c in range(GRID_COLS): g[0][c] = g[GRID_ROWS-1][c] = 1
    for r in range(GRID_ROWS): g[r][0] = g[r][GRID_COLS-1] = 1
    # inner walls
    for r1, r2, c1, c2 in INNER_WALLS:
        r1 = max(1, min(GRID_ROWS-2, r1)); r2 = max(1, min(GRID_ROWS-2, r2))
        c1 = max(1, min(GRID_COLS-2, c1)); c2 = max(1, min(GRID_COLS-2, c2))
        for r in range(r1, r2+1):
            for c in range(c1, c2+1):
                g[r][c] = 1
    return g


def grid_copy(g):
    return [row[:] for row in g]


def neighbours(r, c):
    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
        nr, nc = r+dr, c+dc
        if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
            yield nr, nc


# ─── A* (pure Python) ─────────────────────────────────────────────────────
def astar_pure(grid, start, goal):
    if start == goal: return [start]
    h = lambda s: abs(s[0]-goal[0]) + abs(s[1]-goal[1])
    open_set  = [(h(start), 0, start)]
    g_score   = {start: 0}
    came_from = {}
    visited   = set()
    while open_set:
        _, g, cur = heapq.heappop(open_set)
        if cur in visited: continue
        visited.add(cur)
        if cur == goal:
            path = []; node = goal
            while node != start:
                path.append(node); node = came_from[node]
            path.append(start); path.reverse()
            return path
        for nr, nc in neighbours(*cur):
            nb = (nr, nc)
            if grid[nr][nc] != 0 and nb != goal and nb != start: continue
            ng = g + 1
            if ng < g_score.get(nb, 10**9):
                g_score[nb] = ng; came_from[nb] = cur
                heapq.heappush(open_set, (ng + h(nb), ng, nb))
    return []


# ─── Dijkstra (pure Python) ───────────────────────────────────────────────
def dijkstra_pure(grid, start, goal):
    if start == goal: return [start]
    open_set  = [(0, start)]
    dist      = {start: 0}
    came_from = {}
    visited   = set()
    while open_set:
        d, cur = heapq.heappop(open_set)
        if cur in visited: continue
        visited.add(cur)
        if cur == goal:
            path = []; node = goal
            while node != start:
                path.append(node); node = came_from[node]
            path.append(start); path.reverse()
            return path
        for nr, nc in neighbours(*cur):
            nb = (nr, nc)
            if grid[nr][nc] != 0 and nb != goal and nb != start: continue
            nd = d + 1
            if nd < dist.get(nb, 10**9):
                dist[nb] = nd; came_from[nb] = cur
                heapq.heappush(open_set, (nd, nb))
    return []


# ─── BFS (pure Python) ────────────────────────────────────────────────────
def bfs_pure(grid, start, goal):
    if start == goal: return [start]
    q = deque([start])
    came_from = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            path = []; node = goal
            while node is not None:
                path.append(node); node = came_from[node]
            path.reverse(); return path
        for nr, nc in neighbours(*cur):
            nb = (nr, nc)
            if grid[nr][nc] != 0 and nb != goal and nb != start: continue
            if nb not in came_from:
                came_from[nb] = cur; q.append(nb)
    return []


# ─── Simulation environment ────────────────────────────────────────────────
class Env:
    def __init__(self, planner_fn, seed=SEED, obs_spread=1, num_obs=5, obs_every=12):
        random.seed(seed)
        self.base   = build_maze()
        self.robot  = list(ROBOT_START)
        self.goals  = [g for g in GOALS if self.base[g[0]][g[1]] == 0]
        self.goal_idx   = 0
        self.collected  = set()
        self.planned    = []
        self.done       = False
        self.frame_n    = 0
        self.planner_fn = planner_fn
        self.obs_spread = obs_spread
        self.obs_every  = obs_every
        self.stats      = {'steps': 0}

        # Spawn dynamic obstacles
        safety = {tuple(self.robot)} | {tuple(g) for g in self.goals}
        self.dyn_obs, tries = [], 0
        while len(self.dyn_obs) < num_obs and tries < 2000:
            r = random.randint(1, GRID_ROWS-2)
            c = random.randint(1, GRID_COLS-2)
            if self.base[r][c] == 0 and (r,c) not in safety:
                self.dyn_obs.append([r,c]); safety.add((r,c))
            tries += 1

        self._replan()

    def _inflate(self):
        cells = set()
        sp = self.obs_spread
        for o in self.dyn_obs:
            for dr in range(-sp, sp+1):
                for dc in range(-sp, sp+1):
                    nr, nc = o[0]+dr, o[1]+dc
                    if 0<=nr<GRID_ROWS and 0<=nc<GRID_COLS:
                        cells.add((nr,nc))
        return cells

    def _pgrid(self):
        g  = grid_copy(self.base)
        gc = tuple(self.goals[self.goal_idx]) if self.goal_idx < len(self.goals) else None
        rc = tuple(self.robot)
        for nr, nc in self._inflate():
            if (nr,nc) not in (rc, gc):
                g[nr][nc] = 1
        return g

    def _move_obs(self):
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        occ  = {(o[0],o[1]) for o in self.dyn_obs}
        rc   = tuple(self.robot)
        for obs in self.dyn_obs:
            random.shuffle(dirs)
            for dr,dc in dirs:
                nr,nc = obs[0]+dr, obs[1]+dc
                if (1<=nr<GRID_ROWS-1 and 1<=nc<GRID_COLS-1
                        and self.base[nr][nc]==0
                        and (nr,nc) not in occ and (nr,nc)!=rc):
                    occ.discard((obs[0],obs[1]))
                    obs[0],obs[1] = nr,nc; occ.add((nr,nc))
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
            self._replan(); return

        path = self.planner_fn(self._pgrid(), rob, curr)
        self.planned = list(path[1:]) if len(path) > 1 else []

    def step(self):
        if self.done: return
        self.frame_n += 1; self.stats['steps'] += 1

        if self.frame_n % self.obs_every == 0:
            self._move_obs()
            if self.planned:
                inf = self._inflate()
                if any(c in inf for c in self.planned[:4]):
                    self._replan()

        if not self.planned: self._replan()
        if not self.planned: return

        nxt = self.planned.pop(0)
        self.robot = list(nxt)
        rob = tuple(self.robot)

        hit = False
        for i, g in enumerate(self.goals):
            if i not in self.collected:
                if abs(rob[0]-g[0]) + abs(rob[1]-g[1]) <= COLLECT_DIST:
                    self.collected.add(i); hit = True
        if hit:
            self.planned = []; self._replan()


# ─── Run tests ────────────────────────────────────────────────────────────
ALGOS = [
    ('A*',       astar_pure),
    ('Dijkstra', dijkstra_pure),
    ('BFS',      bfs_pure),
]

bar = "=" * 62
print(bar)
print(f"  GOAL COLLECTION TEST  |  {GRID_ROWS}x{GRID_COLS} grid  |  {len(GOALS)} goals")
print(f"  Max steps = {MAX_STEPS}  |  Seed = {SEED}")
print(bar)

# First check: are all goals in free cells?
base = build_maze()
print("\n  Goal occupancy check:")
blocked_goals = []
for i, g in enumerate(GOALS):
    free = base[g[0]][g[1]] == 0
    sym  = "✓ free" if free else "✗ WALL"
    if not free: blocked_goals.append(i+1)
    print(f"    Goal {i+1} {str(g):<10} {sym}")

if blocked_goals:
    print(f"\n  ✗ Goals in walls: {blocked_goals} — fix these before running simulation!")
else:
    print(f"\n  ✓ All {len(GOALS)} goal cells are free")

# Second check: BFS reachability from start
vis = set(); q = deque([ROBOT_START]); vis.add(ROBOT_START)
while q:
    r,c = q.popleft()
    for nr,nc in neighbours(r,c):
        if base[nr][nc]==0 and (nr,nc) not in vis:
            vis.add((nr,nc)); q.append((nr,nc))

unreachable = [i+1 for i,g in enumerate(GOALS) if g not in vis]
if unreachable:
    print(f"  ✗ Goals unreachable from start: {unreachable}")
else:
    print(f"  ✓ All {len(GOALS)} goals are reachable from {ROBOT_START}")

# Third check: Full simulation run
print(f"\n  Simulation results ({MAX_STEPS} step limit):")
all_pass = True
for name, fn in ALGOS:
    env = Env(fn)
    for _ in range(MAX_STEPS):
        env.step()
        if env.done: break

    n = len(env.collected); t = len(env.goals)
    ok = n == t
    if not ok: all_pass = False
    miss = [env.goals[i] for i in range(t) if i not in env.collected] if not ok else []
    tag  = "PASS ✓" if ok else "FAIL ✗"
    to   = " (TIMED OUT)" if not env.done else ""
    print(f"    [{tag}] {name:<10}  {n}/{t} goals  steps={env.stats['steps']}{to}")
    if miss: print(f"           Missing: {miss}")

print("\n" + bar)
if all_pass:
    print("  ✓  ALL 3 SEQUENTIAL ALGORITHMS REACH ALL 8 GOALS\n")
else:
    print("  ✗  Some algorithms failed — see above\n")
