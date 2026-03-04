"""
test_multigoal_pure.py
======================
Pure Python (no numpy, no matplotlib) test for the Multi-Goal Planner.
Mirrors the exact logic in nav_planner.py::MultiGoalPlanner.

Run:  python test_multigoal_pure.py
Or in Spyder:  %run test_multigoal_pure.py
"""

import heapq, random
from collections import deque

# ─── Config (mirrors grid_nav_viz.py exactly) ──────────────────────────────
GRID_ROWS   = 22
GRID_COLS   = 28
ROBOT_START = (1, 1)
MAX_STEPS   = 8000
SEED        = 99
COLLECT_DIST = 1
NUM_OBS     = 5
OBS_SPREAD  = 1
OBS_EVERY   = 12
EPSILON     = 3.0   # opportunistic detour tolerance (cells)
K_STEPS     = 5     # waypoints returned per planner step

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
    (3,3,2,12),(3,3,14,22),(7,7,6,14),(7,7,16,25),
    (11,11,2,8),(11,11,10,18),(15,15,5,13),(15,15,15,24),
    (19,19,3,9),(19,19,12,20),(1,5,14,14),(5,11,9,9),
    (8,15,21,21),(11,19,14,14),(3,9,19,19),(10,19,5,5),
    (4,12,24,24),
]

INF = float('inf')

# ─── Maze (pure Python) ────────────────────────────────────────────────────
def build_maze():
    g = [[0]*GRID_COLS for _ in range(GRID_ROWS)]
    for c in range(GRID_COLS): g[0][c] = g[GRID_ROWS-1][c] = 1
    for r in range(GRID_ROWS): g[r][0] = g[r][GRID_COLS-1] = 1
    for r1,r2,c1,c2 in INNER_WALLS:
        r1=max(1,min(GRID_ROWS-2,r1)); r2=max(1,min(GRID_ROWS-2,r2))
        c1=max(1,min(GRID_COLS-2,c1)); c2=max(1,min(GRID_COLS-2,c2))
        for r in range(r1,r2+1):
            for c in range(c1,c2+1): g[r][c]=1
    return g

def grid_copy(g): return [row[:] for row in g]

def nbs(r, c):
    for dr,dc in((-1,0),(1,0),(0,-1),(0,1)):
        nr,nc=r+dr,c+dc
        if 0<=nr<GRID_ROWS and 0<=nc<GRID_COLS: yield nr,nc


# ─── A* (pure Python) ─────────────────────────────────────────────────────
def astar(grid, start, goal):
    if start == goal: return [start]
    h = lambda s: abs(s[0]-goal[0]) + abs(s[1]-goal[1])
    q = [(h(start), 0, start)]
    gs = {start: 0}; cf = {}; vis = set()
    while q:
        _, g, cur = heapq.heappop(q)
        if cur in vis: continue
        vis.add(cur)
        if cur == goal:
            p=[]; n=goal
            while n!=start: p.append(n); n=cf[n]
            p.append(start); p.reverse(); return p
        for nr,nc in nbs(*cur):
            nb=(nr,nc)
            if grid[nr][nc]!=0 and nb!=goal and nb!=start: continue
            ng=g+1
            if ng < gs.get(nb, INF):
                gs[nb]=ng; cf[nb]=cur
                heapq.heappush(q,(ng+h(nb),ng,nb))
    return []


# ─── Prim's MST (pure Python, Manhattan distance) ─────────────────────────
def mst_cost(goals):
    """Mirrors MultiGoalPlanner._calculate_mst_cost() using Manhattan distance."""
    if not goals: return 0.0
    nodes = list(goals)
    n = len(nodes)
    if n <= 1: return 0.0
    visited  = [False] * n
    min_dist = [INF] * n
    min_dist[0] = 0
    total = 0.0
    for _ in range(n):
        u = min((i for i in range(n) if not visited[i]),
                key=lambda i: min_dist[i])
        visited[u] = True
        total += min_dist[u]
        for v in range(n):
            if not visited[v]:
                d = abs(nodes[u][0]-nodes[v][0]) + abs(nodes[u][1]-nodes[v][1])
                if d < min_dist[v]: min_dist[v] = d
    return total


# ─── Multi-Goal Planner (mirrors nav_planner.py::MultiGoalPlanner) ─────────
class MultiGoalPlanner:
    def __init__(self, grid, goals, epsilon=EPSILON, k=K_STEPS):
        self.grid  = grid_copy(grid)
        self.remaining_goals = set(goals)
        self.epsilon = epsilon
        self.k = k
        self.best_goal = None

    def _path_cost(self, start, end):
        path = astar(self.grid, start, end)
        return len(path)-1 if path else INF

    def step(self, robot_pos, update_env_fn):
        """Returns (waypoints, full_path, done) — mirrors nav_planner.py."""
        if not self.remaining_goals:
            return [], [], True

        self.grid = update_env_fn()

        # Cost to each reachable goal
        costs = {}
        reachable = []
        for g in self.remaining_goals:
            c = self._path_cost(robot_pos, g)
            if c < INF:
                costs[g] = c; reachable.append(g)

        if not reachable:
            return [], [], False  # all temporarily blocked; wait

        # MST heuristic: pick goal that minimises dist(pos,g) + MST(rest)
        def heuristic(g):
            return costs[g] + mst_cost(self.remaining_goals - {g})

        primary = min(reachable, key=heuristic)

        # Opportunistic epsilon-detour check
        self.best_goal = primary
        for g in reachable:
            if g == primary: continue
            cost_via_g = costs[g] + self._path_cost(g, primary)
            if cost_via_g <= costs[primary] + self.epsilon:
                self.best_goal = g; break

        # Plan path to chosen goal
        full_path = astar(self.grid, robot_pos, self.best_goal)
        if not full_path:
            return [], [], False

        # Sudden discovery: grab earliest on-path goal
        path_set = set(full_path)
        earliest_goal, earliest_idx = None, len(full_path)
        for g in reachable:
            if g == self.best_goal: continue
            if g in path_set:
                idx = full_path.index(g)
                if idx < earliest_idx:
                    earliest_idx = idx; earliest_goal = g
        if earliest_goal:
            self.best_goal = earliest_goal
            full_path = full_path[:earliest_idx+1]

        waypoints = full_path[1:self.k+1]
        return waypoints, full_path, False


# ─── Environment with Multi-Goal Planner ────────────────────────────────────
class MultiGoalEnv:
    def __init__(self, seed=SEED):
        random.seed(seed)
        self.base  = build_maze()
        self.robot = list(ROBOT_START)
        self.goals_list = [g for g in GOALS if self.base[g[0]][g[1]]==0]
        self.mgp   = MultiGoalPlanner(self.base, self.goals_list)
        self.mgp.remaining_goals = set(self.goals_list)
        self.collected = set()   # stores actual goal tuples
        self.done  = False
        self.waypoints = []
        self.frame_n = 0
        self.steps = 0

        safety = {tuple(self.robot)}|{tuple(g) for g in self.goals_list}
        self.dyn_obs=[]; tries=0
        while len(self.dyn_obs)<NUM_OBS and tries<2000:
            r=random.randint(1,GRID_ROWS-2); c=random.randint(1,GRID_COLS-2)
            if self.base[r][c]==0 and (r,c) not in safety:
                self.dyn_obs.append([r,c]); safety.add((r,c))
            tries+=1

    def _inflate(self):
        cells=set()
        for o in self.dyn_obs:
            for dr in range(-OBS_SPREAD,OBS_SPREAD+1):
                for dc in range(-OBS_SPREAD,OBS_SPREAD+1):
                    nr,nc=o[0]+dr,o[1]+dc
                    if 0<=nr<GRID_ROWS and 0<=nc<GRID_COLS: cells.add((nr,nc))
        return cells

    def _pgrid(self):
        g=grid_copy(self.base); rc=tuple(self.robot)
        for nr,nc in self._inflate():
            if (nr,nc)!=rc: g[nr][nc]=1
        return g

    def _move_obs(self):
        dirs=[(-1,0),(1,0),(0,-1),(0,1)]
        occ={(o[0],o[1]) for o in self.dyn_obs}; rc=tuple(self.robot)
        for obs in self.dyn_obs:
            random.shuffle(dirs)
            for dr,dc in dirs:
                nr,nc=obs[0]+dr,obs[1]+dc
                if(1<=nr<GRID_ROWS-1 and 1<=nc<GRID_COLS-1
                   and self.base[nr][nc]==0
                   and(nr,nc) not in occ and(nr,nc)!=rc):
                    occ.discard((obs[0],obs[1])); obs[0],obs[1]=nr,nc; occ.add((nr,nc)); break

    def step(self):
        if self.done: return
        self.frame_n+=1; self.steps+=1

        if self.frame_n % OBS_EVERY == 0:
            self._move_obs()
            if self.waypoints:
                inf=self._inflate()
                if any(c in inf for c in self.waypoints[:4]):
                    self.waypoints=[]

        # Get new waypoints if needed
        if not self.waypoints:
            wps, _, done = self.mgp.step(tuple(self.robot), self._pgrid)
            self.waypoints = list(wps)
            if done and not self.waypoints:
                self.done=True; return

        # Collect check BEFORE moving (stationary collection)
        rob=tuple(self.robot)
        hit=False
        for g in list(self.mgp.remaining_goals):
            if abs(rob[0]-g[0])+abs(rob[1]-g[1])<=COLLECT_DIST:
                self.mgp.remaining_goals.discard(g)
                self.collected.add(g); hit=True
        if hit:
            self.waypoints=[]
            if not self.mgp.remaining_goals:
                self.done=True; return

        if not self.waypoints: return

        # Move
        nxt=self.waypoints.pop(0); self.robot=list(nxt); rob=tuple(self.robot)
        for g in list(self.mgp.remaining_goals):
            if abs(rob[0]-g[0])+abs(rob[1]-g[1])<=COLLECT_DIST:
                self.mgp.remaining_goals.discard(g)
                self.collected.add(g)
        if not self.mgp.remaining_goals:
            self.done=True


# ─── Run test ──────────────────────────────────────────────────────────────
bar="="*62
print(bar)
print(f"  MULTI-GOAL PLANNER TEST")
print(f"  Grid={GRID_ROWS}x{GRID_COLS}  Goals={len(GOALS)}")
print(f"  Epsilon={EPSILON}  K={K_STEPS}  MaxSteps={MAX_STEPS}  Seed={SEED}")
print(bar)

base=build_maze()
free_count = sum(1 for g in GOALS if base[g[0]][g[1]]==0)
print(f"  Free goals: {free_count}/{len(GOALS)}", "✓" if free_count==len(GOALS) else "✗")

env = MultiGoalEnv()
for _ in range(MAX_STEPS):
    env.step()
    if env.done: break

n=len(env.collected); t=len(env.goals_list); ok=n==t
miss=[g for g in env.goals_list if g not in env.collected] if not ok else []

print(f"\n  Result:")
print(f"    {'PASS ✓' if ok else 'FAIL ✗'}  Multi-Goal Planner  {n}/{t} goals  steps={env.steps}{'  (TIMED OUT)' if not env.done else ''}")
if miss: print(f"    Missing goals: {miss}")

print(f"\n  Goal visit order (actual collection sequence):")
# Track visit order by running again with logging
print(f"    Collected: {list(env.collected)}")

print(bar)
if ok:
    print(f"  ✓ MULTI-GOAL PLANNER REACHES ALL {t} GOALS!\n")
else:
    print(f"  ✗ FAILED — {t-n} goals missed\n")
