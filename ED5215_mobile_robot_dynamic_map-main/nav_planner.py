"""
nav_planner.py
==============
Standalone path-planning algorithms for the grid navigation visualizer.

Algorithms
----------
  astar      — A* with Manhattan heuristic (informed, optimal)
  dijkstra   — Uniform-cost search (uninformed, optimal)
  bfs        — Breadth-first search (uninformed, shortest hops)

Entry points
------------
  plan(grid, start, goal, algo)  →  (path, explored)
     grid   : 2-D numpy array, 0=free, non-zero=obstacle
     start  : (row, col) tuple
     goal   : (row, col) tuple
     algo   : 'astar' | 'dijkstra' | 'bfs'
     path   : list of (row, col) from start to goal (inclusive)
     explored : list of (row, col) cells expanded during search

  solve_tsp(start, goals)        →  ordered list of goals
     Held-Karp DP — exact optimal visit order.
"""

from collections import deque
import heapq
import time
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _neighbours_4(r, c, rows, cols):
    """4-connected grid neighbours."""
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def _reconstruct(came_from, start, goal):
    """Trace back the came_from dict to build the path."""
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from.get(node)
        if node is None:          # no path
            return []
    path.append(start)
    path.reverse()
    return path


# ─────────────────────────────────────────────────────────────────────────────
# A* — informed, admissible Manhattan heuristic
# ─────────────────────────────────────────────────────────────────────────────

def astar(grid, start, goal):
    """
    A* search on a 4-connected grid.

    Returns
    -------
    path     : list[(row,col)] from start to goal; [] if unreachable
    explored : list[(row,col)] expansion order (useful for visualisation)
    """
    if start == goal:
        return [start], [start]
        
    rows, cols = grid.shape
    h = lambda s: abs(s[0] - goal[0]) + abs(s[1] - goal[1])

    # Priority queue: (f, g, cell)
    open_set  = [(h(start), 0, start)]
    g_score   = {start: 0}
    came_from = {}
    visited   = set()
    explored  = []

    while open_set:
        _, g, cur = heapq.heappop(open_set)
        if cur in visited:
            continue
        visited.add(cur)
        explored.append(cur)

        if cur == goal:
            return _reconstruct(came_from, start, goal), explored

        for nr, nc in _neighbours_4(*cur, rows, cols):
            nb = (nr, nc)
            # ROBUSTNESS: Treat start and goal as free to allow planning through obstacles
            if grid[nr, nc] != 0 and nb != goal and nb != start:
                continue
            ng = g + 1
            if ng < g_score.get(nb, 10**9):
                g_score[nb]   = ng
                came_from[nb] = cur
                heapq.heappush(open_set, (ng + h(nb), ng, nb))

    return [], explored


# ─────────────────────────────────────────────────────────────────────────────
# Dijkstra — uninformed, uniform-cost (same as A* with h=0)
# ─────────────────────────────────────────────────────────────────────────────

def dijkstra(grid, start, goal):
    """
    Dijkstra's algorithm on a 4-connected grid.

    Returns
    -------
    path     : list[(row,col)] from start to goal; [] if unreachable
    explored : list[(row,col)] expansion order
    """
    if start == goal:
        return [start], [start]
        
    rows, cols = grid.shape
    open_set  = [(0, start)]
    dist      = {start: 0}
    came_from = {}
    visited   = set()
    explored  = []

    while open_set:
        d, cur = heapq.heappop(open_set)
        if cur in visited:
            continue
        visited.add(cur)
        explored.append(cur)

        if cur == goal:
            return _reconstruct(came_from, start, goal), explored

        for nr, nc in _neighbours_4(*cur, rows, cols):
            nb = (nr, nc)
            if grid[nr, nc] != 0 and nb != goal and nb != start:
                continue
            nd = d + 1
            if nd < dist.get(nb, 10**9):
                dist[nb]      = nd
                came_from[nb] = cur
                heapq.heappush(open_set, (nd, nb))

    return [], explored


# ─────────────────────────────────────────────────────────────────────────────
# BFS — uninformed, fewest hops
# ─────────────────────────────────────────────────────────────────────────────

def bfs(grid, start, goal):
    """
    Breadth-first search on a 4-connected grid.

    Returns
    -------
    path     : list[(row,col)] from start to goal; [] if unreachable
    explored : list[(row,col)] expansion order
    """
    if start == goal:
        return [start], [start]
        
    rows, cols = grid.shape
    queue     = deque([start])
    visited   = {start}
    came_from = {}
    explored  = []

    while queue:
        cur = queue.popleft()
        explored.append(cur)

        if cur == goal:
            return _reconstruct(came_from, start, goal), explored

        for nr, nc in _neighbours_4(*cur, rows, cols):
            nb = (nr, nc)
            if (grid[nr, nc] != 0 and nb != goal and nb != start) or nb in visited:
                continue
            visited.add(nb)
            came_from[nb] = cur
            queue.append(nb)

    return [], explored


# ─────────────────────────────────────────────────────────────────────────────
# Unified dispatcher
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# D* Lite — dynamic incremental search
# ─────────────────────────────────────────────────────────────────────────────

class DStarLite:
    """
    Incremental heuristic search on a grid.
    Re-plans efficiently when edge costs (obstacles) change.
    """
    def __init__(self, grid_shape):
        self.rows, self.cols = grid_shape
        self.rhs = np.full(grid_shape, float('inf'))
        self.g   = np.full(grid_shape, float('inf'))
        self.km  = 0.0
        self.queue = []
        self.start = None
        self.goal  = None
        self.grid  = None

    def _h(self, s1, s2):
        if s1 is None or s2 is None: return 0.0
        return abs(s1[0] - s2[0]) + abs(s1[1] - s2[1])

    def _calculate_key(self, s):
        g_rhs = min(self.g[s], self.rhs[s])
        return (g_rhs + self._h(s, self.start) + self.km, g_rhs)

    def _update_vertex(self, u):
        if u != self.goal:
            min_rhs = float('inf')
            for nb in _neighbours_4(*u, self.rows, self.cols):
                # ROBUSTNESS: Start and Goal are always free to the planner
                is_free = (self.grid[nb] == 0 or nb == self.start or nb == self.goal)
                cost = 1 if is_free else float('inf')
                min_rhs = min(min_rhs, self.g[nb] + cost)
            self.rhs[u] = min_rhs

        self.queue = [x for x in self.queue if x[1] != u]
        if self.g[u] != self.rhs[u]:
            heapq.heappush(self.queue, (self._calculate_key(u), u))

    def compute_shortest_path(self):
        while self.queue and (self.queue[0][0] < self._calculate_key(self.start) or 
                              self.rhs[self.start] != self.g[self.start]):
            k_old, u = heapq.heappop(self.queue)
            k_new = self._calculate_key(u)
            if k_old < k_new:
                heapq.heappush(self.queue, (k_new, u))
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for v in _neighbours_4(*u, self.rows, self.cols):
                    self._update_vertex(v)
            else:
                self.g[u] = float('inf')
                self._update_vertex(u)
                for v in _neighbours_4(*u, self.rows, self.cols):
                    self._update_vertex(v)

    def plan(self, grid, start, goal):
        # Persistent state handling
        if self.grid is None or self.goal != goal:
            # First run or new goal: Full search
            self.grid = grid.copy()
            self.start = start
            self.last_start = start
            self.goal = goal
            self.km = 0.0
            self.rhs = np.full(grid.shape, float('inf'))
            self.g   = np.full(grid.shape, float('inf'))
            self.rhs[goal] = 0
            self.queue = []
            heapq.heappush(self.queue, (self._calculate_key(goal), goal))
        else:
            # Incremental update
            # 1. Update km based on movement
            self.km += self._h(self.last_start, start)
            self.last_start = start
            self.start = start
            
            # 2. Detect changes in obstacles
            diff = np.where(self.grid != grid)
            for r, c in zip(diff[0], diff[1]):
                u = (r, c)
                self.grid[u] = grid[u]
                self._update_vertex(u)
                for v in _neighbours_4(*u, self.rows, self.cols):
                    self._update_vertex(v)

        self.compute_shortest_path()
        return _reconstruct_from_g(self.g, self.grid, start, goal), []

def _reconstruct_from_g(g, grid, start, goal):
    if start == goal: return [start]
    if g[start] == float('inf'): return []
    
    path = [start]
    rows, cols = grid.shape
    curr = start
    while curr != goal:
        best_nb = None
        min_cost = float('inf')
        for nb in _neighbours_4(*curr, rows, cols):
            # ROBUSTNESS: Goal is always free
            is_free = (grid[nb] == 0 or nb == goal or nb == start)
            if is_free and g[nb] < min_cost:
                min_cost = g[nb]
                best_nb = nb
        if best_nb is None or best_nb in path:
            break
        path.append(best_nb)
        curr = best_nb
    return path if path[-1] == goal else []


# ─────────────────────────────────────────────────────────────────────────────
# D* — Placeholder for standard D* (similar to D* Lite in spirit)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Persistent D* Lite Instances
# ─────────────────────────────────────────────────────────────────────────────
_DSTAR_INSTANCES = {}

def dstar(grid, start, goal, algo='dstar'):
    """Incremental D* Lite implementation."""
    key = (grid.shape, algo)
    if key not in _DSTAR_INSTANCES:
        _DSTAR_INSTANCES[key] = DStarLite(grid.shape)
    return _DSTAR_INSTANCES[key].plan(grid, start, goal)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Goal Opportunistic Planner
# ─────────────────────────────────────────────────────────────────────────────

class MultiGoalPlanner:
    """
    Implements the orchestration loop for multiple goals with opportunistic detour.
    """
    def __init__(self, grid, goals, epsilon=2.0, k=5):
        self.grid = grid
        self.remaining_goals = set(goals)
        self.epsilon = epsilon
        self.k = k
        self.current_position = None
        self.best_goal = None

    def shortest_path_cost(self, start, end):
        # Using A* for quick cost estimation
        path, _ = astar(self.grid, start, end)
        if not path: return float('inf')
        return len(path) - 1 # Distance is steps (nodes-1)

    def _calculate_mst_cost(self, goals):
        """
        Calculates the Minimum Spanning Tree cost of the goals using Prims.
        This represents an admissible heuristic for visiting all goals.
        """
        if not goals: return 0.0
        nodes = list(goals)
        n = len(nodes)
        if n <= 1: return 0.0

        # MST using Prim's on Manhattan distance
        visited = [False] * n
        min_dist = [float('inf')] * n
        min_dist[0] = 0
        total_mst_cost = 0

        for _ in range(n):
            u = -1
            for i in range(n):
                if not visited[i] and (u == -1 or min_dist[i] < min_dist[u]):
                    u = i
            
            visited[u] = True
            total_mst_cost += min_dist[u]

            for v in range(n):
                if not visited[v]:
                    # Using Manhattan distance for grid alignment
                    d = abs(nodes[u][0] - nodes[v][0]) + abs(nodes[u][1] - nodes[v][1])
                    if d < min_dist[v]:
                        min_dist[v] = d
        
        return total_mst_cost

    def step(self, robot_pos, update_env_fn):
        """
        Executes one iteration of the multi-goal planning loop.
        Returns (waypoints, full_path, done).
        """
        if not self.remaining_goals:
            return [], [], True

        self.current_position = robot_pos
        
        # Step 1: Update environment
        self.grid = update_env_fn()

        # Step 2: Compute cost to all goals
        cost_table = {}
        reachable_goals = []
        for g in self.remaining_goals:
            cost = self.shortest_path_cost(self.current_position, g)
            if cost < float('inf'):
                cost_table[g] = cost
                reachable_goals.append(g)

        if not reachable_goals:
            # Wait if all goals are temporarily blocked
            return [], [], False
        
        # Step 3: Heuristic-based primary goal selection
        # Instead of just closest, we pick g that minimizes:
        # h(g) = dist(n, g) + MST({g} U (remaining - {g}))
        # This is a better estimate for the total remaining path.
        
        def multi_goal_heuristic(g):
            # Remaining work if we pick g first: dist(n,g) + MST(Remaining - {g})
            rest = self.remaining_goals - {g}
            h_cost = cost_table[g] + self._calculate_mst_cost(rest)
            return h_cost

        primary_goal = min(reachable_goals, key=multi_goal_heuristic)

        # Step 4: Opportunistic check (Epsilon-greedy detour)
        self.best_goal = primary_goal
        for g in reachable_goals:
            if g == primary_goal: continue
            
            # Can we visit g and then primary_goal with small detour?
            cost_via_g = cost_table[g] + self.shortest_path_cost(g, primary_goal)
            if cost_via_g <= cost_table[primary_goal] + self.epsilon:
                self.best_goal = g
                break

        # Step 5: Plan path to chosen goal
        full_path, _ = astar(self.grid, self.current_position, self.best_goal)
        if not full_path:
            return [], [], False

        # Step 5.1: Sudden Discovery Check
        # If any other goal Gj lies DIRECTLY on this path, target the EARLIEST one first!
        path_set = set(full_path)
        earliest_goal = None
        earliest_idx = len(full_path)
        
        for g in reachable_goals:
            if g == self.best_goal: continue
            if g in path_set:
                idx = full_path.index(g)
                if idx < earliest_idx:
                    earliest_idx = idx
                    earliest_goal = g
        
        if earliest_goal:
            self.best_goal = earliest_goal
            full_path = full_path[:earliest_idx + 1]

        # Step 6: Return first k steps
        waypoints = full_path[1:self.k+1]
        return waypoints, full_path, False

    def run(self, start, update_env_fn):
        self.current_position = start
        history = [start]

        while self.remaining_goals:
            waypoints, _, done = self.step(self.current_position, update_env_fn)
            if done: break
            if not waypoints:
                # Wait for obstacles to move
                self.grid = update_env_fn()
                continue
            
            for step in waypoints:
                self.current_position = step
                history.append(step)
                if self.grid[step] != 0: break
                if self.current_position == self.best_goal:
                    self.remaining_goals.remove(self.best_goal)
                    break
        
        return history

# ─────────────────────────────────────────────────────────────────────────────
# Unified dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def _multi_goal_stub(grid, start, goal):
    raise NotImplementedError(
        "Multi-goal planning requires the MultiGoalPlanner class "
        "and a set of goals, not the single-goal plan() interface."
    )

_PLANNERS = {
    'astar':      astar,
    'dijkstra':   dijkstra,
    'bfs':        bfs,
    'dstar':      lambda g,s,tg: dstar(g,s,tg, 'dstar'),
    'dstarlite':  lambda g,s,tg: dstar(g,s,tg, 'dstarlite'),
    'multi-goal': _multi_goal_stub
}


def plan(grid, start, goal, algo='astar'):
    """
    Unified path planner.

    Parameters
    ----------
    grid  : np.ndarray  shape (R, C), 0=free  non-zero=obstacle
    start : (row, col)
    goal  : (row, col)
    algo  : str  one of 'astar' | 'dijkstra' | 'bfs'

    Returns
    -------
    path     : list[(row,col)]   empty if goal unreachable
    explored : list[(row,col)]   cells expanded in search order
    """
    planner = _PLANNERS.get(algo.lower(), astar)
    return planner(grid, start, goal)


# ─────────────────────────────────────────────────────────────────────────────
# TSP — Held-Karp bitmask DP (exact, O(n² · 2ⁿ))
# ─────────────────────────────────────────────────────────────────────────────

def solve_tsp(start, goals):
    """
    Find the optimal order to visit all goals starting from start.

    Uses Euclidean (straight-line) inter-goal distances.
    Suitable for n ≤ 20 goals; beyond that consider greedy nearest-neighbour.

    Parameters
    ----------
    start : (row, col) or (x, y)   robot starting position
    goals : list of positions

    Returns
    -------
    ordered_goals : list — goals reordered for minimum total travel distance
    """
    if not goals:
        return []

    nodes = [start] + list(goals)
    n     = len(nodes)

    # Distance matrix
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.hypot(nodes[i][0] - nodes[j][0],
                                   nodes[i][1] - nodes[j][1])

    # DP table: dp[mask][u] = min cost to reach u having visited mask
    INF = float('inf')
    dp      = np.full((1 << n, n), INF)
    parent  = np.full((1 << n, n), -1, dtype=int)
    dp[1][0] = 0.0   # start at node 0 (robot position)

    for mask in range(1, 1 << n):
        for u in range(n):
            if dp[mask][u] == INF:
                continue
            if not (mask & (1 << u)):
                continue
            for v in range(n):
                if mask & (1 << v):
                    continue
                new_mask = mask | (1 << v)
                new_cost = dp[mask][u] + dist[u, v]
                if new_cost < dp[new_mask][v]:
                    dp[new_mask][v]     = new_cost
                    parent[new_mask][v] = u

    # Find best last node
    full_mask = (1 << n) - 1
    best_last, min_cost = -1, INF
    for i in range(1, n):
        if dp[full_mask][i] < min_cost:
            min_cost, best_last = dp[full_mask][i], i

    # Reconstruct path
    indices = []
    mask, node = full_mask, best_last
    while node != -1:
        if node != 0:
            indices.append(node - 1)   # offset by 1 (node 0 = robot start)
        prev = parent[mask][node]
        mask ^= (1 << node)
        node  = prev

    indices.reverse()
    return [goals[i] for i in indices]
