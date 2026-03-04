import numpy as np
import heapq


class DStarLite:
    """
    D* Lite path planner for a 2-D grid.
    Supports static obstacles (walls) and dynamic obstacles (Bills).
    Static obstacles are kept in base_map and never erased by dynamic updates.
    """

    def __init__(self, x_min=-1.0, x_max=11.0, y_min=-1.0, y_max=11.0, resolution=0.2):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.res   = resolution

        self.nx = int((x_max - x_min) / resolution)
        self.ny = int((y_max - y_min) / resolution)

        # base_map: permanently blocked cells (walls).  1=free, inf=obstacle.
        self.base_map = np.ones((self.nx, self.ny))
        # map: combined static + dynamic obstacles used for planning
        self.map = np.ones((self.nx, self.ny))

        self.goal       = None
        self.start      = None
        self.last_start = None

        self.rhs = np.full((self.nx, self.ny), float('inf'))
        self.g   = np.full((self.nx, self.ny), float('inf'))
        self.km  = 0.0
        self.queue = []

        # Grid cells currently occupied by Bills (dynamic obstacles)
        self.dynamic_obstacles = set()

        # BFS-precomputed navigable distance from start through free cells.
        # None until precompute_h_map() is called.
        self.h_map = None

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------
    def world_to_grid(self, x, y):
        gx = int((x - self.x_min) / self.res)
        gy = int((y - self.y_min) / self.res)
        return (max(0, min(self.nx - 1, gx)),
                max(0, min(self.ny - 1, gy)))

    def grid_to_world(self, gx, gy):
        x = gx * self.res + self.x_min + self.res / 2
        y = gy * self.res + self.y_min + self.res / 2
        return x, y

    # ------------------------------------------------------------------
    # D* Lite internals
    # ------------------------------------------------------------------
    def precompute_h_map(self):
        """
        BFS from self.start outward through FREE cells only.

        Result: self.h_map[i, j]  = shortest navigable distance (in metres)
                from cell (i,j) to self.start, travelling only through
                cells where map != inf (i.e. no walls, no Bills).

        This replaces the straight-line Euclidean heuristic so D* expands
        nodes in order of TRUE navigable proximity to the robot — the path
        priority correctly routes around walls rather than through them.
        """
        from collections import deque
        h = np.full((self.nx, self.ny), float('inf'))
        if self.start is None:
            self.h_map = h
            return

        h[self.start] = 0.0
        q = deque([self.start])
        while q:
            s = q.popleft()
            for nb in self.get_neighbors(s):
                if self.map[nb] != float('inf') and h[nb] == float('inf'):
                    h[nb] = h[s] + self.res   # convert cell steps → metres
                    q.append(nb)
        self.h_map = h
        reachable = np.sum(h < float('inf'))
        print(f"  [BFS h-map] {reachable} free cells reachable from start {self.start}")

    def heuristic(self, s1, s2):
        """
        Navigable-distance heuristic for D* Lite.

        When s2 is the robot start and h_map has been precomputed, returns
        the true shortest FREE-SPACE distance from s1 to s2 (BFS through
        open cells, never through walls).  Falls back to Euclidean only
        when the BFS map is unavailable or s1 is unreachable.
        """
        if s1 is None or s2 is None:
            return 0.0
        # Use BFS navigable distance when querying h(s, start)
        if self.h_map is not None and s2 == self.start:
            val = float(self.h_map[s1])
            if val < float('inf'):
                return val
        # Fallback: admissible Euclidean lower bound
        return float(np.sqrt((s1[0] - s2[0]) ** 2 + (s1[1] - s2[1]) ** 2))

    def calculate_key(self, s):
        if self.start is None:
            return (float('inf'), float('inf'))
        g_rhs = min(float(self.g[s]), float(self.rhs[s]))
        return (g_rhs + self.heuristic(s, self.start) + self.km, g_rhs)

    def get_neighbors(self, s):
        """4-connected neighbours only (no diagonals) — prevents corner-cutting."""
        result = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = s[0] + dx, s[1] + dy
            if 0 <= nx < self.nx and 0 <= ny < self.ny:
                result.append((nx, ny))
        return result

    def _edge_cost(self, u, v):
        """Cost to move from u to v (must be adjacent cells).
        Wall cells have infinite cost. Cells adjacent to walls carry a
        proximity penalty to keep paths through corridor centres.
        Uses self.res (one grid step in metres) as the base move cost.
        """
        if self.map[v] != 1.0:
            return float('inf')
        cost = self.res   # one grid step = resolution metres
        # Mild wall proximity penalty — prefers centre of corridors but never
        # blocks navigation through them.  0.5 = 3.5x more expensive than open.
        for vn in self.get_neighbors(v):
            if self.base_map[vn] == float('inf'):
                cost += 0.5
                break
        return cost

    def update_vertex(self, u):
        if u != self.goal:
            if self.map[u] == float('inf'):
                self.rhs[u] = float('inf')
            else:
                min_rhs = float('inf')
                for v in self.get_neighbors(u):
                    c = self._edge_cost(u, v)
                    min_rhs = min(min_rhs, float(self.g[v]) + c)
                self.rhs[u] = min_rhs

        # Update priority queue
        self.queue = [(k, n) for k, n in self.queue if n != u]
        heapq.heapify(self.queue)
        if float(self.g[u]) != float(self.rhs[u]):
            heapq.heappush(self.queue, (self.calculate_key(u), u))

    def compute_shortest_path(self):
        if self.start is None or self.goal is None:
            return
        while self.queue:
            top_key = self.queue[0][0]
            start_key = self.calculate_key(self.start)
            if not (top_key < start_key or
                    float(self.rhs[self.start]) != float(self.g[self.start])):
                break
            k_old, u = heapq.heappop(self.queue)
            k_new = self.calculate_key(u)
            if k_old < k_new:
                heapq.heappush(self.queue, (k_new, u))
            elif float(self.g[u]) > float(self.rhs[u]):
                self.g[u] = self.rhs[u]
                for v in self.get_neighbors(u):
                    self.update_vertex(v)
            else:
                self.g[u] = float('inf')
                self.update_vertex(u)
                for v in self.get_neighbors(u):
                    self.update_vertex(v)

    # ------------------------------------------------------------------
    # Initialisation — CORRECT ORDER: set_start → set_goal → add obstacles
    # ------------------------------------------------------------------
    def set_start(self, start_world):
        """Store the robot's starting position."""
        if start_world is None:
            return
        self.start = self.world_to_grid(start_world[0], start_world[1])
        if self.last_start is None:
            self.last_start = self.start

    def set_goal(self, goal_world):
        """Set goal and begin D* expansion.  Snaps to nearest free cell if needed."""
        gx, gy = self.world_to_grid(goal_world[0], goal_world[1])

        # Snap to nearest free cell when goal falls inside an inflated wall
        if self.map[(gx, gy)] == float('inf'):
            found = False
            for radius in range(1, 15):
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if abs(dx) != radius and abs(dy) != radius:
                            continue
                        nx_, ny_ = gx + dx, gy + dy
                        if 0 <= nx_ < self.nx and 0 <= ny_ < self.ny \
                                and self.map[(nx_, ny_)] == 1.0:
                            print(f"  [Goal snapped grid({gx},{gy})→({nx_},{ny_})]")
                            gx, gy = nx_, ny_
                            found = True
                            break
                    if found: break
                if found: break

        self.goal = (gx, gy)
        self.rhs[self.goal] = 0.0
        heapq.heappush(self.queue, (self.calculate_key(self.goal), self.goal))
        self.compute_shortest_path()

    # ------------------------------------------------------------------
    # Static obstacles (walls — permanent)
    # ------------------------------------------------------------------
    def add_boundary_walls(self, thickness=2):
        """Block the outer cells as permanent boundary walls."""
        changed = set()
        for ix in range(self.nx):
            for iy in range(self.ny):
                border = (ix < thickness or ix >= self.nx - thickness or
                          iy < thickness or iy >= self.ny - thickness)
                if border and self.base_map[ix, iy] == 1.0:
                    self.base_map[ix, iy] = float('inf')
                    self.map[ix, iy] = float('inf')
                    node = (ix, iy)
                    changed.add(node)
                    for n in self.get_neighbors(node):
                        changed.add(n)
        print(f"  Boundary: {len(changed)} cells blocked ({thickness} cells thick)")
        if changed and self.goal is not None:
            for node in changed:
                self.update_vertex(node)
            self.compute_shortest_path()

    def add_static_obstacle(self, x, y, size_x, size_y, orientation=0.0):
        """Block a rectangular wall region permanently in the grid.
        safety=0.6m: robot body (~0.4m) + 0.2m extra clearance on each side.
        """
        gx, gy = self.world_to_grid(x, y)
        safety = 0.4          # 0.4m: sufficient clearance, preserves corridor width
        rx = int((size_x + safety) / (2 * self.res)) + 1
        ry = int((size_y + safety) / (2 * self.res)) + 1
        if abs(orientation) > 0.78:   # wall rotated > 45° → swap dims
            rx, ry = ry, rx

        changed = set()
        for ix in range(max(0, gx - rx), min(self.nx, gx + rx + 1)):
            for iy in range(max(0, gy - ry), min(self.ny, gy + ry + 1)):
                if self.base_map[ix, iy] == 1.0:
                    self.base_map[ix, iy] = float('inf')
                    self.map[ix, iy] = float('inf')
                    node = (ix, iy)
                    changed.add(node)
                    for n in self.get_neighbors(node):
                        changed.add(n)

        if changed and self.goal is not None:
            for node in changed:
                self.update_vertex(node)
            self.compute_shortest_path()

    # ------------------------------------------------------------------
    # Dynamic obstacles (Bills — updated every control step)
    # ------------------------------------------------------------------
    def update_obstacles(self, robot_pos_world, bills_pos_world):
        """Re-plan around Bills without ever erasing static walls."""
        if robot_pos_world is None or bills_pos_world is None:
            return

        self.start = self.world_to_grid(robot_pos_world[0], robot_pos_world[1])

        # Build new set of Bill-occupied cells (5×5 box around each Bill)
        new_dyn = set()
        for bx, by in bills_pos_world:
            gx, gy = self.world_to_grid(bx, by)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx_, ny_ = gx + dx, gy + dy
                    if 0 <= nx_ < self.nx and 0 <= ny_ < self.ny:
                        new_dyn.add((nx_, ny_))

        appeared   = new_dyn - self.dynamic_obstacles
        disappeared = self.dynamic_obstacles - new_dyn

        changed = set()
        for node in appeared:
            if self.map[node] == 1.0:
                self.map[node] = float('inf')
                changed.add(node)
                for n in self.get_neighbors(node):
                    changed.add(n)

        for node in disappeared:
            if self.base_map[node] == 1.0:   # restore only if not a wall
                self.map[node] = 1.0
                changed.add(node)
                for n in self.get_neighbors(node):
                    changed.add(n)

        self.dynamic_obstacles = new_dyn

        if changed:
            # km accumulates robot displacement — always use Euclidean here,
            # NOT the BFS h_map (which is goal-relative, not displacement).
            dx = float(self.start[0] - self.last_start[0]) * self.res
            dy = float(self.start[1] - self.last_start[1]) * self.res
            self.km += float(np.sqrt(dx*dx + dy*dy))
            self.last_start = self.start
            for node in changed:
                self.update_vertex(node)
            self.compute_shortest_path()

    # ------------------------------------------------------------------
    # Waypoint extraction
    # ------------------------------------------------------------------
    def get_next_waypoint(self, robot_pos_world):
        new_start = self.world_to_grid(robot_pos_world[0], robot_pos_world[1])

        if self.last_start is None:
            self.last_start = new_start

        if new_start != self.start:
            # km accumulates robot displacement — always Euclidean, not BFS h_map.
            dx = float(new_start[0] - self.last_start[0]) * self.res
            dy = float(new_start[1] - self.last_start[1]) * self.res
            self.km += float(np.sqrt(dx*dx + dy*dy))
            self.last_start = new_start
            self.start = new_start
            self.compute_shortest_path()

        # Greedy lookahead — follow 7 steps along the D* gradient.
        # 7 steps = 1.4 m ahead: far enough for smooth straight runs,
        # close enough to still react to walls and Bills.
        curr = self.start
        path = [curr]
        for _ in range(7):
            u = path[-1]
            best, best_cost = None, float('inf')
            for v in self.get_neighbors(u):
                c = self._edge_cost(u, v) + float(self.g[v])
                if c < best_cost:
                    best_cost, best = c, v
            if best and best not in path:
                path.append(best)
            else:
                break

        target = path[-1]

        # ── Stuck-escape: greedy lookahead found no improvement ───────────────
        # Prefer non-wall-adjacent free cells first; accept wall-adjacent only
        # if there is truly no other option, so the robot never freezes.
        if target == curr:
            open_escape,  open_cost  = None, float('inf')   # away from walls
            tight_escape, tight_cost = None, float('inf')   # wall-adjacent
            for v in self.get_neighbors(curr):
                if self.map[v] == float('inf'):
                    continue                               # hard wall — skip
                gv = float(self.g[v])
                # Check whether v is wall-adjacent
                near_wall = any(self.base_map[nb] == float('inf')
                                for nb in self.get_neighbors(v))
                if not near_wall and gv < open_cost:
                    open_cost,  open_escape  = gv, v
                elif near_wall and gv < tight_cost:
                    tight_cost, tight_escape = gv, v
            # Prefer open cell; fall back to wall-adjacent only if unavoidable
            target = open_escape or tight_escape or curr

        # Truly surrounded — return current position (caller handles fallback)
        if target == curr:
            return (robot_pos_world[0], robot_pos_world[1])

        wx, wy = self.grid_to_world(target[0], target[1])

        # Hard clamp — keep waypoint inside safe interior
        margin = self.res * 3
        wx = max(self.x_min + margin, min(self.x_max - margin, wx))
        wy = max(self.y_min + margin, min(self.y_max - margin, wy))
        return (wx, wy)


# ==================================================================
# TSP solver — Held-Karp bitmask DP for optimal goal visit order
# ==================================================================
def solve_tsp(start_pos, goal_positions):
    """Returns goal_positions reordered for minimum total travel distance."""
    if not goal_positions:
        return []

    nodes = [start_pos] + list(goal_positions)
    n = len(nodes)

    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(
                np.array(nodes[i][:2]) - np.array(nodes[j][:2]))

    dp = np.full((1 << n, n), float('inf'))
    parent = np.full((1 << n, n), -1)
    dp[1][0] = 0.0

    for mask in range(1, 1 << n):
        for u in range(n):
            if dp[mask][u] == float('inf'):
                continue
            for v in range(n):
                if not (mask & (1 << v)):
                    new_mask = mask | (1 << v)
                    new_cost = dp[mask][u] + dist[u, v]
                    if new_cost < dp[new_mask][v]:
                        dp[new_mask][v] = new_cost
                        parent[new_mask][v] = u

    full_mask = (1 << n) - 1
    best_last, min_cost = -1, float('inf')
    for i in range(1, n):
        if dp[full_mask][i] < min_cost:
            min_cost = dp[full_mask][i]
            best_last = i

    # Reconstruct path
    indices = []
    mask, node = full_mask, best_last
    while node != -1:
        if node != 0:
            indices.append(node - 1)
        prev = parent[mask][node]
        mask ^= (1 << node)
        node = prev

    indices.reverse()
    return [goal_positions[i] for i in indices]


# ==================================================================
# Multi-Goal Planner
# Implements the pseudocode loop:
#   1. Update dynamic map
#   2. Compute cost to all goals
#   3. Score goals with MST future estimate
#   4. Opportunistic detour check
#   5. Plan path to selected goal
#   6. Rolling (receding-horizon) execution
# ==================================================================

def _mst_cost(positions):
    """
    Prim's algorithm MST on Euclidean distances.
    Returns the total weight of the MST spanning `positions`.
    Used as a lower-bound future-cost estimate for the remaining goals.

    Parameters
    ----------
    positions : list of (x, y[, z]) world-coordinate tuples
    Returns
    -------
    float  — total MST edge weight, or 0.0 if fewer than 2 nodes
    """
    if len(positions) < 2:
        return 0.0

    pts = [np.array(p[:2], dtype=float) for p in positions]
    n = len(pts)
    in_tree = [False] * n
    min_edge = [float('inf')] * n
    min_edge[0] = 0.0
    total = 0.0

    for _ in range(n):
        # Pick the cheapest vertex not yet in the MST
        u = min((i for i in range(n) if not in_tree[i]),
                key=lambda i: min_edge[i])
        in_tree[u] = True
        total += min_edge[u]
        # Relax edges from u
        for v in range(n):
            if not in_tree[v]:
                d = float(np.linalg.norm(pts[u] - pts[v]))
                if d < min_edge[v]:
                    min_edge[v] = d

    return total


class MultiGoalPlanner:
    """
    Drives a robot through an ordered set of goals in a dynamic environment.

    Wraps a ``DStarLite`` instance and implements the full pseudocode loop:

    WHILE remaining_goals is not empty:
        1. update_dynamic_map()
        2. build cost_table  — ShortestPathCost via D* Lite for each goal
        3. score every goal  : cost_to_goal + MST_cost(remaining - {goal})
        4. opportunistic check — redirect if a detour costs ≤ ε extra
        5. ComputeShortestPath to best_goal
        6. rolling execution  — move first_k steps, replan on obstacle change

    Parameters
    ----------
    dstar        : DStarLite — already initialised with static obstacles
    goal_set     : list of (x, y) world tuples
    rolling_k    : int — steps to execute before checking environment again
    epsilon      : float — tolerance for opportunistic detour (metres)
    """

    def __init__(self, dstar: DStarLite, goal_set: list,
                 rolling_k: int = 5, epsilon: float = 0.3):
        self.dstar     = dstar
        # remaining_goals is a mutable list; goals are removed when reached
        self.remaining_goals = list(goal_set)
        self.k         = rolling_k
        self.epsilon   = epsilon
        self.best_goal = None          # world-coordinate tuple (x, y)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _shortest_path_cost(self, from_world, to_world) -> float:
        """
        Estimate D* g-value cost from from_world to to_world.

        Strategy
        --------
        • Temporarily set goal = to_world and start = from_world,
          run compute_shortest_path, read g[start].
        • Restore original goal afterwards to avoid corrupting the
          planner state used for actual waypoint following.

        Falls back to Euclidean distance when the D* graph is
        disconnected (g = inf), so the planner always gets a usable
        number.
        """
        d = self.dstar

        # Save state
        prev_goal       = d.goal
        prev_start      = d.start
        prev_last_start = d.last_start
        prev_g          = d.g.copy()
        prev_rhs        = d.rhs.copy()
        prev_queue      = list(d.queue)
        prev_km         = d.km

        try:
            # Set up a fresh D* query: goal = to_world, start = from_world
            d.g   = np.full((d.nx, d.ny), float('inf'))
            d.rhs = np.full((d.nx, d.ny), float('inf'))
            d.km  = 0.0
            d.queue = []

            goal_cell  = d.world_to_grid(to_world[0],   to_world[1])
            start_cell = d.world_to_grid(from_world[0], from_world[1])

            d.goal  = goal_cell
            d.start = start_cell
            d.last_start = start_cell

            d.rhs[goal_cell] = 0.0
            import heapq as _hq
            _hq.heappush(d.queue, (d.calculate_key(goal_cell), goal_cell))
            d.compute_shortest_path()

            cost = float(d.g[start_cell])
            if cost == float('inf'):
                # Disconnected — fall back to Euclidean
                cost = float(np.linalg.norm(
                    np.array(from_world[:2]) - np.array(to_world[:2])))
        finally:
            # Restore original planner state unconditionally
            d.goal       = prev_goal
            d.start      = prev_start
            d.last_start = prev_last_start
            d.g          = prev_g
            d.rhs        = prev_rhs
            d.queue      = prev_queue
            d.km         = prev_km

        return cost

    def _extract_path(self, from_world, to_world, max_steps: int = 200):
        """
        Follow the D* g-gradient from from_world towards to_world and
        return up to max_steps world-coordinate waypoints.

        The path is extracted by greedily stepping to the
        lowest-cost neighbour (same as get_next_waypoint but returns
        the full sequence rather than just one step).
        """
        d = self.dstar
        curr = d.world_to_grid(from_world[0], from_world[1])
        goal = d.world_to_grid(to_world[0],   to_world[1])
        path = [curr]

        for _ in range(max_steps):
            if path[-1] == goal:
                break
            u = path[-1]
            best, best_cost = None, float('inf')
            for v in d.get_neighbors(u):
                c = d._edge_cost(u, v) + float(d.g[v])
                if c < best_cost:
                    best_cost, best = c, v
            if best is None or best in path:
                break
            path.append(best)

        # Convert grid cells → world coordinates
        return [d.grid_to_world(gx, gy) for gx, gy in path]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, robot_pos_world, bills_pos_world,
             obstacle_changed_fn=None):
        """
        Execute ONE iteration of the multi-goal planning loop.

        Call this from your main control loop on every planning tick.

        Parameters
        ----------
        robot_pos_world   : (x, y) — current robot world position
        bills_pos_world   : list of (x, y) — Bill positions this tick
        obstacle_changed_fn : callable() → bool, or None
            When provided, called after each micro-step; if it returns
            True the rolling execution aborts early and the loop
            replans next tick.

        Returns
        -------
        waypoints : list of (x, y) world positions
            The first ``k`` steps of the path to the chosen goal.
            Empty list when all goals have been visited.
        done      : bool — True when remaining_goals is empty
        """
        if not self.remaining_goals:
            return [], True

        # ----------------------------------------------------------------
        # Step 1 — Update dynamic map (Bills + robot position)
        # ----------------------------------------------------------------
        self.dstar.update_obstacles(robot_pos_world, bills_pos_world)

        # ----------------------------------------------------------------
        # Step 2 — Compute ShortestPathCost to every remaining goal
        # ----------------------------------------------------------------
        cost_table = {}
        for g in self.remaining_goals:
            cost_table[g] = self._shortest_path_cost(robot_pos_world, g)

        # ----------------------------------------------------------------
        # Step 3 — Goal-level evaluation  (cost + MST future estimate)
        # ----------------------------------------------------------------
        best_goal  = None
        best_score = float('inf')

        for g in self.remaining_goals:
            others          = [o for o in self.remaining_goals if o != g]
            future_estimate = _mst_cost(others)          # MST lower bound
            score           = cost_table[g] + future_estimate

            print(f"  [Goal eval] goal={g}  cost={cost_table[g]:.2f}"
                  f"  mst={future_estimate:.2f}  score={score:.2f}")

            if score < best_score:
                best_score = score
                best_goal  = g

        # ----------------------------------------------------------------
        # Step 4 — Opportunistic check
        # Redirect if visiting another goal en-route is nearly free.
        # ----------------------------------------------------------------
        for g in self.remaining_goals:
            if g == best_goal:
                continue

            cost_via_g = (
                cost_table[g]
                + self._shortest_path_cost(g, best_goal)
            )

            if cost_via_g <= cost_table[best_goal] + self.epsilon:
                print(f"  [Opportunistic] redirecting to {g} "
                      f"(via-cost={cost_via_g:.2f} vs "
                      f"direct={cost_table[best_goal]:.2f})")
                best_goal = g
                break

        self.best_goal = best_goal
        print(f"  [MultiGoal] → heading to {best_goal}")

        # ----------------------------------------------------------------
        # Step 5 — Plan path to selected goal via D* Lite
        # ----------------------------------------------------------------
        self.dstar.set_start(robot_pos_world)
        self.dstar.set_goal(best_goal)

        # ----------------------------------------------------------------
        # Step 6 — Rolling execution: extract first k steps
        # ----------------------------------------------------------------
        full_path = self._extract_path(robot_pos_world, best_goal)
        waypoints = full_path[1: self.k + 1]   # skip current position cell

        # Check opportunistically whether we've reached the goal
        # (caller is responsible for actually moving the robot)
        goal_reached = False
        if full_path:
            last_wp = full_path[-1]
            dist_to_goal = float(np.linalg.norm(
                np.array(last_wp[:2]) - np.array(best_goal[:2])))
            if dist_to_goal < self.dstar.res * 2:
                goal_reached = True

        if goal_reached:
            print(f"  [MultiGoal] ✓ reached {best_goal} — removing from queue")
            self.remaining_goals.remove(best_goal)
            self.best_goal = None

        done = len(self.remaining_goals) == 0
        return waypoints, done

    def is_done(self) -> bool:
        """True when all goals have been visited."""
        return len(self.remaining_goals) == 0
