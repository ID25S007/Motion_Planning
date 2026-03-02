#!/usr/bin/env python

"""
Mobile robot — TSP priority queue + D* Lite (BFS heuristic) per goal.

Full algorithm
══════════════
PHASE 1 — PLANNING  (before simulation starts)
  a. Locate all 5 goal spheres.
  b. Run Held-Karp DP-TSP to find the GLOBALLY OPTIMAL visit order.
  c. Load that order into a MIN-HEAP priority queue keyed by visit index
     (index 0 = nearest/first goal floats to the top).

PHASE 2 — EXECUTION  (simulation running, repeat until queue empty)
  1. Pop the lowest-index goal from the priority queue.
  2. Reset D* Lite for the new goal (wall maps preserved):
        a. Clear start AND goal vicinities.
        b. Run BFS from the robot's START cell through FREE cells only
           → precompute navigable h_map (no walls, no Bills).
        c. D* uses h_map as its heuristic → path priority is guided by
           TRUE traversable distance, never straight-line through walls.
  3. Navigation loop:
        a. Every step: update Bills into D* map (dynamic obstacles).
        b. D* emits the next waypoint through FREE space (wall cost = inf).
        c. If path temporarily lost (Bills blocking), fall back to direct
           heading and keep moving.
        d. When robot is within COLLECT_RADIUS of sphere → hide it → pop next.
"""

import heapq
import time
import os
import numpy as np
import matplotlib
try:
    matplotlib.use('Qt5Agg')   # Spyder live plot window
except Exception:
    matplotlib.use('Agg')      # fallback: save-only (no pop-up)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import planner
import sim_interface
import control

# ── Arena ─────────────────────────────────────────────────────────────────────
ARENA_X_MIN = -1.0
ARENA_X_MAX = 11.0
ARENA_Y_MIN = -1.0
ARENA_Y_MAX = 11.0
GRID_RES    =  0.2          # 20 cm / cell  → 60 × 60 grid

# ── Tuning ────────────────────────────────────────────────────────────────────
COLLECT_RADIUS = 0.7        # m  — sphere hidden when robot passes this close
BORDER_MARGIN  = 0.5        # m  — last-resort software stop (D* boundary handles main avoidance)
STEP_DELAY     = 0.06       # s  — between control steps
MAX_NAV_STEPS  = 600        # max steps allowed per individual goal


# ── Helpers ───────────────────────────────────────────────────────────────────
def dist2d(a, b):
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def clear_vicinity(dstar, world_pos, label=""):
    """
    Unblock ONLY the exact grid cell at world_pos so D* can use it as a
    start or goal.  Neighbours are deliberately left as-is:
      • If a neighbour is a wall cell (base_map=inf), it stays blocked and
        the proximity penalty in _edge_cost still fires correctly — this
        keeps paths away from wall faces even after clearing the start cell.
      • Cleaning neighbours was previously erasing wall-proximity records,
        causing D* to route right alongside walls with zero penalty.
    Returns True if the cell was actually blocked and needed clearing.
    """
    cell = dstar.world_to_grid(world_pos[0], world_pos[1])
    if dstar.map[cell] == float('inf'):
        dstar.base_map[cell] = 1.0
        dstar.map[cell]      = 1.0
        dstar.update_vertex(cell)
        # Also update neighbours so their rhs values reflect the newly free cell
        for nb in dstar.get_neighbors(cell):
            dstar.update_vertex(nb)
        dstar.compute_shortest_path()
        if label:
            print(f"    [clear] Unblocked {label} cell {cell}")
        return True
    return False


def prepare_dstar_for_goal(dstar, goal_world, robot_pos):
    """
    Re-initialise D* Lite for a brand-new goal:
      1. Wipe planning arrays (keep obstacle maps intact).
      2. Set start → clear start vicinity → set goal → clear goal vicinity.
      3. BFS-precompute h_map from robot start through FREE cells.
         From this point the D* heuristic reflects TRUE navigable distance,
         not straight-line Euclidean through walls.
      4. Final compute_shortest_path().
    Returns True when a finite-cost path exists.
    """
    # ── 1. Wipe D* planning state ────────────────────────────────────────────
    dstar.rhs[:] = float('inf')
    dstar.g[:]   = float('inf')
    dstar.km     = 0.0
    dstar.queue  = []

    # ── 2. Set start, clear vicinity ─────────────────────────────────────────
    dstar.set_start(robot_pos)
    dstar.last_start = dstar.start
    clear_vicinity(dstar, robot_pos, label="start")

    # ── 3. Set goal, seed queue ───────────────────────────────────────────────
    dstar.set_goal(goal_world)
    clear_vicinity(dstar, goal_world, label="goal")

    # ── 4. BFS h_map: navigable distances from start through free space ───────
    #   h_map[cell] = exact shortest path length (m) from cell to robot start
    #   travelling ONLY through cells where map != inf (free space).
    #   D* heuristic uses this instead of straight-line Euclidean so the
    #   priority queue correctly ranks cells by traversable, wall-aware cost.
    dstar.precompute_h_map()

    # ── 5. Final path computation with updated heuristic ─────────────────────
    dstar.compute_shortest_path()

    reachable = float(dstar.g[dstar.start]) < float('inf')
    print(f"    [D*] {'Path found ✓' if reachable else 'No path — direct-heading fallback active'}")
    return reachable


# ── Results Plot ──────────────────────────────────────────────────────────────
def plot_results(traj_x, traj_y, bill_tx, bill_ty, bill2x, bill2y,
                 spheres, collection_events, goal_segments,
                 static_obstacles, n_collected, n_total):
    """
    Draw a full mission summary figure and save it as mission_result.png.
    Shows:
      • Robot trajectory coloured by sub-goal segment
      • Bills movement paths (dotted)
      • Sphere start positions (green stars)
      • Collection events  (gold numbered stars)
      • Static wall rectangles
      • Arena boundary
      • Mission stats text box
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    # ── Arena boundary ────────────────────────────────────────────────────────
    ax.add_patch(mpatches.Rectangle(
        (ARENA_X_MIN, ARENA_Y_MIN),
        ARENA_X_MAX - ARENA_X_MIN, ARENA_Y_MAX - ARENA_Y_MIN,
        linewidth=2, edgecolor='#00d4ff', facecolor='none', zorder=1))

    # ── Static walls ──────────────────────────────────────────────────────────
    for w in static_obstacles:
        wx, wy   = w['pos'][0], w['pos'][1]
        wsx, wsy = w['size'][0], w['size'][1]
        ax.add_patch(mpatches.Rectangle(
            (wx - wsx/2, wy - wsy/2), wsx, wsy,
            linewidth=0, facecolor='#4a4a6a', alpha=0.8, zorder=2))

    # ── Bills paths ───────────────────────────────────────────────────────────
    if bill_tx:
        ax.plot(bill_tx, bill_ty, ':', color='#ff6b6b', lw=1,
                alpha=0.5, label='Bill 1 path', zorder=3)
    if bill2x:
        ax.plot(bill2x, bill2y, ':', color='#ff9f43', lw=1,
                alpha=0.5, label='Bill 2 path', zorder=3)

    # ── Robot trajectory — one colour per sub-goal ────────────────────────────
    seg_colors = plt.cm.plasma(np.linspace(0.1, 0.9, max(len(goal_segments), 1)))
    for idx, (name, seg) in enumerate(goal_segments):
        if not seg:
            continue
        xs = [p[0] for p in seg]
        ys = [p[1] for p in seg]
        ax.plot(xs, ys, '-', color=seg_colors[idx], lw=1.8,
                alpha=0.9, label=f'→ {name}', zorder=4)

    # Full path faint underlay
    if traj_x:
        ax.plot(traj_x, traj_y, '-', color='white', lw=0.5,
                alpha=0.15, zorder=3)

    # ── Sphere start positions ────────────────────────────────────────────────
    for pos, _, name in spheres:
        ax.plot(pos[0], pos[1], '*', color='#2ecc71', markersize=14,
                zorder=6, markeredgecolor='white', markeredgewidth=0.5)
        ax.text(pos[0] + 0.15, pos[1] + 0.15, name.replace('/Sphere', 'S'),
                color='#2ecc71', fontsize=7, zorder=7)

    # ── Collection events — numbered gold stars ───────────────────────────────
    for ex, ey, ename, enum in collection_events:
        ax.plot(ex, ey, '*', color='#f9ca24', markersize=18,
                zorder=8, markeredgecolor='white', markeredgewidth=0.8)
        ax.text(ex, ey + 0.35, str(enum), color='white', fontsize=8,
                ha='center', fontweight='bold', zorder=9)

    # ── Start marker ─────────────────────────────────────────────────────────
    if traj_x:
        ax.plot(traj_x[0], traj_y[0], 'o', color='#00d4ff', markersize=10,
                zorder=9, label='Start', markeredgecolor='white')
        ax.plot(traj_x[-1], traj_y[-1], 's', color='#fd79a8', markersize=10,
                zorder=9, label='End', markeredgecolor='white')

    # ── Labels and legend ─────────────────────────────────────────────────────
    ax.set_xlim(ARENA_X_MIN - 0.3, ARENA_X_MAX + 0.3)
    ax.set_ylim(ARENA_Y_MIN - 0.3, ARENA_Y_MAX + 0.3)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', color='white', fontsize=11)
    ax.set_ylabel('Y (m)', color='white', fontsize=11)
    ax.set_title(f'Robot Mission Results — {n_collected}/{n_total} Spheres Collected',
                 color='white', fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')

    legend = ax.legend(loc='upper right', fontsize=7,
                       facecolor='#2d2d4e', edgecolor='#555577',
                       labelcolor='white', framealpha=0.9)

    # ── Stats text box ────────────────────────────────────────────────────────
    stats = (f"Collected : {n_collected}/{n_total}\n"
             f"Path pts  : {len(traj_x)}\n"
             f"Algorithm : TSP + D* Lite + BFS-h")
    ax.text(ARENA_X_MIN + 0.1, ARENA_Y_MAX - 0.3, stats,
            color='white', fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#2d2d4e',
                      edgecolor='#00d4ff', alpha=0.85))

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'mission_result.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\n  [Plot saved → {out_path}]")
    try:
        plt.show()                   # works if Qt5Agg/TkAgg backend is active
    except Exception:
        pass
    # Also open the PNG in the default Windows viewer
    try:
        os.startfile(out_path)
    except Exception:
        print("  [Open the PNG manually to view results]")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── Connect & load scene ──────────────────────────────────────────────────
    if not sim_interface.sim_init():
        print("Failed connecting to remote API server")
        return

    sim_interface.load_scene(os.path.abspath('my_scene.ttt'))
    sim_interface.get_handles()

    # ── Discover goal spheres ─────────────────────────────────────────────────
    goal_names   = ['/Sphere[0]', '/Sphere[1]', '/Sphere[2]',
                    '/Sphere[3]', '/Sphere[4]']
    goal_handles = [sim_interface.get_handle(n) for n in goal_names]

    spheres = []                   # (pos, handle, name)
    for i, handle in enumerate(goal_handles):
        if handle is None:
            continue
        pos = sim_interface.localize_object(handle)
        if pos is not None:
            spheres.append((pos, handle, goal_names[i]))

    if not spheres:
        print("ERROR: No goal spheres found!")
        sim_interface.sim_shutdown()
        return

    print(f"\nFound {len(spheres)} spheres:")
    for pos, _, name in spheres:
        print(f"  {name}  at ({pos[0]:.2f}, {pos[1]:.2f})")

    # ── PHASE 1 : Held-Karp DP-TSP → optimal visit order ─────────────────────
    robot_pre_sim  = sim_interface.localize_robot()
    goal_positions = [pos for pos, _, _ in spheres]

    print("\nRunning Held-Karp DP-TSP to find optimal visit order …")
    tsp_ordered_positions = planner.solve_tsp(robot_pre_sim, goal_positions)

    # Map TSP positions back to (pos, handle, name)
    ordered_spheres = []
    for tsp_pos in tsp_ordered_positions:
        for pos, handle, name in spheres:
            if np.linalg.norm(np.array(tsp_pos) - np.array(pos)) < 0.01:
                ordered_spheres.append((pos, handle, name))
                break

    print("Optimal TSP visit order:")
    for i, (pos, _, name) in enumerate(ordered_spheres):
        print(f"  {i+1}. {name}  at ({pos[0]:.2f}, {pos[1]:.2f})")

    # ── Load TSP order into MIN-HEAP priority queue ───────────────────────────
    #
    #   Each entry: (priority_index, sphere_index, pos, handle, name)
    #   priority_index = TSP visit order (0 = first goal → pops first).
    #   Using heapq guarantees O(log n) insert/pop and the minimum index
    #   (= next TSP goal) always surfaces at the top.
    #
    goal_queue = []          # min-heap
    _counter   = 0           # unique tiebreaker — prevents comparing pos tuples
    for i, (pos, handle, name) in enumerate(ordered_spheres):
        heapq.heappush(goal_queue, (i, _counter, pos, handle, name))
        _counter += 1

    print(f"\nPriority queue loaded ({len(goal_queue)} goals):")
    for entry in sorted(goal_queue):
        pri, _, pos, _, name = entry
        print(f"  priority {pri}: {name}")

    # ── Start simulation ───────────────────────────────────────────────────────
    if not sim_interface.start_simulation():
        print("Failed to start simulation")
        sim_interface.sim_shutdown()
        return

    sim_interface.freeze_goal_spheres([h for _, h, _ in spheres])

    # ── Build ONE D* planner — walls registered once, kept for all goals ───────
    static_obstacles = sim_interface.get_static_obstacles()

    dstar = planner.DStarLite(
        x_min=ARENA_X_MIN, x_max=ARENA_X_MAX,
        y_min=ARENA_Y_MIN, y_max=ARENA_Y_MAX,
        resolution=GRID_RES,
    )

    robot_state = sim_interface.localize_robot()
    dstar.set_start(robot_state)
    dstar.last_start = dstar.start

    # Permanent boundary walls
    # thickness=4 cells = 0.8 m from each arena edge.
    # This matches the robot body clearance so D* NEVER plans a path
    # that approaches the physical boundary — no software guard needed.
    dstar.add_boundary_walls(thickness=4)

    # Permanent internal walls (each inflated by safety margin)
    for wall in static_obstacles:
        dstar.add_static_obstacle(
            wall['pos'][0], wall['pos'][1],
            wall['size'][0], wall['size'][1],
            wall['orientation'][2],
        )

    # ── PHASE 2 : Pop from queue → D* with BFS heuristic → collect → repeat ──
    total             = len(goal_queue)
    goal_n            = 0
    collected_handles = set()

    # ── Result tracking (for the end-of-run plot) ─────────────────────────────
    traj_x,  traj_y  = [], []          # robot path
    bill_tx, bill_ty = [], []          # Bill 1 path
    bill2x,  bill2y  = [], []          # Bill 2 path
    collection_events = []             # (x, y, name, goal_number)
    goal_segments     = []             # [(name, [(x,y)...]), ...] one per sub-goal

    while goal_queue:
        # ── Pop lowest-priority (= next TSP goal) from heap ──────────────────
        priority, _, goal_pos, goal_handle, goal_name = heapq.heappop(goal_queue)

        # Skip if already collected en-route
        if goal_handle in collected_handles:
            print(f"\n  (Skipping {goal_name} — already collected en-route)")
            continue

        goal_n += 1
        robot_state = sim_interface.localize_robot()
        if robot_state is None:
            print("ERROR: Lost robot position — aborting.")
            break

        remaining = [(n, p) for pri, _, p, h, n in sorted(goal_queue)
                     if h not in collected_handles]

        print(f"\n{'═'*64}")
        print(f"  GOAL {goal_n}/{total} [priority={priority}]: {goal_name}")
        print(f"    Position  : ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")
        print(f"    Robot at  : ({robot_state[0]:.2f}, {robot_state[1]:.2f})")
        print(f"    Distance  : {dist2d(robot_state, goal_pos):.2f} m")
        print(f"    Queue remaining: {[n for n,_ in remaining] if remaining else '(empty)'}")
        print(f"{'═'*64}")

        # ── D* reset with BFS heuristic ───────────────────────────────────────
        has_path = prepare_dstar_for_goal(dstar, goal_pos, robot_state)

        # Reset PID integrator — prevents heading-error windup from previous goal
        control.reset_pid()

        # ── Navigation loop ────────────────────────────────────────────────────
        reached  = False
        nav_step = 0
        seg_xy   = []                  # track this sub-goal's path segment

        while nav_step < MAX_NAV_STEPS:
            nav_step += 1

            robot_state = sim_interface.localize_robot()
            if robot_state is None:
                print("ERROR: Lost robot position!")
                break

            # ── Record robot position ─────────────────────────────────────────
            traj_x.append(robot_state[0])
            traj_y.append(robot_state[1])
            seg_xy.append((robot_state[0], robot_state[1]))

            # ── Collect current target on proximity ───────────────────────────
            if dist2d(robot_state, goal_pos) <= COLLECT_RADIUS:
                sim_interface.setvel_pioneers(0.0, 0.0)
                sim_interface.hide_goal_sphere(goal_handle)
                collected_handles.add(goal_handle)
                collection_events.append((robot_state[0], robot_state[1],
                                          goal_name, goal_n))
                print(f"\n  ✓ {goal_name} collected "
                      f"({len(collected_handles)}/{total} done, step {nav_step})")
                reached = True
                break

            # ── Opportunistic collection of other spheres in range ─────────────
            for entry in list(goal_queue):
                pri2, idx2, pos2, handle2, name2 = entry
                if handle2 not in collected_handles and \
                   dist2d(robot_state, pos2) <= COLLECT_RADIUS:
                    sim_interface.hide_goal_sphere(handle2)
                    collected_handles.add(handle2)
                    print(f"  ✓ Bonus: {name2} collected en-route!")

            # ── Boundary guard: active push-back toward arena interior ────────
            near_edge = (
                robot_state[0] < ARENA_X_MIN + BORDER_MARGIN or
                robot_state[0] > ARENA_X_MAX - BORDER_MARGIN or
                robot_state[1] < ARENA_Y_MIN + BORDER_MARGIN or
                robot_state[1] > ARENA_Y_MAX - BORDER_MARGIN
            )
            if near_edge:
                # Compute a push-back waypoint toward the arena centre.
                # Clamp the robot's current position to a safe interior point
                # and drive there until the robot is safely inside.
                arena_cx = (ARENA_X_MIN + ARENA_X_MAX) / 2.0
                arena_cy = (ARENA_Y_MIN + ARENA_Y_MAX) / 2.0
                safe_x = max(ARENA_X_MIN + BORDER_MARGIN + 0.4,
                             min(ARENA_X_MAX - BORDER_MARGIN - 0.4,
                                 robot_state[0]))
                safe_y = max(ARENA_Y_MIN + BORDER_MARGIN + 0.4,
                             min(ARENA_Y_MAX - BORDER_MARGIN - 0.4,
                                 robot_state[1]))
                # Blend: 30% toward safe clamp, 70% toward arena centre
                push_x = 0.3 * safe_x + 0.7 * arena_cx
                push_y = 0.3 * safe_y + 0.7 * arena_cy
                print(f"  !! Near boundary ({robot_state[0]:.2f},{robot_state[1]:.2f})"
                      f" → pushing to ({push_x:.2f},{push_y:.2f})")
                V, W = control.gtg(robot_state, (push_x, push_y))
                sim_interface.setvel_pioneers(V, W)
                time.sleep(STEP_DELAY)
                continue

            # ── Update Bills (dynamic obstacles) ──────────────────────────────
            bills_pos = sim_interface.localize_bills()
            dstar.update_obstacles(robot_state, bills_pos)

            # Record Bill positions
            if bills_pos:
                if len(bills_pos) >= 1:
                    bill_tx.append(bills_pos[0][0]); bill_ty.append(bills_pos[0][1])
                if len(bills_pos) >= 2:
                    bill2x.append(bills_pos[1][0]); bill2y.append(bills_pos[1][1])

            # ── Waypoint: D* (through free space) or direct fallback ──────────
            has_path = float(dstar.g[dstar.start]) < float('inf')
            if has_path:
                waypoint = dstar.get_next_waypoint(robot_state)
            else:
                waypoint = (goal_pos[0], goal_pos[1])   # direct heading

            # ── Drive ─────────────────────────────────────────────────────────
            V, W = control.gtg(robot_state, waypoint)
            heading_err = (np.arctan2(waypoint[1] - robot_state[1],
                                      waypoint[0] - robot_state[0])
                           - robot_state[2])
            heading_err = ((heading_err + np.pi) % (2 * np.pi)) - np.pi

            print(f"  [{nav_step:3d}] "
                  f"Pos:({robot_state[0]:.2f},{robot_state[1]:.2f})  "
                  f"Wp:({waypoint[0]:.2f},{waypoint[1]:.2f})  "
                  f"Dist:{dist2d(robot_state, goal_pos):.2f}m  "
                  f"Err:{np.degrees(heading_err):.1f}°"
                  + ("  [no-path→direct]" if not has_path else ""))

            sim_interface.setvel_pioneers(V, W)
            time.sleep(STEP_DELAY)

        if not reached:
            sim_interface.setvel_pioneers(0.0, 0.0)
            print(f"\n  ✗ Could not reach {goal_name} in {MAX_NAV_STEPS} steps.")

        goal_segments.append((goal_name, seg_xy))

    # ── Mission complete ───────────────────────────────────────────────────────
    sim_interface.setvel_pioneers(0.0, 0.0)
    n = len(collected_handles)
    print(f"\n{'═'*64}")
    print(f"  MISSION COMPLETE — {n}/{total} spheres collected")
    print(f"{'═'*64}\n")
    sim_interface.sim_shutdown()
    time.sleep(2.0)

    # ── Plot results ───────────────────────────────────────────────────────────
    plot_results(
        traj_x, traj_y,
        bill_tx, bill_ty,
        bill2x,  bill2y,
        spheres, collection_events,
        goal_segments,
        static_obstacles,
        n, total
    )


if __name__ == '__main__':
    main()
    print('Program ended')