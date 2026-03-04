"""
verify_all.py
=============
Headless simulation — no video, no display.
Runs A*, Dijkstra, BFS, D*, D*-Lite and the Multi-Goal Planner
on the SAME grid/goals and prints goal-collection statistics.

Run from Spyder with F5, or:
    python verify_all.py
"""

import random
import sys

# ── Force headless matplotlib before grid_nav_viz is imported ─────────────
import matplotlib
matplotlib.use('Agg')

import grid_nav_viz as gnv
from nav_planner import MultiGoalPlanner, plan

MAX_STEPS = 5000          # safety cap per run
SEED      = 42            # reproducible random obstacle movement

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def run_sequential(algo_name):
    """Run A*/Dijkstra/BFS/D*/D*Lite sequentially through all goals."""
    random.seed(SEED)
    gnv.ALGORITHM = algo_name
    env = gnv.GridNavEnv()

    for _ in range(MAX_STEPS):
        env.step()
        if env.done:
            break

    collected = len(env.collected)
    total     = len(env.goals)
    status    = "✓ ALL DONE" if collected == total else f"✗ STOPPED ({collected}/{total})"
    print(f"  [{algo_name:<12}] {status}  steps={env.stats['steps']}  replans={env.stats['replans']}")
    if collected < total:
        missing = [env.goals[i] for i in range(total) if i not in env.collected]
        print(f"               Missing goals: {missing}")
    return collected == total


def run_multigoal():
    """Run the Multi-Goal Planner."""
    random.seed(SEED)
    env = gnv.GridNavEnv()                     # base env for the grid/obs
    mgp = MultiGoalPlanner(env.base.copy(), list(env.goals), epsilon=3.0, k=5)
    mgp.current_position = tuple(env.robot)

    robot    = list(env.robot)
    visited  = set()
    waypoints = []
    collected = set()
    steps    = 0
    replans  = 0

    for _ in range(MAX_STEPS):
        steps += 1
        env.frame_n += 1
        if env.frame_n % gnv.OBS_MOVE_EVERY == 0:
            env._move_obs()

        if not waypoints:
            wps, _, done = mgp.step(tuple(robot), lambda: env._pgrid())
            waypoints = list(wps)
            replans  += 1
            if done and not waypoints:
                break

        if not waypoints:
            break

        nxt = waypoints.pop(0)
        robot = list(nxt)
        pos   = tuple(robot)
        visited.add(pos)

        # collect check
        changed = False
        for g in list(mgp.remaining_goals):
            if abs(pos[0]-g[0]) + abs(pos[1]-g[1]) <= gnv.COLLECT_DIST:
                mgp.remaining_goals.discard(g)
                collected.add(g)
                changed = True
        if changed:
            waypoints = []
        if not mgp.remaining_goals:
            break

    total  = len(env.goals)
    n_done = len(collected)
    status = "✓ ALL DONE" if n_done == total else f"✗ STOPPED ({n_done}/{total})"
    print(f"  [multi-goal  ] {status}  steps={steps}  replans={replans}")
    if n_done < total:
        missing = [g for g in env.goals if g not in collected]
        print(f"               Missing goals: {missing}")
    return n_done == total


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("  GOAL COLLECTION VERIFICATION")
    print(f"  Grid : {gnv.GRID_ROWS}×{gnv.GRID_COLS}   Goals: {len(gnv.GOALS)}")
    print(f"  Max steps per run: {MAX_STEPS}   Seed: {SEED}")
    print("=" * 60)

    results = {}
    for algo in ['astar', 'dijkstra', 'bfs', 'dstar', 'dstarlite']:
        results[algo] = run_sequential(algo)

    results['multi-goal'] = run_multigoal()

    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total  = len(results)
    print(f"\n  Summary: {passed}/{total} algorithms completed all goals\n")
    if passed == total:
        print("  ✓ ALL ALGORITHMS VERIFIED — every goal reached!\n")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  ✗ Issues found in: {failed}\n")

    sys.exit(0 if passed == total else 1)
