"""
generate_all_videos.py
======================
Generates comparison videos for ALL algorithms on the same 8-goal mission:
  1. A*
  2. Dijkstra
  3. BFS
  4. D*
  5. D* Lite
  6. Multi-Goal Planner (Strategic)

Run from Spyder:   %run generate_all_videos.py
Or from console:   python generate_all_videos.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import grid_nav_viz
from nav_planner import MultiGoalPlanner

# ═══════════════════════════════════════════════════════════════════════════
#  MULTI-GOAL ADAPTED ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════

class MultiGoalEnvAdapted(grid_nav_viz.GridNavEnv):
    """
    Wraps GridNavEnv to drive the MultiGoalPlanner instead of the
    sequential A* / D* / BFS used by the base class.
    """
    def __init__(self):
        # The base class __init__ calls _replan() which uses ALGORITHM.
        # 'multi-goal' would hit the NotImplementedError stub, so we
        # temporarily switch to 'astar' just for the initial path.
        _prev_algo = grid_nav_viz.ALGORITHM
        grid_nav_viz.ALGORITHM = 'astar'
        super().__init__()
        grid_nav_viz.ALGORITHM = _prev_algo

        # Build planner on the same goals/grid as the base env
        self.mgp = MultiGoalPlanner(
            self.base, list(self.goals), epsilon=3.0, k=5)
        self.mgp.remaining_goals = set(tuple(g) for g in self.goals)
        self.waypoints = []
        self.planned   = []   # reset — MGP will fill this

    def _pgrid(self):
        """Override: keep best_goal (and robot) free from obstacle inflation.
        Falls back to base class behaviour during super().__init__() when mgp
        doesn't exist yet."""
        g  = self.base.copy()
        rc = tuple(self.robot)
        # Determine which goal cell to protect from obstacle inflation
        if hasattr(self, 'mgp') and self.mgp.best_goal is not None:
            gc = self.mgp.best_goal
        else:
            # mgp not ready yet — use the same logic as the base class
            idx = getattr(self, '_goal_idx', 0)
            gc = tuple(self.goals[idx]) if idx < len(self.goals) else None
        for r, c in self._inflate():
            if (r, c) not in (rc, gc):
                g[r, c] = 1
        return g

    @property
    def goal_idx(self):
        """
        During super().__init__() mgp doesn't exist yet — fall back to the
        plain integer (_goal_idx).  Once mgp is set, reflect its best_goal.
        """
        if not hasattr(self, 'mgp') or self.mgp.best_goal is None:
            return getattr(self, '_goal_idx', 0)
        try:
            return [tuple(g) for g in self.goals].index(tuple(self.mgp.best_goal))
        except ValueError:
            return len(self.goals)   # best_goal already collected

    @goal_idx.setter
    def goal_idx(self, v):
        """Base class writes goal_idx as an integer during init — store it."""
        self._goal_idx = v

    def step(self):
        if self.done: return
        self.frame_n += 1
        self.stats['steps'] += 1

        # 1. Periodic obstacle movement
        if self.frame_n % grid_nav_viz.OBS_MOVE_EVERY == 0:
            self._move_obs()
            if grid_nav_viz.REPLAN_ON_BLOCK and self.waypoints:
                if any(c in self._inflate() for c in self.waypoints[:4]):
                    self.waypoints = []

        # 2. Get new waypoints from Multi-Goal Planner if needed
        if not self.waypoints:
            wps, full_path, planner_done = self.mgp.step(
                tuple(self.robot), self._pgrid)
            self.waypoints = list(wps)
            self.planned   = full_path         # orange path visualisation
            self.stats['replans'] += 1
            if planner_done and not self.waypoints:
                self.done = True; return

        # 3. (Stationary) collection check — robot hasn't moved yet
        rob_pos = tuple(self.robot)
        for g in list(self.mgp.remaining_goals):
            if abs(rob_pos[0]-g[0]) + abs(rob_pos[1]-g[1]) <= grid_nav_viz.COLLECT_DIST:
                self.mgp.remaining_goals.discard(g)
                for i, og in enumerate(self.goals):
                    if tuple(og) == tuple(g):
                        self.collected.add(i); break
        if not self.mgp.remaining_goals:
            self.done = True; return

        if not self.waypoints: return

        # 4. Move one step
        nxt = self.waypoints.pop(0)
        self.robot = list(nxt)
        self.visited.append(tuple(nxt))

        # 5. Post-move collection check
        rob_pos = tuple(self.robot)
        hit = False
        for g in list(self.mgp.remaining_goals):
            if abs(rob_pos[0]-g[0]) + abs(rob_pos[1]-g[1]) <= grid_nav_viz.COLLECT_DIST:
                self.mgp.remaining_goals.discard(g)
                hit = True
                for i, og in enumerate(self.goals):
                    if tuple(og) == tuple(g):
                        self.collected.add(i); break

        if hit:
            self.waypoints = []      # force replan to choose next goal
            if not self.mgp.remaining_goals:
                self.done = True


# ═══════════════════════════════════════════════════════════════════════════
#  VIDEO GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

def run_gen(algo_name, display_name, filename, env_class=grid_nav_viz.GridNavEnv):
    """
    Create and save one animation video for the given algorithm.

    Parameters
    ----------
    algo_name    : str   algorithm key used by grid_nav_viz  (e.g. 'astar')
    display_name : str   human-readable label shown in video title
    filename     : str   output filename without extension    (e.g. 'video_astar')
    env_class    : class environment class to instantiate
    """
    print(f"\n{'─'*58}")
    print(f"  Generating: {display_name}  →  {filename}.mp4")
    print(f"{'─'*58}")

    # Tell the base environment which algorithm to use
    grid_nav_viz.ALGORITHM = algo_name

    # Build environment + visualiser
    env = env_class()
    viz = grid_nav_viz.Visualizer(env)
    viz.title.set_text(
        f'{display_name}  ·  {grid_nav_viz.GRID_ROWS}×{grid_nav_viz.GRID_COLS}'
        f'  ·  Goals: {len(env.goals)}  ·  Obstacles: {grid_nav_viz.NUM_DYN_OBS}')

    # Build animation
    anim = animation.FuncAnimation(
        viz.fig, viz.update,
        frames=grid_nav_viz.VIDEO_FRAMES,
        interval=1000 // grid_nav_viz.VIDEO_FPS,
        repeat=False)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    mp4_path = os.path.join(out_dir, filename + '.mp4')
    gif_path = os.path.join(out_dir, filename + '.gif')

    # Try MP4 first (needs imageio_ffmpeg)
    saved = False
    try:
        import imageio_ffmpeg
        plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = animation.FFMpegWriter(
            fps=grid_nav_viz.VIDEO_FPS, bitrate=2000,
            extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        print(f"  Rendering {grid_nav_viz.VIDEO_FRAMES} frames …", flush=True)
        anim.save(mp4_path, writer=writer, dpi=120)
        print(f"  ✓  Saved: {mp4_path}")
        saved = True
    except Exception as e:
        print(f"  MP4 failed ({e}). Trying GIF …")

    # Fallback: animated GIF
    if not saved:
        try:
            anim.save(gif_path,
                      writer=animation.PillowWriter(
                          fps=grid_nav_viz.VIDEO_FPS), dpi=100)
            print(f"  ✓  Saved: {gif_path}")
            saved = True
        except Exception as e:
            print(f"  GIF also failed: {e}")

    # Print final stats
    print(f"  Stats | Steps: {env.stats['steps']}"
          f"  Replans: {env.stats['replans']}"
          f"  Collected: {len(env.collected)}/{len(env.goals)}")

    plt.close(viz.fig)   # free memory before next video
    return saved


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import random
    SEED = 42
    random.seed(SEED)

    print("=" * 58)
    print("  ALL-ALGORITHM VIDEO GENERATOR")
    print(f"  Grid: {grid_nav_viz.GRID_ROWS}×{grid_nav_viz.GRID_COLS}"
          f"   Goals: {len(grid_nav_viz.GOALS)}"
          f"   Frames: {grid_nav_viz.VIDEO_FRAMES}"
          f"   FPS: {grid_nav_viz.VIDEO_FPS}")
    print("=" * 58)

    VIDEOS = [
        # (algo_key,      display_name,         output_file)
        ('astar',      'A*  (Informed Optimal)',  'video_astar'),
        ('dijkstra',   'Dijkstra (Cost Optimal)', 'video_dijkstra'),
        ('bfs',        'BFS (Shortest Hops)',      'video_bfs'),
        ('dstar',      'D*  (Incremental)',        'video_dstar'),
        ('dstarlite',  'D* Lite (Incremental)',    'video_dstarlite'),
    ]

    results = {}
    for algo, label, fname in VIDEOS:
        ok = run_gen(algo, label, fname)
        results[label] = ok

    # Multi-Goal Planner (uses its own env class)
    ok = run_gen('multi-goal', 'Multi-Goal Planner (Strategic)',
                 'video_multigoal', env_class=MultiGoalEnvAdapted)
    results['Multi-Goal Planner'] = ok

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  GENERATION SUMMARY")
    print("=" * 58)
    for label, ok in results.items():
        print(f"  {'✓' if ok else '✗'}  {label}")
    print("=" * 58)
    n_ok = sum(results.values())
    print(f"\n  {n_ok}/{len(results)} videos saved successfully.\n")
