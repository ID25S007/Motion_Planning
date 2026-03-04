"""
gen_multigoal.py
================
Generates ONLY the Multi-Goal Planner video for quick isolated testing.
Run from Spyder:  %run gen_multigoal.py
"""

import os, sys, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import grid_nav_viz
from nav_planner import MultiGoalPlanner

SEED = 42
random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
class MultiGoalEnvAdapted(grid_nav_viz.GridNavEnv):
    """GridNavEnv driven by MultiGoalPlanner instead of sequential planners."""

    def __init__(self):
        # base __init__ calls _replan() → _pgrid() → needs mgp which isn't ready.
        # Guard: temporarily use 'astar' so base class doesn't crash.
        _prev = grid_nav_viz.ALGORITHM
        grid_nav_viz.ALGORITHM = 'astar'
        super().__init__()
        grid_nav_viz.ALGORITHM = _prev

        # NOW build mgp (safe — base __init__ is complete)
        self.mgp = MultiGoalPlanner(
            self.base.copy(), list(self.goals), epsilon=3.0, k=5)
        self.mgp.remaining_goals = set(tuple(g) for g in self.goals)
        self.waypoints = []
        self.planned   = []

    # ── Overrides ──────────────────────────────────────────────────────────
    def _pgrid(self):
        """Protect mgp.best_goal (not goal_idx) from obstacle inflation."""
        g  = self.base.copy()
        rc = tuple(self.robot)
        if hasattr(self, 'mgp') and self.mgp.best_goal is not None:
            gc = self.mgp.best_goal
        else:
            idx = getattr(self, '_goal_idx', 0)
            gc  = tuple(self.goals[idx]) if idx < len(self.goals) else None
        for r, c in self._inflate():
            if (r, c) not in (rc, gc):
                g[r, c] = 1
        return g

    @property
    def goal_idx(self):
        """Fall back to integer during init; read mgp.best_goal after."""
        if not hasattr(self, 'mgp') or self.mgp.best_goal is None:
            return getattr(self, '_goal_idx', 0)
        try:
            return [tuple(g) for g in self.goals].index(tuple(self.mgp.best_goal))
        except ValueError:
            return len(self.goals)

    @goal_idx.setter
    def goal_idx(self, v):
        self._goal_idx = v  # base class writes integers here during init

    # ── Simulation step ────────────────────────────────────────────────────
    def step(self):
        if self.done: return
        self.frame_n += 1
        self.stats['steps'] += 1

        # 1. Obstacle movement
        if self.frame_n % grid_nav_viz.OBS_MOVE_EVERY == 0:
            self._move_obs()
            if grid_nav_viz.REPLAN_ON_BLOCK and self.waypoints:
                if any(c in self._inflate() for c in self.waypoints[:4]):
                    self.waypoints = []

        # 2. Get next waypoints from planner
        if not self.waypoints:
            wps, full_path, done = self.mgp.step(tuple(self.robot), self._pgrid)
            self.waypoints = list(wps)
            self.planned   = full_path
            self.stats['replans'] += 1
            if done and not self.waypoints:
                self.done = True; return

        # 3. Pre-move collection check (stationary pickup)
        rob = tuple(self.robot)
        for g in list(self.mgp.remaining_goals):
            if abs(rob[0]-g[0]) + abs(rob[1]-g[1]) <= grid_nav_viz.COLLECT_DIST:
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
        rob  = tuple(self.robot)
        hit  = False
        for g in list(self.mgp.remaining_goals):
            if abs(rob[0]-g[0]) + abs(rob[1]-g[1]) <= grid_nav_viz.COLLECT_DIST:
                self.mgp.remaining_goals.discard(g)
                hit = True
                for i, og in enumerate(self.goals):
                    if tuple(og) == tuple(g):
                        self.collected.add(i); break
        if hit:
            self.waypoints = []
            if not self.mgp.remaining_goals:
                self.done = True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 58)
print("  Multi-Goal Planner Video Generator")
print("=" * 58)

grid_nav_viz.ALGORITHM = 'multi-goal'

try:
    env = MultiGoalEnvAdapted()
    print(f"  ✓ Env created  |  Goals: {len(env.goals)}  "
          f"  Obstacles: {grid_nav_viz.NUM_DYN_OBS}")
except Exception as e:
    print(f"  ✗ Env creation failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

viz = grid_nav_viz.Visualizer(env)
viz.title.set_text(
    f'Multi-Goal Planner (Strategic)  ·  '
    f'{grid_nav_viz.GRID_ROWS}×{grid_nav_viz.GRID_COLS}  ·  '
    f'Goals: {len(env.goals)}  ·  Obstacles: {grid_nav_viz.NUM_DYN_OBS}')

anim = animation.FuncAnimation(
    viz.fig, viz.update,
    frames=grid_nav_viz.VIDEO_FRAMES,
    interval=1000 // grid_nav_viz.VIDEO_FPS,
    repeat=False)

out_dir  = os.path.dirname(os.path.abspath(__file__))
mp4_path = os.path.join(out_dir, 'video_multigoal.mp4')
gif_path = os.path.join(out_dir, 'video_multigoal.gif')

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
    print(f"  MP4 failed ({e}) — trying GIF …")

if not saved:
    try:
        anim.save(gif_path,
                  writer=animation.PillowWriter(fps=grid_nav_viz.VIDEO_FPS),
                  dpi=100)
        print(f"  ✓  Saved: {gif_path}")
    except Exception as e:
        print(f"  GIF also failed: {e}")
        import traceback; traceback.print_exc()

print(f"\n  Stats | Steps: {env.stats['steps']}"
      f"  Replans: {env.stats['replans']}"
      f"  Collected: {len(env.collected)}/{len(env.goals)}")
plt.close(viz.fig)
print("  Done.")
