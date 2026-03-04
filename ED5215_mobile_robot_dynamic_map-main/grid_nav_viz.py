#!/usr/bin/env python3
"""
Grid Navigation Visualizer — VIDEO EXPORT MODE
================================================
Saves nav_result.mp4 (or nav_result.gif) in the project folder.
No display window needed — runs fully headless.

Run:
    python grid_nav_viz.py
or in Spyder console:
    %run grid_nav_viz.py

PATH COLOURS
    GREEN  = cells already visited by the robot
    ORANGE = planned path ahead
"""

import matplotlib
matplotlib.use('Agg')   # headless — no display needed for video export

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import random
import time

from nav_planner import plan, solve_tsp


# ════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS
# ════════════════════════════════════════════════════════════════════════════

GRID_ROWS   = 22
GRID_COLS   = 28
ROBOT_START = (1, 1)

GOALS = [
    (2,  25),   # ① top-right
    (10, 13),   # ② centre
    (19,  2),   # ③ bottom-left
    (5,   7),   # ④ left
    (18, 24),   # ⑤ bottom-right
    (1,  20),   # ⑥ top-mid
    (12, 4),    # ⑦ mid-left (moved from (12,5) to avoid wall)
    (8,  22),   # ⑧ mid-right
]

ALGORITHM       = 'astar'   # 'astar' | 'dijkstra' | 'bfs'

NUM_DYN_OBS     = 5
OBS_MOVE_EVERY  = 12
OBS_SPREAD      = 1           # Reduced from 2 to allow squeezing past in tight corridors
COLLECT_DIST    = 1
REPLAN_ON_BLOCK = True

# ── Video settings ────────────────────────────────────────────────────────
VIDEO_FILE   = 'nav_result'   # output filename (no extension)
VIDEO_FPS    = 15             # frames per second
VIDEO_FRAMES = 1500           # Increased from 900 to ensure all 8 goals are reached


# ════════════════════════════════════════════════════════════════════════════
#  MAZE WALLS
# ════════════════════════════════════════════════════════════════════════════

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

# ════════════════════════════════════════════════════════════════════════════
#  COLOURS
# ════════════════════════════════════════════════════════════════════════════

C_BG        = '#0d0d1a'
C_WALL      = '#2e4057'
C_VISITED   = '#00b894'   # green
C_PLANNED   = '#e17055'   # orange
C_GOAL_OPEN = '#fdcb6e'
C_GOAL_DONE = '#55efc4'
C_OBS       = '#d63031'
C_ROBOT     = '#74b9ff'


# ════════════════════════════════════════════════════════════════════════════
#  MAZE BUILDER
# ════════════════════════════════════════════════════════════════════════════

def build_maze():
    maze = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int8)
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1
    for r1, r2, c1, c2 in INNER_WALLS:
        r1 = max(1, min(GRID_ROWS-2, r1));  r2 = max(1, min(GRID_ROWS-2, r2))
        c1 = max(1, min(GRID_COLS-2, c1));  c2 = max(1, min(GRID_COLS-2, c2))
        maze[r1:r2+1, c1:c2+1] = 1
    return maze


# ════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT
# ════════════════════════════════════════════════════════════════════════════

class GridNavEnv:
    def __init__(self):
        self.base  = build_maze()
        self.robot = list(ROBOT_START)
        self.goals = [g for g in GOALS if self.base[g[0], g[1]] == 0]
        # self.goals = solve_tsp(ROBOT_START, self.goals) or self.goals
        self.goal_idx  = 0
        self.collected = set()
        self.done      = False
        self.visited   = [tuple(self.robot)]
        self.planned   = []
        self.stats     = {'steps': 0, 'replans': 0}
        self.dyn_obs   = self._spawn_obs()
        self.frame_n   = 0
        self._replan()

    def _spawn_obs(self):
        safety = {tuple(self.robot)} | {tuple(g) for g in self.goals}
        obs, tries = [], 0
        while len(obs) < NUM_DYN_OBS and tries < 2000:
            r = random.randint(1, GRID_ROWS-2)
            c = random.randint(1, GRID_COLS-2)
            if self.base[r, c] == 0 and (r, c) not in safety:
                obs.append([r, c]); safety.add((r, c))
            tries += 1
        return obs

    def _inflate(self):
        cells = set()
        for o in self.dyn_obs:
            for dr in range(-OBS_SPREAD, OBS_SPREAD+1):
                for dc in range(-OBS_SPREAD, OBS_SPREAD+1):
                    nr, nc = o[0]+dr, o[1]+dc
                    if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
                        cells.add((nr, nc))
        return cells

    def _pgrid(self):
        g = self.base.copy()
        gc = tuple(self.goals[self.goal_idx]) if self.goal_idx < len(self.goals) else None
        rc = tuple(self.robot)
        for r, c in self._inflate():
            if (r, c) not in (rc, gc):
                g[r, c] = 1
        return g

    def _move_obs(self):
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        occupied = {(o[0], o[1]) for o in self.dyn_obs}
        rc = tuple(self.robot)
        for obs in self.dyn_obs:
            random.shuffle(dirs)
            for dr, dc in dirs:
                nr, nc = obs[0]+dr, obs[1]+dc
                if (1 <= nr < GRID_ROWS-1 and 1 <= nc < GRID_COLS-1
                        and self.base[nr, nc] == 0
                        and (nr, nc) not in occupied and (nr, nc) != rc):
                    occupied.discard((obs[0], obs[1]))
                    obs[0], obs[1] = nr, nc
                    occupied.add((nr, nc))
                    break

    def _advance_to_next_goal(self):
        """Move goal_idx to the next uncollected goal."""
        while self.goal_idx < len(self.goals) and self.goal_idx in self.collected:
            self.goal_idx += 1

    def _replan(self):
        # Advance past any already-collected goals first
        self._advance_to_next_goal()
        
        if self.goal_idx >= len(self.goals):
            # Double-check: any stragglers?
            uncollected = [i for i in range(len(self.goals)) if i not in self.collected]
            if not uncollected:
                self.planned = []; self.done = True
            else:
                self.goal_idx = uncollected[0]
                self._replan()
            return
        
        curr_goal = tuple(self.goals[self.goal_idx])
        
        # Are we already standing on (or adjacent to) the goal?
        rob_pos = tuple(self.robot)
        if abs(rob_pos[0]-curr_goal[0]) + abs(rob_pos[1]-curr_goal[1]) <= COLLECT_DIST:
            self.collected.add(self.goal_idx)
            self.goal_idx += 1
            self._replan()
            return

        path, _ = plan(self._pgrid(), rob_pos, curr_goal, ALGORITHM)
        
        if len(path) == 0:
            # Temporarily blocked by dynamic obstacles — wait one frame
            self.planned = []
        else:
            self.planned = list(path[1:]) if len(path) > 1 else []
        self.stats['replans'] += 1

    def step(self):
        if self.done: return
        self.frame_n += 1; self.stats['steps'] += 1
        
        # 1. Periodic physical environment update
        if self.frame_n % OBS_MOVE_EVERY == 0:
            self._move_obs()
            if REPLAN_ON_BLOCK and self.planned:
                if any(c in self._inflate() for c in self.planned[:4]):
                    self._replan()
        
        # 2. Replan if we have no path
        if not self.planned:
            self._replan()
        
        # 3. Movement execution
        if not self.planned: return
        
        nxt = self.planned.pop(0)
        self.robot = list(nxt)
        self.visited.append(nxt)
        
        # 4. Collection: Check if we are NEAR any uncollected goal
        rob_pos = tuple(self.robot)
        collected_anything = False
        for i, g in enumerate(self.goals):
            if i not in self.collected:
                if abs(rob_pos[0]-g[0]) + abs(rob_pos[1]-g[1]) <= COLLECT_DIST:
                    self.collected.add(i)
                    collected_anything = True

        if collected_anything:
            self.planned = []   # Trigger fresh replan for next goal
            self._replan()      # _replan handles all index advancement


# ════════════════════════════════════════════════════════════════════════════
#  VISUALIZER
# ════════════════════════════════════════════════════════════════════════════

class Visualizer:
    def __init__(self, env):
        self.env = env
        self.fig = plt.figure(figsize=(14, 9), facecolor=C_BG)

        gs = self.fig.add_gridspec(1, 2, width_ratios=[5, 1.3],
                                   left=0.02, right=0.98,
                                   top=0.93, bottom=0.06, wspace=0.04)
        self.ax   = self.fig.add_subplot(gs[0])
        self.ax_l = self.fig.add_subplot(gs[1])

        self.title = self.fig.suptitle(
            f'Grid Navigation  ·  {ALGORITHM.upper()}  '
            f'·  Goals: {len(env.goals)}  ·  Obstacles: {NUM_DYN_OBS}',
            color='white', fontsize=13, fontweight='bold', y=0.97)

        self._setup_grid()
        self._draw_walls()
        self._draw_legend()

        self.plan_patches = []
        self.vis_patches  = []

        self.obs_patches = [
            mpatches.FancyBboxPatch((0, 0), 0.82, 0.82,
                boxstyle='round,pad=0.06',
                facecolor=C_OBS, edgecolor='white',
                linewidth=0.5, alpha=0.9, zorder=5)
            for _ in env.dyn_obs]
        for p in self.obs_patches:
            self.ax.add_patch(p)

        self.robot_patch = plt.Circle(
            self._ctr(ROBOT_START), 0.38,
            facecolor=C_ROBOT, edgecolor='white',
            linewidth=2, zorder=8)
        self.ax.add_patch(self.robot_patch)

        self.goal_artists = []
        for i, (gr, gc) in enumerate(env.goals):
            cx, cy = self._ctr((gr, gc))
            s, = self.ax.plot(cx, cy, '*', color=C_GOAL_OPEN,
                              markersize=20, zorder=7,
                              markeredgecolor='white', markeredgewidth=0.5)
            t  = self.ax.text(cx, cy-0.55, str(i+1), color='white',
                              fontsize=7, ha='center', fontweight='bold', zorder=9)
            self.goal_artists.append((s, t))

        self.info = self.ax.text(
            0.01, 0.99, '', transform=self.ax.transAxes,
            color='white', fontsize=8, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='#1a1a3a',
                      edgecolor='#5566aa', alpha=0.9))

        self.prog_ax = self.fig.add_axes([0.03, 0.015, 0.70, 0.018])
        self.prog_ax.set_facecolor('#1c1c3a')
        self.prog_ax.set_xlim(0, len(env.goals)); self.prog_ax.set_ylim(0, 1)
        self.prog_ax.axis('off')
        self.pbar = self.prog_ax.barh(0.5, 0, height=1,
                                      color=C_VISITED, align='center')[0]
        self.prog_ax.text(len(env.goals)/2, 0.5, 'Goal Progress',
                          color='white', fontsize=7, ha='center',
                          va='center', alpha=0.4)

    def _ctr(self, rc):
        r, c = rc
        return c+0.5, (GRID_ROWS-1-r)+0.5

    def _org(self, rc):
        r, c = rc
        return c, GRID_ROWS-1-r

    def _setup_grid(self):
        ax = self.ax
        ax.set_facecolor(C_BG)
        ax.set_xlim(0, GRID_COLS); ax.set_ylim(0, GRID_ROWS)
        ax.set_aspect('equal')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in ax.spines.values(): sp.set_edgecolor('#334455')
        for i in range(GRID_COLS+1): ax.axvline(i, color='#1a2333', lw=0.3, zorder=0)
        for j in range(GRID_ROWS+1): ax.axhline(j, color='#1a2333', lw=0.3, zorder=0)
        self.ax_l.set_facecolor('#111122'); self.ax_l.axis('off')

    def _draw_walls(self):
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if self.env.base[r, c] == 1:
                    x, y = self._org((r, c))
                    self.ax.add_patch(mpatches.Rectangle(
                        (x, y), 1, 1, facecolor=C_WALL,
                        edgecolor='#0a0a18', linewidth=0.25, zorder=1))

    def _draw_legend(self):
        ax = self.ax_l
        ax.set_title('Legend', color='white', fontsize=9, fontweight='bold', pad=6)
        for i, (clr, lbl) in enumerate([
            (C_ROBOT,     'Robot'),
            (C_VISITED,   'Covered path (green)'),
            (C_PLANNED,   'Planned path (orange)'),
            (C_GOAL_OPEN, 'Goal — open'),
            (C_GOAL_DONE, 'Goal — collected'),
            (C_WALL,      'Wall'),
            (C_OBS,       'Dynamic obstacle'),
        ]):
            y = 0.88 - i*0.11
            ax.add_patch(mpatches.Rectangle((0.05, y), 0.18, 0.08,
                facecolor=clr, edgecolor='white', linewidth=0.5,
                transform=ax.transAxes))
            ax.text(0.30, y+0.04, lbl, color='white', fontsize=7.5,
                    va='center', transform=ax.transAxes)
        ax.text(0.50, 0.06, f'Algorithm\n{ALGORITHM.upper()}',
                color='#74b9ff', fontsize=9, ha='center', va='center',
                fontweight='bold', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='#1a2a4a',
                          edgecolor='#74b9ff', alpha=0.85))

    def update(self, frame_num):
        env = self.env
        env.step()

        # Planned path — orange, rebuild each frame
        for p in self.plan_patches: p.remove()
        self.plan_patches = []
        for cell in env.planned:
            x, y = self._org(cell)
            p = mpatches.Rectangle((x+0.1, y+0.1), 0.8, 0.8,
                facecolor=C_PLANNED, alpha=0.55, edgecolor='none', zorder=2)
            self.ax.add_patch(p)
            self.plan_patches.append(p)

        # Visited path — green, append only new cells
        while len(self.vis_patches) < len(env.visited):
            cell = env.visited[len(self.vis_patches)]
            x, y = self._org(cell)
            p = mpatches.Rectangle((x+0.05, y+0.05), 0.9, 0.9,
                facecolor=C_VISITED, alpha=0.45, edgecolor='none', zorder=2)
            self.ax.add_patch(p)
            self.vis_patches.append(p)

        # Robot
        rx, ry = self._ctr(env.robot)
        self.robot_patch.set_center((rx, ry))

        # Obstacles
        for i, obs in enumerate(env.dyn_obs):
            ox, oy = self._org(obs)
            self.obs_patches[i].set_x(ox+0.09)
            self.obs_patches[i].set_y(oy+0.09)

        # Goals
        for i, (star, _) in enumerate(self.goal_artists):
            star.set_color(C_GOAL_DONE if i in env.collected else C_GOAL_OPEN)
            star.set_markersize(14 if i in env.collected else 20)

        # Info
        nxt = (f"Goal {env.goal_idx+1}/{len(env.goals)}: {env.goals[env.goal_idx]}"
               if env.goal_idx < len(env.goals) else 'ALL DONE!')
        self.info.set_text(
            f"Step      : {env.stats['steps']}\n"
            f"Replans   : {env.stats['replans']}\n"
            f"Collected : {len(env.collected)}/{len(env.goals)}\n"
            f"Path len  : {len(env.planned)} cells\n"
            f"Target    : {nxt}")
        self.pbar.set_width(len(env.collected))

        if env.done:
            self.title.set_text(
                f'ALL {len(env.goals)} GOALS COLLECTED  '
                f'| {env.stats["steps"]} steps'
                f'| {env.stats["replans"]} replans')
            self.title.set_color('#00b894')


# ════════════════════════════════════════════════════════════════════════════
#  MAIN — saves video
# ════════════════════════════════════════════════════════════════════════════

def main():
    import os
    out_dir = os.path.dirname(os.path.abspath(__file__))
    mp4_path = os.path.join(out_dir, VIDEO_FILE + '.mp4')
    gif_path = os.path.join(out_dir, VIDEO_FILE + '.gif')

    print("═" * 58)
    print(f"  Grid Navigation Video Export")
    print(f"  Algorithm : {ALGORITHM.upper()}")
    print(f"  Grid      : {GRID_ROWS} × {GRID_COLS}")
    print(f"  Goals     : {len(GOALS)}   Obstacles: {NUM_DYN_OBS}")
    print(f"  Frames    : {VIDEO_FRAMES}  FPS: {VIDEO_FPS}"
          f"  → ~{VIDEO_FRAMES//VIDEO_FPS}s video")
    print("═" * 58)

    env = GridNavEnv()
    viz = Visualizer(env)

    anim = animation.FuncAnimation(
        viz.fig, viz.update,
        frames=VIDEO_FRAMES,
        interval=1000 // VIDEO_FPS,
        repeat=False)

    # ── Try MP4 first (imageio-ffmpeg bundled binary) ─────────────────────
    saved = False
    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        plt.rcParams['animation.ffmpeg_path'] = ffmpeg_bin
        writer = animation.FFMpegWriter(
            fps=VIDEO_FPS, bitrate=2500,
            extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        print(f"\n  Rendering MP4 → {mp4_path}")
        print(f"  Please wait ({VIDEO_FRAMES} frames) …\n")
        anim.save(mp4_path, writer=writer, dpi=130)
        print(f"\n  ✓  Saved: {mp4_path}")
        saved = True
    except Exception as e:
        print(f"  MP4 failed: {e}")

    # ── Fallback: animated GIF via Pillow ─────────────────────────────────
    if not saved:
        try:
            print(f"  Saving GIF → {gif_path}  (may be large) …")
            anim.save(gif_path,
                      writer=animation.PillowWriter(fps=VIDEO_FPS), dpi=100)
            print(f"  ✓  Saved: {gif_path}")
            saved = True
        except Exception as e:
            print(f"  GIF failed: {e}")

    if not saved:
        print("  Could not save — check imageio-ffmpeg and Pillow installs.")
        return

    print(f"\n  Steps    : {env.stats['steps']}")
    print(f"  Replans  : {env.stats['replans']}")
    print(f"  Collected: {len(env.collected)}/{len(env.goals)}")
    if saved:
        os.startfile(mp4_path if os.path.exists(mp4_path) else gif_path)


if __name__ == '__main__':
    main()
