import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import grid_nav_viz
from nav_planner import astar

def run_gen(algo, filename):
    print(f"--- Generating Video: {filename} ({algo}) ---")
    grid_nav_viz.ALGORITHM = algo
    grid_nav_viz.VIDEO_FILE = filename
    grid_nav_viz.VIDEO_FRAMES = 50 # Short test
    
    env = grid_nav_viz.GridNavEnv()
    viz = grid_nav_viz.Visualizer(env)
    
    anim = animation.FuncAnimation(
        viz.fig, viz.update,
        frames=grid_nav_viz.VIDEO_FRAMES,
        interval=50,
        repeat=False)

    out_path = filename + '.mp4'
    try:
        import imageio_ffmpeg
        plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = animation.FFMpegWriter(fps=10)
        anim.save(out_path, writer=writer)
        print(f"Successfully saved {out_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_gen('astar', 'test_astar_vid')
