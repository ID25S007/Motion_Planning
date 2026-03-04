import sys
try:
    import numpy as np
    print("numpy OK")
except ImportError as e:
    print(f"numpy MISSING: {e}")

try:
    import matplotlib
    import matplotlib.pyplot as plt
    print("matplotlib OK")
except ImportError as e:
    print(f"matplotlib MISSING: {e}")

try:
    import grid_nav_viz
    print("grid_nav_viz OK")
except Exception as e:
    print(f"grid_nav_viz ERROR: {e}")

try:
    import nav_planner
    print("nav_planner OK")
except Exception as e:
    print(f"nav_planner ERROR: {e}")

try:
    import imageio_ffmpeg
    print("imageio_ffmpeg OK")
except ImportError:
    print("imageio_ffmpeg MISSING")
