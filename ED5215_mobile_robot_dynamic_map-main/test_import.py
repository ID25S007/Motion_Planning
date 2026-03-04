import traceback
try:
    print("Testing import grid_nav_viz...")
    import grid_nav_viz
    print("grid_nav_viz import successful")
except Exception as e:
    print("Caught exception during import:")
    traceback.print_exc()
