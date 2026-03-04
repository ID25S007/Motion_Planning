import numpy as np
import robot_params
import os
import simConst

try:
  import sim
  import time
except:
  print ('--------------------------------------------------------------')
  print ('"sim.py" could not be imported. This means very probably that')
  print ('either "sim.py" or the remoteApi library could not be found.')
  print ('Make sure both are in the same folder as this file,')
  print ('or appropriately adjust the file "sim.py"')
  print ('--------------------------------------------------------------')
  print ('')

client_ID = []


def sim_init():
  global sim
  global client_ID
  
  #Initialize sim interface
  sim.simxFinish(-1) # just in case, close all opened connections
  client_ID=sim.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to CoppeliaSim    
  if client_ID!=-1:
    print ('Connected to remote API server')
    return True
  else:
    return False

def load_scene(scene_path):
  global sim
  global client_ID
  
  # Ensure path is absolute for CoppeliaSim
  if not os.path.isabs(scene_path):
    scene_path = os.path.abspath(scene_path)
    
  print(f"Attempting to load scene: {scene_path}")
  
  # Best practice: stop simulation before loading a new scene
  sim.simxStopSimulation(client_ID, sim.simx_opmode_blocking)
  
  # Load the specified scene
  res = sim.simxLoadScene(client_ID, scene_path, 0, sim.simx_opmode_blocking)
  if res == sim.simx_return_ok:
    print(f"Successfully loaded scene: {scene_path}")
    # Give CoppeliaSim time to initialize the loaded scene
    time.sleep(2.0)
    return True
  else:
    print(f"Failed to load scene: {scene_path}. Error code: {res}")
    return False

def get_handle(name):
  global sim
  global client_ID
  
  # Ensure name starts with '/' for recent CoppeliaSim versions
  if not name.startswith('/'):
    name = '/' + name
    
  res, handle = sim.simxGetObjectHandle(client_ID, name, sim.simx_opmode_blocking)
  if res == sim.simx_return_ok:
    print(f"Successfully obtained handle for: {name}")
    return handle
  else:
    print(f"Failed to get handle for: {name}. Error code: {res}")
    return None

def localize_object(handle):
  global sim
  global client_ID
  
  # Get position of object (e.g. goal sphere)
  res, position = sim.simxGetObjectPosition(client_ID, handle, -1, sim.simx_opmode_blocking)
  if res == sim.simx_return_ok:
    return [position[0], position[1]]
  else:
    return None

def get_handles():
  #Get the handles to the sim items

  global pioneer_handle
  global pioneer_left_motor_handle
  global pioneer_right_motor_handle
  global bill1_handle
  global bill2_handle

  # Handle to Pioneer1:
  res , pioneer_handle = sim.simxGetObjectHandle(client_ID, "/Pioneer1", sim.simx_opmode_blocking)
  res,  pioneer_left_motor_handle = sim.simxGetObjectHandle(client_ID, "/Pioneer1/left", sim.simx_opmode_blocking)
  res,  pioneer_right_motor_handle = sim.simxGetObjectHandle(client_ID, "/Pioneer1/right", sim.simx_opmode_blocking)
  
  # Get the position of the Pioneer1 for the first time in streaming mode
  res , pioneer_1_Position = sim.simxGetObjectPosition(client_ID, pioneer_handle, -1 , sim.simx_opmode_streaming)
  res , pioneer_1_Orientation = sim.simxGetObjectOrientation(client_ID, pioneer_handle, -1 , sim.simx_opmode_streaming)
  
  # Stop all joint actuations:Make sure Pioneer1 is stationary:
  res = sim.simxSetJointTargetVelocity(client_ID, pioneer_left_motor_handle, 0, sim.simx_opmode_streaming)
  res = sim.simxSetJointTargetVelocity(client_ID, pioneer_right_motor_handle, 0, sim.simx_opmode_streaming)
  
  # Handle to Bills:
  res , bill1_handle = sim.simxGetObjectHandle(client_ID, "/Bill0/Bill", sim.simx_opmode_blocking)
  res , bill2_handle = sim.simxGetObjectHandle(client_ID, "/Bill1/Bill", sim.simx_opmode_blocking)
  
  # Get the position of the Bills for the first time in streaming mode
  res , pioneer_1_Position = sim.simxGetObjectPosition(client_ID, bill1_handle, -1 , sim.simx_opmode_streaming)
  res , pioneer_1_Position = sim.simxGetObjectPosition(client_ID, bill2_handle, -1 , sim.simx_opmode_streaming)
  
  print ("Succesfully obtained handles")

  return

def start_simulation():
  global sim
  global client_ID

  ###Start the Simulation: Keep printing out status messages!!!
  res = sim.simxStartSimulation(client_ID, sim.simx_opmode_oneshot_wait)

  if res == sim.simx_return_ok:
    print ("---!!! Started Simulation !!! ---")
    return True
  else:
    return False

def localize_object(handle):
    """Get world position (x, y) of any object by handle."""
    global sim
    global client_ID
    res, position = sim.simxGetObjectPosition(client_ID, handle, -1, sim.simx_opmode_blocking)
    if res == sim.simx_return_ok:
        return [position[0], position[1]]
    return None

def localize_robot():
    """Return [x, y, theta] of the Pioneer robot. Returns None on failure."""
    global sim
    global client_ID
    global pioneer_handle

    res_pos, p = sim.simxGetObjectPosition(client_ID, pioneer_handle, -1, sim.simx_opmode_buffer)
    res_ori, o = sim.simxGetObjectOrientation(client_ID, pioneer_handle, -1, sim.simx_opmode_buffer)

    if res_pos != sim.simx_return_ok or res_ori != sim.simx_return_ok:
        # Try blocking call as fallback
        res_pos, p = sim.simxGetObjectPosition(client_ID, pioneer_handle, -1, sim.simx_opmode_blocking)
        res_ori, o = sim.simxGetObjectOrientation(client_ID, pioneer_handle, -1, sim.simx_opmode_blocking)

    if res_pos == sim.simx_return_ok and res_ori == sim.simx_return_ok:
        return [p[0], p[1], o[2]]

    print("FAILED to localize robot!")
    return None

def get_static_obstacles():
    """Discover all wall objects in the scene and return their pose + size."""
    global sim
    global client_ID

    obstacles = []
    # sim_object_shape_type = 0 per simConst.py (NOT 6 — that is sim_object_reserved1!)
    res, handles, _, _, names = sim.simxGetObjectGroupData(
        client_ID, simConst.sim_object_shape_type, 0, sim.simx_opmode_blocking)

    if res != sim.simx_return_ok:
        print("ERROR: Could not retrieve scene objects from CoppeliaSim!")
        return obstacles

    for i, name in enumerate(names):
        if "Wall" not in name:
            continue

        handle = handles[i]

        # Get position and orientation (required)
        res_pos, pos = sim.simxGetObjectPosition(client_ID, handle, -1, sim.simx_opmode_blocking)
        res_ori, ori = sim.simxGetObjectOrientation(client_ID, handle, -1, sim.simx_opmode_blocking)

        if res_pos != sim.simx_return_ok:
            continue

        # --- Method 1: Try bounding box via simConst API ---
        sz_x, sz_y = None, None
        try:
            res_mnx, min_x = sim.simxGetObjectFloatParameter(client_ID, handle, simConst.sim_objfloatparam_objbbox_min_x, sim.simx_opmode_blocking)
            res_mxx, max_x = sim.simxGetObjectFloatParameter(client_ID, handle, simConst.sim_objfloatparam_objbbox_max_x, sim.simx_opmode_blocking)
            res_mny, min_y = sim.simxGetObjectFloatParameter(client_ID, handle, simConst.sim_objfloatparam_objbbox_min_y, sim.simx_opmode_blocking)
            res_mxy, max_y = sim.simxGetObjectFloatParameter(client_ID, handle, simConst.sim_objfloatparam_objbbox_max_y, sim.simx_opmode_blocking)
            if all(r == sim.simx_return_ok for r in [res_mnx, res_mxx, res_mny, res_mxy]):
                sz_x = abs(max_x - min_x)
                sz_y = abs(max_y - min_y)
                # Ensure sensible minimum (bounding box can be zero for flat objects)
                sz_x = max(sz_x, 0.1)
                sz_y = max(sz_y, 0.1)
        except Exception:
            pass

        # --- Method 2: Fallback — parse dimensions from CoppeliaSim wall name ---
        # e.g. "20cmHighWall100cm" → length=1.0m, thickness=0.2m
        # e.g. "20cmHighWall50cm"  → length=0.5m, thickness=0.2m
        if sz_x is None or sz_x < 0.05 or sz_y < 0.05:
            thickness = 0.2  # All walls are 20cm thick
            length = 1.0     # Default: 100cm long
            if "50cm" in name and name.index("50cm") > name.index("Wall"):
                length = 0.5
            sz_x, sz_y = length, thickness

        obstacles.append({
            "name": name, "pos": pos,
            "size": [sz_x, sz_y], "orientation": ori
        })
        print(f"  Wall: {name[:30]:30s} pos=({pos[0]:.2f},{pos[1]:.2f}) "
              f"size=({sz_x:.2f}x{sz_y:.2f}) ori={ori[2]:.2f}rad")

    print(f"Discovered {len(obstacles)} static wall sections")
    return obstacles

def localize_bills():
    #Function that will return the current location of Bills      
    global sim
    global client_ID
    global bill1_handle
    global bill2_handle
    
    res , bill1_Position = sim.simxGetObjectPosition(client_ID, bill1_handle, -1 , sim.simx_opmode_buffer)
    res , bill2_Position = sim.simxGetObjectPosition(client_ID, bill2_handle, -1 , sim.simx_opmode_buffer)
    
    print("bill1", int(bill1_Position[0]) , int(bill1_Position[1]))
    print("bill2", int(bill2_Position[0]) , int(bill2_Position[1]))

    return [[bill1_Position[0], bill1_Position[1]], [bill2_Position[0], bill2_Position[1]]]          

def freeze_goal_spheres(goal_handles):
    """Disable dynamic physics on all goal spheres so they stay in place.
    Call this AFTER start_simulation() so the engine is running."""
    global sim
    global client_ID
    # sim_objectintparam_model_dynamic = 3004 → set to 0 to make static
    # This is the CoppeliaSim constant for the shape's dynamic property
    for handle in goal_handles:
        if handle is None: continue
        # Remove from dynamics engine (makes it a static, non-moveable object)
        sim.simxSetObjectIntParameter(client_ID, handle,
                                      simConst.sim_shapeintparam_static,
                                      1, sim.simx_opmode_oneshot_wait)
    print(f"  Frozen {len([h for h in goal_handles if h])} goal spheres (static physics)")

def setvel_pioneers(V, W):
  #Function to set the linear and rotational velocity of pioneers
  global sim
  global client_ID
  global pioneer_left_motor_handle
  global pioneer_right_motor_handle

  # Limit v,w from controller to +/- of their max
  w = max(min(W, robot_params.pioneer_max_W), -1.0*robot_params.pioneer_max_W)
  v = max(min(V, robot_params.pioneer_max_V), -1.0*robot_params.pioneer_max_V)
          
  # Compute desired vel_r, vel_l needed to ensure w
  Vr = ((2.0*v) + (w*robot_params.pioneer_track_width))/(2*robot_params.pioneer_wheel_radius)
  Vl = ((2.0*v) - (w*robot_params.pioneer_track_width))/(2*robot_params.pioneer_wheel_radius)
                      
  # Set velocity
  sim.simxSetJointTargetVelocity(client_ID, pioneer_left_motor_handle, Vl, sim.simx_opmode_oneshot_wait)
  sim.simxSetJointTargetVelocity(client_ID, pioneer_right_motor_handle, Vr, sim.simx_opmode_oneshot_wait)
  
  return  

def hide_goal_sphere(handle):
    """Move a goal sphere far beneath the floor so it appears to disappear.
    We also freeze it in place by disabling dynamics so it cannot be pushed."""
    global sim
    global client_ID
    if handle is None: return
    
    # Move sphere 50m below the ground — effectively removing it from view
    sim.simxSetObjectPosition(client_ID, handle, -1, [0, 0, -50], sim.simx_opmode_oneshot_wait)
    print(f"  [Sphere {handle} collected and hidden]")

def sim_shutdown():
  #Gracefully shutdown simulation

  global sim
  global client_ID

  #Stop simulation
  res = sim.simxStopSimulation(client_ID, sim.simx_opmode_oneshot_wait)
  if res == sim.simx_return_ok:
    print ("---!!! Stopped Simulation !!! ---")

  # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
  sim.simxGetPingTime(client_ID)

  # Now close the connection to CoppeliaSim:
  sim.simxFinish(client_ID)      

  return            
