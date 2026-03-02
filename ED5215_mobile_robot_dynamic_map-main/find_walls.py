import sim
import os
import time

def find_walls():
    sim.simxFinish(-1)
    client_ID = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    if client_ID == -1:
        print("Failed to connect to sim")
        return

    print("Connected to CoppeliaSim")
    
    # We'll try to get all object handles
    res, handles, _, _, names = sim.simxGetObjectGroupData(client_ID, sim.sim_appobj_object_type, 0, sim.simx_opmode_blocking)
    
    if res != sim.simx_return_ok:
        print(f"Failed to get object handles: {res}")
        sim.simxFinish(client_ID)
        return

    walls = []
    for i, name in enumerate(names):
        if "Wall" in name:
            handle = handles[i]
            res, pos = sim.simxGetObjectPosition(client_ID, handle, -1, sim.simx_opmode_blocking)
            # Try to get bounding box to estimate size
            res, min_x = sim.simxGetObjectFloatParameter(client_ID, handle, sim.sim_objfloatparam_objbbox_min_x, sim.simx_opmode_blocking)
            res, max_x = sim.simxGetObjectFloatParameter(client_ID, handle, sim.sim_objfloatparam_objbbox_max_x, sim.simx_opmode_blocking)
            res, min_y = sim.simxGetObjectFloatParameter(client_ID, handle, sim.sim_objfloatparam_objbbox_min_y, sim.simx_opmode_blocking)
            res, max_y = sim.simxGetObjectFloatParameter(client_ID, handle, sim.sim_objfloatparam_objbbox_max_y, sim.simx_opmode_blocking)
            
            size_x = max_x - min_x
            size_y = max_y - min_y
            
            # Since orientation matters, we should also get orientation
            res, orientation = sim.simxGetObjectOrientation(client_ID, handle, -1, sim.simx_opmode_blocking)
            
            walls.append({
                "name": name,
                "pos": pos,
                "size": [size_x, size_y],
                "orientation": orientation
            })
            print(f"Found Wall: {name} at {pos} size {size_x}x{size_y} rot {orientation[2]}")

    sim.simxFinish(client_ID)
    return walls

if __name__ == "__main__":
    find_walls()
