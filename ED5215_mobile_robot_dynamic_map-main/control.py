import robot_params
import numpy as np 

prev_heading_error = 0.0
total_heading_error = 0.0

def reset_pid():
    """Reset PID state between goals to prevent integral windup.
    Call this once before starting navigation toward each new goal."""
    global prev_heading_error, total_heading_error
    prev_heading_error  = 0.0
    total_heading_error = 0.0

def at_goal(robot_state, goal_state):    
    
    #check if we have reached goal point
    d = np.sqrt(((goal_state[0] - robot_state[0])**2) + ((goal_state[1] - robot_state[1])**2))
    
    if d <= robot_params.goal_threshold:
        print("Reached goal")
        return True
    else:
        return False

def gtg(robot_state, goal_state):  
    #The Go to goal controller
    
    global prev_heading_error
    global total_heading_error   
    
    #Controller parameters — tuned for 0.06 s control loop
    Kp = 0.06     # proportional: snaps heading fast, W=0.6 at ~10° error
    Kd = 0.002    # derivative:   damps overshoot
    Ki = 0.0      # integral:     disabled (avoids windup)
    dt = 0.06     # must match STEP_DELAY in main.py

    #determine how far to rotate to face the goal point
    #PS. ALL ANGLES ARE IN RADIANS
    delta_theta = (np.arctan2((goal_state[1] - robot_state[1]), (goal_state[0] - robot_state[0]))) - robot_state[2]
    #restrict angle to (-pi,pi)
    delta_theta = ((delta_theta + np.pi)%(2.0*np.pi)) - np.pi
    
    #Error is delta_theta in degrees
    e_new = ((delta_theta*180.0)/np.pi)
    e_dot = (e_new - prev_heading_error)/dt 
    total_heading_error = (total_heading_error + e_new)*dt
    #control input for angular velocity
    W = (Kp*e_new) + (Ki*total_heading_error) + (Kd*e_dot)
    prev_heading_error = e_new
  
    #find distance to goal
    d = np.sqrt(((goal_state[0] - robot_state[0])**2) + ((goal_state[1] - robot_state[1])**2))
    
    #velocity parameters
    distThresh = 0.1#mm
    
    orientation_factor = max(0.0, np.cos(delta_theta))
    #control input for linear velocity
    V = robot_params.pioneer_max_V * np.tanh(3.0 * (d - distThresh)) * orientation_factor
    
    #request robot to execute velocity
    return[V,W]
