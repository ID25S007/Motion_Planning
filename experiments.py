import os
import time
import pandas as pd
import random
from grid_map import GridMap
from multi_goal_planner import MultiGoalPlanner
from visualization import plot_trajectory, plot_metrics

def generate_valid_locations(grid, num_locations):
    """
    Randomly generates a list of free (start + goals) coordinates on the map.
    """
    locations = []
    while len(locations) < num_locations:
        x = random.randint(0, grid.width - 1)
        y = random.randint(0, grid.height - 1)
        # Verify valid and unique
        if grid.is_free((x, y)) and (x, y) not in locations:
            locations.append((x, y))
    return locations

def run_experiments():
    # Setup results output
    output_dir = "results"
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Experiment sweeping parameters
    grid_sizes = [30] #, 50]
    num_goals_list = [3, 5]
    planner_types = ["astar", "weighted_astar"] # could add "anytime_astar"
    sequencing_types = ["nearest_neighbor", "optimal"]
    trials = 2
    
    results = []
    
    print("--- Starting Synthetic Benchmarks ---")
    
    for size in grid_sizes:
        for goals_count in num_goals_list:
            for planner in planner_types:
                for sequencer in sequencing_types:
                    # Skip optimal sequencing constraint for large N (it's brute force!)
                    if sequencer == "optimal" and goals_count > 6:
                        continue
                        
                    for trial in range(trials):
                        print(f"Running: Grid {size}x{size} | {goals_count} Goals | {planner} | {sequencer} | Trial {trial+1}/{trials}")
                        
                        # Set a seed so different planners test same environment geometry
                        seed = trial * 10 
                        
                        # 1. Generate Environment
                        # Keep density relatively low to guarantee paths
                        grid = GridMap(size, size, obstacle_density=0.15, seed=seed)
                        
                        # 2. Pick Start and Goals consistently across the trial
                        random.seed(seed) 
                        locations = generate_valid_locations(grid, goals_count + 1)
                        start = locations[0]
                        goals = locations[1:]
                        
                        # 3. Initialize Mission Planner
                        mp = MultiGoalPlanner(grid, planner_type=planner, sequencing_type=sequencer)
                        
                        # 4. Plan the mission & log time
                        start_time = time.time()
                        full_trajectory, metrics = mp.plan_mission(start, goals)
                        runtime = time.time() - start_time
                        
                        # 5. Record trial results
                        trial_data = {
                            "Grid_Size": size,
                            "Num_Goals": goals_count,
                            "Planner": planner,
                            "Sequencing": sequencer,
                            "Total_Cost": metrics["total_mission_cost"],
                            "Nodes_Expanded": metrics["total_nodes_expanded"],
                            "Runtime_Sec": runtime,
                            "Path_Length": len(full_trajectory)
                        }
                        results.append(trial_data)
                        
                        # Only plot the very first trial to avoid flooding folders
                        if trial == 0 and size == grid_sizes[0]:
                            plot_filename = f"traj_{planner}_{sequencer}_{goals_count}G_trial{trial}.png"
                            # This will pop up a window — let's uncomment for now so it runs headlessly
                            plot_trajectory(grid, start, goals, full_trajectory, output_dir=plots_dir, filename=plot_filename)

    # 6. Save Data
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n--- Experiments Completed ---")
    print(f"Results saved to: {csv_path}")
    
    # 7. Generate Comparison Metrics plots
    print("Generating aggregate metrics charts...")
    plot_metrics(df, output_dir=plots_dir)

if __name__ == "__main__":
    run_experiments()
