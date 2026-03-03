import matplotlib.pyplot as plt
import numpy as np
import os

def plot_trajectory(grid, start, goals, trajectory, output_dir=None, filename="map_trajectory.png"):
    """
    Plots the grid, start node, goal nodes, and the computed trajectory.
    """
    plt.figure(figsize=(10, 10))
    
    # 1. Plot the grid map
    # Create a color map: 0->white (free), 1->black (obstacle)
    cmap = plt.cm.get_cmap('Greys', 2)
    plt.imshow(grid.grid, cmap=cmap, origin='lower', extent=[-0.5, grid.width-0.5, -0.5, grid.height-0.5])
    
    # 2. Plot gridlines
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, grid.width, 1))
    ax.set_yticks(np.arange(-0.5, grid.height, 1))
    ax.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # 3. Plot start and goals
    plt.scatter(*start, marker='o', color='blue', s=150, label='Start')
    
    for i, goal in enumerate(goals):
        plt.scatter(*goal, marker='*', color='gold', s=200, edgecolor='black')
        plt.text(goal[0]+0.2, goal[1]+0.2, f'G{i+1}', fontsize=12, fontweight='bold', color='gold')
        
    # Dummy plot for legend
    plt.scatter([], [], marker='*', color='gold', s=100, label='Goals')
    
    # 4. Plot the trajectory
    if trajectory:
        xs, ys = zip(*trajectory)
        plt.plot(xs, ys, '-', color='red', linewidth=3, alpha=0.7, label='Trajectory')
        
    plt.title("Multi-Goal Motion Planning Trajectory")
    plt.legend(loc='upper right')
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_metrics(results_df, output_dir=None):
    """
    Uses pandas dataframe (from experiment_runner) to plot graphs
    """
    if results_df.empty:
        print("No data to plot.")
        return
        
    # Example: Plot Cost vs Number of Goals grouped by Planner
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for planner in results_df['Planner'].unique():
        subset = results_df[results_df['Planner'] == planner]
        avg_cost = subset.groupby('Num_Goals')['Total_Cost'].mean()
        plt.plot(avg_cost.index, avg_cost.values, marker='o', label=planner)
        
    plt.title('Average Mission Cost vs Number of Goals')
    plt.xlabel('Number of Goals')
    plt.ylabel('Total Cost')
    plt.legend()
    plt.grid(True)
    
    # Example: Plot Expansions vs Sequence Method
    plt.subplot(1, 2, 2)
    for planner in results_df['Planner'].unique():
        subset = results_df[results_df['Planner'] == planner]
        avg_exp = subset.groupby('Num_Goals')['Nodes_Expanded'].mean()
        plt.plot(avg_exp.index, avg_exp.values, marker='s', label=planner)
        
    plt.title('Average Nodes Expanded vs Number of Goals')
    plt.xlabel('Number of Goals')
    plt.ylabel('Nodes Expanded')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "metrics_plot.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Metrics plot saved to: {save_path}")
        
    plt.show()
