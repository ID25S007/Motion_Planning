from search_planners import LocalPlanner
import sequencing  # We will implement this next

class MultiGoalPlanner:
    def __init__(self, grid, planner_type="astar", sequencing_type="optimal"):
        self.grid = grid
        self.planner_type = planner_type
        self.sequencing_type = sequencing_type
        self.local_planner = LocalPlanner(grid, method=planner_type)

    def compute_pairwise_costs(self, start, goals):
        """
        Computes the cost and paths between all pairs:
        - start to all goals
        - each goal to all other goals
        """
        all_nodes = [start] + goals
        num_nodes = len(all_nodes)
        
        cost_matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        paths_matrix = [[[] for _ in range(num_nodes)] for _ in range(num_nodes)]
        
        total_nodes_expanded = 0
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                    
                path, cost, expanded = self.local_planner.plan(all_nodes[i], all_nodes[j])
                
                cost_matrix[i][j] = cost
                paths_matrix[i][j] = path
                total_nodes_expanded += expanded
                
        return cost_matrix, paths_matrix, total_nodes_expanded

    def plan_mission(self, start, goals):
        """
        Plans the complete mission traversing the start to all goals.
        """
        # 1. Compute pairwise costs and paths
        cost_matrix, paths_matrix, expansion_cost_phase = self.compute_pairwise_costs(start, goals)
        
        # 2. Get the optimal ordering from the sequencing module
        goal_indices = list(range(1, len(goals) + 1))
        
        if self.sequencing_type == "optimal":
            ordered_indices, seq_cost = sequencing.solve_optimal(cost_matrix, 0, goal_indices)
        elif self.sequencing_type == "nearest_neighbor":
            ordered_indices, seq_cost = sequencing.nearest_neighbor(cost_matrix, 0, goal_indices)
        else:
            raise ValueError(f"Unknown sequencing type: {self.sequencing_type}")
            
        # 3. Concatenate the paths
        full_trajectory = []
        current_idx = 0
        
        for next_idx in ordered_indices:
            # Append the path (excluding the first node to avoid duplicates, except for the very first segment)
            segment = paths_matrix[current_idx][next_idx]
            if not full_trajectory:
                full_trajectory.extend(segment)
            else:
                full_trajectory.extend(segment[1:])
            
            current_idx = next_idx
            
        # 4. Return metrics
        metrics = {
            "total_mission_cost": seq_cost,
            "total_nodes_expanded": expansion_cost_phase, # Adding planning expansions
            "sequence_chosen": [goals[i-1] for i in ordered_indices] # actual goal coordinates
        }
        
        return full_trajectory, metrics
