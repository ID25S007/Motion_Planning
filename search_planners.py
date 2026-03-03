import heapq
import time
import math

class LocalPlanner:
    def __init__(self, grid, method="astar", weight=1.0, time_limit=5.0):
        self.grid = grid
        self.method = method.lower()
        self.weight = weight
        self.time_limit = time_limit

    def manhattan_distance(self, node, goal):
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    def plan(self, start, goal):
        if self.method == "astar":
            return self._astar(start, goal, weight=1.0)
        elif self.method == "weighted_astar":
            return self._astar(start, goal, weight=self.weight)
        elif self.method == "anytime_astar":
            return self._anytime_astar(start, goal)
        else:
            raise ValueError(f"Unknown planner method: {self.method}")

    def _astar(self, start, goal, weight=1.0):
        # fringe elements: (f_cost, g_cost, current_node, path)
        fringe = []
        heapq.heappush(fringe, (0, 0, start, [start]))
        
        g_costs = {start: 0}
        nodes_expanded = 0
        
        while fringe:
            current_f, current_g, current_node, current_path = heapq.heappop(fringe)
            
            # Reached goal
            if current_node == goal:
                return current_path, current_g, nodes_expanded
                
            # If we pulled out a stale, sub-optimal path, ignore it
            if current_g > g_costs.get(current_node, float('inf')):
                continue
                
            nodes_expanded += 1
            
            for neighbor in self.grid.get_neighbors(current_node):
                edge_cost = 1  # Assuming uniform grid distance cost
                tentative_g = current_g + edge_cost
                
                if tentative_g < g_costs.get(neighbor, float('inf')):
                    g_costs[neighbor] = tentative_g
                    h_cost = self.manhattan_distance(neighbor, goal)
                    f_cost = tentative_g + (weight * h_cost)
                    
                    new_path = list(current_path)
                    new_path.append(neighbor)
                    
                    heapq.heappush(fringe, (f_cost, tentative_g, neighbor, new_path))
                    
        return [], float('inf'), nodes_expanded

    def _weighted_astar(self, start, goal):
        return self._astar(start, goal, weight=self.weight)

    def _anytime_astar(self, start, goal):
        """
        Iteratively decreases weight to find better paths within a time limit.
        """
        current_weight = self.weight
        best_path = []
        best_cost = float('inf')
        total_expanded = 0
        
        start_time = time.time()
        
        while current_weight >= 1.0:
            if time.time() - start_time > self.time_limit:
                break
                
            path, cost, expanded = self._astar(start, goal, weight=current_weight)
            total_expanded += expanded
            
            if path and cost < best_cost:
                best_cost = cost
                best_path = path
                
            current_weight -= 0.5
            
        return best_path, best_cost, total_expanded
