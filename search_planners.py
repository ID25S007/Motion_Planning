class LocalPlanner:
    def __init__(self, grid, method="astar", weight=1.0):
        # TODO: initialize local planner
        pass

    def plan(self, start, goal):
        # TODO: call appropriate internal method based on initialized method
        # TODO: return path, cost, nodes_expanded
        pass

    def _astar(self, start, goal):
        # TODO: A* implementation
        pass

    def _weighted_astar(self, start, goal):
        # TODO: Weighted A* implementation
        pass
        
    def _anytime_astar(self, start, goal):
        # TODO: Anytime A* implementation
        pass
