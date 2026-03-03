import random
import matplotlib.pyplot as plt

class GridMap:
    def __init__(self, width, height, obstacle_density, seed=None):
        self.width = width
        self.height = height
        self.obstacle_density = obstacle_density
        
        if seed is not None:
            random.seed(seed)
            
        self.grid = self._generate_grid()

    def _generate_grid(self):
        # 0 = free, 1 = obstacle
        grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        
        # Calculate number of obstacles to place
        total_cells = self.width * self.height
        num_obstacles = int(total_cells * self.obstacle_density)
        
        # Place obstacles randomly
        cells_placed = 0
        while cells_placed < num_obstacles:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            
            # Don't place obstacle on existing obstacle
            if grid[y][x] == 0:
                grid[y][x] = 1
                cells_placed += 1
                
        return grid

    def is_free(self, node):
        x, y = node
        # Check boundaries
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
            
        # Check obstacle (grid is accessed as grid[y][x])
        return self.grid[y][x] == 0
        
    def ensure_free(self, node):
        """Forces a node to be free (useful for ensuring start/goals aren't blocked)"""
        x, y = node
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = 0

    def get_neighbors(self, node):
        x, y = node
        neighbors = []
        
        # 4-connected grid (Up, Down, Left, Right)
        directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_free((nx, ny)):
                neighbors.append((nx, ny))
                
        return neighbors

    def visualize(self):
        plt.figure(figsize=(8, 8))
        # grid is plotted such that (0,0) is bottom left
        plt.imshow(self.grid, cmap='Greys', origin='lower')
        plt.title('Environment Grid Map')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        plt.show()
