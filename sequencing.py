import itertools

def solve_optimal(cost_matrix, start_index, goal_indices):
    """
    Brute-force permutation of all possible goal orderings to find the minimum cost.
    Useful for finding the true optimal sequence for a small number of goals.
    
    Args:
        cost_matrix (list of lists): 2D square matrix of pairwise costs.
        start_index (int): The index of the starting point in the cost_matrix.
        goal_indices (list of int): A list of indices corresponding to the goals in the cost_matrix.
        
    Returns:
        tuple: (best_sequence, min_cost)
               best_sequence is a list of goal indices in the order they should be visited.
               min_cost is the total cost of traversing that sequence.
    """
    if not goal_indices:
        return [], 0
    min_cost = float('inf')
    best_sequence = None
    
    # Generate all possible permutations of the goals
    for sequence in itertools.permutations(goal_indices):
        current_cost = 0
        current_node = start_index
        
        # Calculate the cost of this specific permutation
        for next_node in sequence:
            current_cost += cost_matrix[current_node][next_node]
            current_node = next_node
            
        # Update if we found a better/cheaper sequence
        if current_cost < min_cost:
            min_cost = current_cost
            best_sequence = list(sequence)
            
    return best_sequence, min_cost


def nearest_neighbor(cost_matrix, start_index, goal_indices):
    """
    Greedy heuristic: always visit the nearest unvisited goal next.
    Much faster than optimal, but not guaranteed to find the absolute shortest total sequence.
    
    Args:
        cost_matrix (list of lists): 2D square matrix of pairwise costs.
        start_index (int): The index of the starting point in the cost_matrix.
        goal_indices (list of int): A list of indices corresponding to goals.
        
    Returns:
        tuple: (sequence, total_cost)
    """
    if not goal_indices:
        return [], 0
        
    unvisited = set(goal_indices)
    current_node = start_index
    sequence = []
    total_cost = 0
    
    while unvisited:
        # Find the nearest unvisited neighbor
        nearest = None
        min_dist = float('inf')
        
        for candidate in unvisited:
            dist = cost_matrix[current_node][candidate]
            if dist < min_dist:
                min_dist = dist
                nearest = candidate
                
        # Move to that nearest neighbor
        sequence.append(nearest)
        total_cost += min_dist
        current_node = nearest
        unvisited.remove(nearest)
        
    return sequence, total_cost
