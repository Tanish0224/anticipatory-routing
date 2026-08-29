import numpy as np
import networkx as nx
from collections import deque

class OracleDijkstra:
    """
    An Oracle planner that perfectly foresees all future traffic states.
    It builds a space-time graph for the episode horizon and finds the 
    absolute optimal path. It serves purely as an upper bound for analysis.
    """
    def __init__(self, env):
        self.env = env
        self.max_steps = env.max_steps
        self.num_nodes = env.num_nodes
        self.dest = env.dest_node
        self.edges = env.traffic_gen.edges
        self.edge_indices = {e: i for i, e in enumerate(self.edges)}
        self.future_weights = [] # Shape: [max_steps, num_edges]
        
    def reset(self, future_weights):
        # future_weights is pre-computed by the environment runner
        self.future_weights = future_weights
        self.current_step = 0
        self.dest = self.env.dest_node
        
        # Precompute the optimal space-time path backwards from the destination
        # DP table: min_cost[time][node]
        self.min_cost = np.full((self.max_steps + 1, self.num_nodes), np.inf)
        self.best_next_node = np.full((self.max_steps + 1, self.num_nodes), -1, dtype=int)
        
        # Base case: at any time, being at destination costs 0
        for t in range(self.max_steps + 1):
            self.min_cost[t][self.dest] = 0.0
            
        # DP backwards in time
        for t in range(self.max_steps - 1, -1, -1):
            for u in range(self.num_nodes):
                if u == self.dest:
                    continue
                # For each neighbor
                for v in self.env.graph.successors(u):
                    edge_idx = self.edge_indices[(u, v)]
                    cost = self.future_weights[t][edge_idx]
                    
                    if self.min_cost[t+1][v] + cost < self.min_cost[t][u]:
                        self.min_cost[t][u] = self.min_cost[t+1][v] + cost
                        self.best_next_node[t][u] = v
                        
    def act(self, obs):
        if self.current_step >= self.max_steps:
            return 0 # random, we failed to reach
            
        current_node = np.argmax(obs[:self.num_nodes])
        next_node = self.best_next_node[self.current_step][current_node]
        
        self.current_step += 1
        
        if next_node == -1:
            # No valid path found within time horizon, just stay or pick random neighbor
            neighbors = list(self.env.graph.successors(current_node))
            if neighbors:
                next_node = neighbors[0]
            else:
                next_node = current_node
                
        return next_node
