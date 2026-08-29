import networkx as nx
import numpy as np

class ReactiveDijkstra:
    """
    Computes the shortest path at each step using the CURRENT observed edge weights.
    It reacts perfectly to present conditions but assumes they will remain constant.
    """
    def __init__(self, env):
        self.env = env
        
    def act(self, obs):
        # Extract graph state from environment
        current_node = self.env.current_node
        dest_node = self.env.dest_node
        graph = self.env.graph
        
        try:
            # Dijkstra using current 'weight' attribute on edges
            path = nx.shortest_path(graph, source=current_node, target=dest_node, weight='weight')
            if len(path) > 1:
                next_node = path[1]
                return self.env.node_to_idx[next_node]
            else:
                # Already at dest (shouldn't happen in step)
                return self.env.node_to_idx[current_node]
        except nx.NetworkXNoPath:
            # Fallback to random valid action if no path
            valid = self.env.get_valid_actions()
            return np.random.choice(valid) if valid else 0

class PredictiveDijkstra:
    """
    Computes shortest path using Holt's Linear Trend (Double Exponential Smoothing) 
    of edge weights. This acts as a highly competitive classical forecasting + planning 
    baseline that can anticipate trends (momentum) in traffic.
    """
    def __init__(self, env, alpha=0.3, beta=0.1):
        self.env = env
        self.alpha = alpha
        self.beta = beta
        self.level = None
        self.trend = None
        
    def reset(self):
        # Initialize Holt's components
        initial_weights = np.array([self.env.graph.edges[e]['weight'] for e in self.env.traffic_gen.edges])
        self.level = initial_weights.copy()
        self.trend = np.zeros_like(initial_weights)
        
    def update_forecast(self):
        current_weights = np.array([self.env.graph.edges[e]['weight'] for e in self.env.traffic_gen.edges])
        if self.level is None:
            self.reset()
        else:
            last_level = self.level.copy()
            self.level = self.alpha * current_weights + (1 - self.alpha) * (last_level + self.trend)
            self.trend = self.beta * (self.level - last_level) + (1 - self.beta) * self.trend
            
    def act(self, obs=None):
        self.update_forecast()
        
        current_node = self.env.current_node
        dest_node = self.env.dest_node
        
        # Forecast 1 step into the future
        forecast_weights = self.level + self.trend
        
        # Ensure weights don't go negative (Dijkstra requirement)
        forecast_weights = np.clip(forecast_weights, 0.01, None)
        
        # Create a temporary graph with forecasted weights
        temp_graph = self.env.graph.copy()
        weight_dict = {self.env.traffic_gen.edges[i]: forecast_weights[i] for i in range(len(forecast_weights))}
        nx.set_edge_attributes(temp_graph, weight_dict, 'weight')
        
        try:
            path = nx.shortest_path(temp_graph, source=current_node, target=dest_node, weight='weight')
            if len(path) > 1:
                next_node = path[1]
                return self.env.node_to_idx[next_node]
            else:
                return self.env.node_to_idx[current_node]
        except nx.NetworkXNoPath:
            valid = self.env.get_valid_actions()
            return np.random.choice(valid) if valid else 0
