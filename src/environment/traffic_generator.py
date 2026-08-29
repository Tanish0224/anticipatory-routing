import numpy as np
import networkx as nx

class TrafficGenerator:
    """
    Generates time-varying edge weights for a network graph to simulate traffic congestion.
    Supports multiple regimes:
    0: Static
    1: Independent dynamic noise
    2: Spatiotemporally correlated congestion
    """
    def __init__(self, graph: nx.DiGraph, regime: int = 0, rho: float = 0.0, noise_std: float = 0.1, base_multiplier: float = 2.0):
        self.graph = graph
        self.num_edges = graph.number_of_edges()
        self.edges = list(graph.edges())
        self.edge_indices = {edge: i for i, edge in enumerate(self.edges)}
        
        # Build upstream adjacency matrix for propagation (diffusion, regime 2)
        self.P = np.zeros((self.num_edges, self.num_edges))
        for i, (v, w) in enumerate(self.edges):
            upstream_edges = [self.edge_indices[(u, v_inner)] for u, v_inner in self.edges if v_inner == v]
            if len(upstream_edges) > 0:
                self.P[i, upstream_edges] = 1.0 / len(upstream_edges) # average upstream congestion
                
        # Build directional upstream map for translating wave (regime 3)
        self.upstream_map_E = {i: [] for i in range(self.num_edges)}
        for i, (v, w) in enumerate(self.edges):
            v_pos = graph.nodes[v].get('pos', v) # fallback if pos not set
            w_pos = graph.nodes[w].get('pos', w)
            for j, (u, v_inner) in enumerate(self.edges):
                if v_inner == v:
                    u_pos = graph.nodes[u].get('pos', u)
                    if isinstance(u_pos, tuple) and isinstance(v_pos, tuple):
                        if u_pos[1] < v_pos[1]: # flowing East
                            self.upstream_map_E[i].append(j)
                            
        self.regime = regime
        self.rho = rho # correlation parameter
        self.noise_std = noise_std
        self.base_multiplier = base_multiplier
        
        # Extract base weights
        self.base_weights = np.array([graph.edges[e].get('base_weight', 1.0) for e in self.edges])
        self.congestion = np.zeros(self.num_edges)
        
    def reset(self, np_random: np.random.Generator):
        self.np_random = np_random
        self.congestion = np.zeros(self.num_edges)
        if self.regime in [1, 2]:
            self.congestion = np.clip(self.np_random.normal(0, self.noise_std, self.num_edges), 0, 1)
        return self._get_current_weights()
        
    def step(self):
        if self.regime == 0:
            pass
        elif self.regime == 1:
            noise = self.np_random.normal(0, self.noise_std * 2, self.num_edges)
            self.congestion = np.clip(noise, 0, 1)
        elif self.regime == 2:
            # Diffusing spatiotemporal correlation
            propagation = self.P @ self.congestion
            historical = 0.7 * self.congestion + 0.3 * propagation
            noise = self.np_random.normal(0, self.noise_std, self.num_edges)
            self.congestion = np.clip(self.rho * historical + (1 - self.rho) * np.abs(noise), 0, 1)
        elif self.regime == 3:
            # Translating kinematic wave (Eastward)
            propagation = np.zeros(self.num_edges)
            for i in range(self.num_edges):
                up_edges = self.upstream_map_E[i]
                if len(up_edges) > 0:
                    propagation[i] = np.max(self.congestion[up_edges])
            
            speed = 0.9 # Fast wave movement
            historical = (1.0 - speed) * self.congestion + speed * propagation
            noise = self.np_random.normal(0, self.noise_std, self.num_edges)
            
            # Spontaneous shock generation on western boundary
            if self.np_random.random() < 0.2:
                west_edges = [i for i, (u, v) in enumerate(self.edges) 
                              if isinstance(self.graph.nodes[u].get('pos', u), tuple) 
                              and self.graph.nodes[u].get('pos', u)[1] == 0 
                              and self.graph.nodes[v].get('pos', v)[1] == 1]
                if west_edges:
                    shock_edge = self.np_random.choice(west_edges)
                    historical[shock_edge] = 1.0
                    
            self.congestion = np.clip(self.rho * historical + (1 - self.rho) * np.abs(noise), 0, 1)
            
        return self._get_current_weights()
        
    def _get_current_weights(self):
        # Actual weight = base_weight * (1 + congestion * multiplier)
        # So if congestion is 0, weight = base. If 1, weight = base * (1 + multiplier)
        current_weights = self.base_weights * (1.0 + self.congestion * self.base_multiplier)
        
        # Update the graph object so classical algorithms can query it easily if needed
        # (Though they should strictly observe only via the state vector)
        weight_dict = {self.edges[i]: current_weights[i] for i in range(self.num_edges)}
        nx.set_edge_attributes(self.graph, weight_dict, 'weight')
        
        return current_weights
