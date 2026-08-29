import sys
import os
import numpy as np
import networkx as nx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utilities.graph_utils import create_grid_network

class KinematicTrafficGenerator:
    """
    Simulates a translating traffic wave (kinematic wave) moving across the grid.
    Instead of isotropic diffusion, congestion shifts directionally.
    """
    def __init__(self, graph: nx.DiGraph, rho: float = 0.9, noise_std: float = 0.05, base_multiplier: float = 5.0, wave_direction='E'):
        self.graph = graph
        self.num_edges = graph.number_of_edges()
        self.edges = list(graph.edges())
        self.edge_indices = {edge: i for i, edge in enumerate(self.edges)}
        
        self.rho = rho
        self.noise_std = noise_std
        self.base_multiplier = base_multiplier
        self.wave_direction = wave_direction
        
        # Build directional upstream map instead of P matrix
        self.upstream_map = {i: [] for i in range(self.num_edges)}
        
        for i, (v, w) in enumerate(self.edges):
            v_pos = graph.nodes[v]['pos']
            w_pos = graph.nodes[w]['pos']
            
            for j, (u, v_inner) in enumerate(self.edges):
                if v_inner == v:
                    u_pos = graph.nodes[u]['pos']
                    if wave_direction == 'E' and u_pos[1] < v_pos[1]:
                        self.upstream_map[i].append(j)
                    elif wave_direction == 'S' and u_pos[0] < v_pos[0]:
                        self.upstream_map[i].append(j)
                        
        self.base_weights = np.array([graph.edges[e].get('base_weight', 1.0) for e in self.edges])
        self.congestion = np.zeros(self.num_edges)
        
    def reset(self, np_random: np.random.Generator):
        self.np_random = np_random
        self.congestion = np.zeros(self.num_edges)
        return self._get_current_weights()
        
    def step(self):
        # Kinematic wave: MAX pool from upstream to maintain amplitude
        propagation = np.zeros(self.num_edges)
        for i in range(self.num_edges):
            up_edges = self.upstream_map[i]
            if len(up_edges) > 0:
                propagation[i] = np.max(self.congestion[up_edges])
        
        # persistence vs propagation
        speed = 0.9 # Move fast!
        historical = (1.0 - speed) * self.congestion + speed * propagation
        
        noise = self.np_random.normal(0, self.noise_std, self.num_edges)
        
        # Spontaneous shock generation on the upstream boundary
        if self.np_random.random() < 0.2:
            west_edges = [i for i, (u, v) in enumerate(self.edges) if self.graph.nodes[u]['pos'][1] == 0 and self.graph.nodes[v]['pos'][1] == 1]
            if west_edges:
                shock_edge = self.np_random.choice(west_edges)
                historical[shock_edge] = 1.0
                
        self.congestion = np.clip(self.rho * historical + (1 - self.rho) * np.abs(noise), 0, 1)
        
    def _get_current_weights(self):
        return self.base_weights * (1.0 + self.congestion * self.base_multiplier)

def test_wave():
    graph = create_grid_network(10, 10, seed=42)
    gen = KinematicTrafficGenerator(graph, rho=1.0, noise_std=0.0)
    gen.reset(np.random.RandomState(42))
    
    # Inject a manual shock on the left boundary
    west_edges = [i for i, (u, v) in enumerate(gen.edges) if graph.nodes[u]['pos'][1] == 0 and graph.nodes[v]['pos'][1] == 1]
    shock_edge = west_edges[3] # some edge on row 3
    gen.congestion[shock_edge] = 1.0
    
    for t in range(20):
        print(f"--- Step {t} ---")
        for i, val in enumerate(gen.congestion):
            if val > 0.1:
                u, v = gen.edges[i]
                u_pos, v_pos = graph.nodes[u]['pos'], graph.nodes[v]['pos']
                print(f"Edge {u_pos}->{v_pos}: {val:.3f}")
        gen.step()

if __name__ == "__main__":
    test_wave()
