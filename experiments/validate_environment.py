import sys
import os
import numpy as np
import networkx as nx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utilities.graph_utils import create_grid_network
from src.environment.traffic_generator import TrafficGenerator

def test_graph_properties():
    print("--- 1. Graph Properties ---")
    G = create_grid_network(10, 10, seed=42)
    assert nx.is_strongly_connected(G), "Graph is not strongly connected!"
    
    # Check edge weights positivity
    weights = [G.edges[e]['base_weight'] for e in G.edges()]
    assert min(weights) > 0, "Non-positive edge weights found!"
    print("Graph is strongly connected. All edge weights positive.")
    
def test_seed_reproducibility():
    print("--- 2. Seed Reproducibility ---")
    G = create_grid_network(10, 10, seed=42)
    gen1 = TrafficGenerator(G, regime=3, rho=0.9)
    gen2 = TrafficGenerator(G, regime=3, rho=0.9)
    
    w1 = gen1.reset(np.random.RandomState(42))
    w2 = gen2.reset(np.random.RandomState(42))
    assert np.allclose(w1, w2), "Initial reset not reproducible!"
    
    gen1.step()
    gen2.step()
    assert np.allclose(gen1.congestion, gen2.congestion), "Step not reproducible!"
    print("Seed reproducibility verified.")
    
def test_time_to_impact():
    print("--- 3. Time-to-Impact ---")
    G = create_grid_network(10, 10, seed=42)
    gen = TrafficGenerator(G, regime=3, rho=1.0, noise_std=0.0) # no noise to isolate wave
    gen.reset(np.random.RandomState(42))
    
    # Find west edges
    west_edges = [i for i, (u, v) in enumerate(gen.edges) if G.nodes[u]['pos'][1] == 0 and G.nodes[v]['pos'][1] == 1]
    shock_edge = west_edges[5] # middle row
    gen.congestion[shock_edge] = 1.0
    
    # Step and track max distance
    for t in range(1, 15):
        gen.step()
        affected = [i for i, c in enumerate(gen.congestion) if c > 0.5] # high intensity threshold
        if len(affected) == 0:
            print(f"Step {t}: Wave dissipated below 0.5")
            break
        max_col = max([G.nodes[gen.edges[i][1]]['pos'][1] for i in affected])
        print(f"Step {t}: Wave reached column {max_col}. Edges >0.5: {len(affected)}")
        
if __name__ == "__main__":
    test_graph_properties()
    test_seed_reproducibility()
    test_time_to_impact()
