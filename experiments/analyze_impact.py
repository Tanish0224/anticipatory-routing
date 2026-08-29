import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utilities.graph_utils import create_grid_network
from src.environment.traffic_generator import TrafficGenerator

def analyze_time_to_impact(rows=4, cols=4):
    print(f"--- Time-to-Impact Analysis ({rows}x{cols}) ---")
    graph = create_grid_network(rows=rows, cols=cols, seed=42)
    traffic_gen = TrafficGenerator(graph, regime=2, rho=1.0, noise_std=0.0)
    
    # Simulate propagation from a single point
    # Reset traffic
    traffic_gen.congestion = np.zeros(traffic_gen.num_edges)
    
    # Inject shock at a specific edge (e.g., edge 0)
    shock_edge = 0
    traffic_gen.congestion[shock_edge] = 1.0
    traffic_gen.np_random = np.random.RandomState(42)
    
    print(f"Step 0: Congestion at edge {shock_edge} = 1.0")
    for t in range(1, 6):
        traffic_gen.step()
        # Find edges with non-zero congestion
        affected = np.where(traffic_gen.congestion > 0.01)[0]
        max_val = np.max(traffic_gen.congestion)
        print(f"Step {t}: {len(affected)} edges affected. Max congestion: {max_val:.4f}")
        # Check max propagation distance
        # To do this rigorously, we'd check shortest path distances in the line graph,
        # but just seeing how fast it spreads is a good proxy.

if __name__ == "__main__":
    analyze_time_to_impact(4, 4)
    analyze_time_to_impact(6, 6)
    analyze_time_to_impact(8, 8)
