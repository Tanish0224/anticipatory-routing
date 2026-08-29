import sys
import os
import networkx as nx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utilities.graph_utils import create_grid_network

def validate_scale():
    scales = [4, 6, 8, 10, 12]
    
    print("--- Environment Scale Diagnostics ---")
    print(f"{'Scale':>8} | {'Nodes':>6} | {'Edges':>6} | {'Shortest Path (Hops)':>20} | {'Est. Impact (Hops)':>20}")
    print("-" * 75)
    
    for s in scales:
        G = create_grid_network(s, s, seed=42)
        source = 0
        dest = s * s - 1
        
        # Calculate topological shortest path (ignoring weights)
        try:
            shortest_path = nx.shortest_path(G, source=source, target=dest)
            m = len(shortest_path) - 1 # number of hops
        except nx.NetworkXNoPath:
            m = -1
            
        # Estimated time to impact
        # Wave travels from column 0 to column s-1
        # Wave speed is roughly 1 column per step in max-pooling Kinematic Wave
        # But actually, max-pooling over neighbors with diagonal cross-edges 
        # means it propagates 1 column per step.
        k = s - 1
        
        print(f"{s}x{s}:     | {len(G.nodes):>6} | {len(G.edges):>6} | {m:>20} | {k:>20}")

if __name__ == "__main__":
    validate_scale()
