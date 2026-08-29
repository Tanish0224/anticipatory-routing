import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utilities.graph_utils import create_grid_network
from src.environment.traffic_generator import TrafficGenerator

def test_traffic():
    graph = create_grid_network(rows=4, cols=4, seed=42)
    rhos = [0.0, 0.2, 0.5, 0.8, 1.0]
    steps = 100
    
    os.makedirs('figures', exist_ok=True)
    
    plt.figure(figsize=(15, 8))
    
    for idx, rho in enumerate(rhos):
        gen = TrafficGenerator(graph, regime=2, rho=rho)
        gen.reset(np.random.default_rng(42))
        
        history = []
        for _ in range(steps):
            history.append(gen.step())
            
        history = np.array(history) # shape: (steps, num_edges)
        
        # Plot edge 0 over time
        plt.subplot(2, 3, idx+1)
        plt.plot(history[:, 0], label='Edge 0')
        plt.plot(history[:, 1], label='Edge 1')
        plt.title(f'Traffic over time (rho={rho})')
        plt.xlabel('Step')
        plt.ylabel('Edge Cost')
        
    plt.tight_layout()
    plt.savefig('figures/traffic_validation.png')
    plt.close()
    
    # Calculate Auto-correlation for lag=1
    print("Traffic Autocorrelation (Lag=1) vs Rho:")
    for rho in rhos:
        gen = TrafficGenerator(graph, regime=2, rho=rho)
        gen.reset(np.random.default_rng(42))
        
        history = []
        for _ in range(1000):
            history.append(gen.step())
            
        history = np.array(history)
        
        autocorr_sum = 0
        for e in range(history.shape[1]):
            series = history[:, e]
            if np.std(series) > 1e-6:
                autocorr_sum += np.corrcoef(series[:-1], series[1:])[0,1]
        
        print(f"  Rho = {rho:.1f} -> Autocorrelation = {autocorr_sum / history.shape[1]:.3f}")

if __name__ == '__main__':
    test_traffic()
