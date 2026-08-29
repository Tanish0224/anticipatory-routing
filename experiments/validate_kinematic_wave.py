import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utilities.graph_utils import create_grid_network
from src.environment.traffic_generator import TrafficGenerator

def track_wave():
    rows, cols = 10, 10
    rhos = [0.0, 0.5, 0.9, 1.0]
    timesteps = 15
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, rho in enumerate(rhos):
        G = create_grid_network(rows, cols, seed=42)
        # Generate with zero noise to strictly track the structural wave
        gen = TrafficGenerator(G, regime=3, rho=rho, noise_std=0.0)
        
        gen.reset(np.random.RandomState(42))
        
        # Inject shock on the West boundary (column 0 -> 1)
        west_edges = [i for i, (u, v) in enumerate(gen.edges) if G.nodes[u]['pos'][1] == 0 and G.nodes[v]['pos'][1] == 1]
        
        # We inject a shock at the center of the west wall
        shock_edge = west_edges[len(west_edges)//2]
        gen.congestion[shock_edge] = 1.0
        
        center_of_mass_cols = []
        peak_amplitudes = []
        
        for t in range(timesteps):
            # Calculate Center of Mass (column)
            total_mass = np.sum(gen.congestion)
            if total_mass > 0:
                com_col = np.sum([gen.congestion[i] * G.nodes[gen.edges[i][1]]['pos'][1] for i in range(gen.num_edges)]) / total_mass
            else:
                com_col = 0.0
            
            peak_amp = np.max(gen.congestion)
            
            center_of_mass_cols.append(com_col)
            peak_amplitudes.append(peak_amp)
            
            gen.step()
            
        ax = axes[idx]
        ax.plot(range(timesteps), center_of_mass_cols, label='Wave Center of Mass (Column)', color='blue', marker='o')
        ax.set_ylabel('Grid Column', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_ylim(0, 10)
        
        ax2 = ax.twinx()
        ax2.plot(range(timesteps), peak_amplitudes, label='Peak Amplitude', color='red', marker='x', linestyle='--')
        ax2.set_ylabel('Congestion Amplitude', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 1.1)
        
        ax.set_title(rf'Wave Propagation ($\rho$ = {rho})')
        ax.set_xlabel('Time Step')
        ax.grid(True)
        
    plt.tight_layout()
    plt.savefig('figures/kinematic_wave_validation.png')
    print("Wave validation complete. Saved to figures/kinematic_wave_validation.png")

if __name__ == "__main__":
    track_wave()
