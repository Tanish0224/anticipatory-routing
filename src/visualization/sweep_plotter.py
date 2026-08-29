import json
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_sweep_results(results_path='results/sweep_results.json', output_dir='figures'):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(results_path):
        print(f"Results file {results_path} not found.")
        return
        
    with open(results_path, 'r') as f:
        data = json.load(f)
        
    rhos = [float(r.split('_')[1]) for r in data.keys()]
    agents = list(data[list(data.keys())[0]].keys())
    
    # 1. Performance (Routing Cost) vs Correlation
    plt.figure(figsize=(10, 6))
    
    colors = {
        'Reactive_Dijkstra': 'black',
        'Predictive_Dijkstra': 'gray',
        'DQN_H1': 'red',
        'DQN_H3': 'blue'
    }
    
    markers = {
        'Reactive_Dijkstra': 's',
        'Predictive_Dijkstra': '^',
        'DQN_H1': 'x',
        'DQN_H3': 'o'
    }
    
    for agent in agents:
        # Negative reward represents total routing delay
        delays = [-data[f"rho_{rho}"][agent]['mean_reward'] for rho in rhos]
        std_devs = [data[f"rho_{rho}"][agent]['std_reward'] for rho in rhos]
        
        plt.errorbar(rhos, delays, yerr=std_devs, label=agent.replace('_', ' '), 
                     color=colors.get(agent, 'green'), marker=markers.get(agent, 'o'), 
                     capsize=5, alpha=0.8, linewidth=2)
                     
    plt.title('Routing Delay vs. Spatial-Temporal Correlation (\u03C1)', fontsize=14)
    plt.xlabel('Correlation Strength (\u03C1)', fontsize=12)
    plt.ylabel('Mean Routing Delay (Lower is Better)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add a vertical shaded region showing the "Crossover Region"
    plt.axvspan(0.4, 0.8, color='yellow', alpha=0.1, label='Crossover Hypothesis Region')
    
    plt.savefig(os.path.join(output_dir, 'correlation_sweep.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Sweep plots saved to {output_dir}/correlation_sweep.png")

if __name__ == '__main__':
    plot_sweep_results()
