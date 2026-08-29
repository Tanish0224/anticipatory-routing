import json
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_results(results_path='results/summary.json', output_dir='figures'):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(results_path, 'r') as f:
        data = json.load(f)
        
    regimes = list(data.keys())
    agents = list(data[regimes[0]].keys())
    
    # 1. Success Rate Bar Chart
    plt.figure(figsize=(10, 6))
    x = np.arange(len(regimes))
    width = 0.25
    
    for i, agent in enumerate(agents):
        success_rates = [data[reg][agent]['success_rate'] * 100 for reg in regimes]
        plt.bar(x + (i - 1)*width, success_rates, width, label=agent.replace('_', ' '))
        
    plt.title('Success Rate by Regime')
    plt.xlabel('Traffic Regime')
    plt.ylabel('Success Rate (%)')
    plt.xticks(x, [r.replace('_', ' ') for r in regimes])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'success_rates.png'))
    plt.close()
    
    # 2. Mean Delay (Negative Reward) Bar Chart
    # Note: Using negative reward as positive delay for interpretability
    plt.figure(figsize=(10, 6))
    for i, agent in enumerate(agents):
        # We cap the negative reward so massive failure penalties (-1000) don't squash the chart
        # Only include successful delays, but since summary averages all, we plot raw mean reward
        mean_delays = [-data[reg][agent]['mean_reward'] for reg in regimes]
        plt.bar(x + (i - 1)*width, mean_delays, width, label=agent.replace('_', ' '))
        
    plt.title('Average Routing Cost (Delay) by Regime')
    plt.xlabel('Traffic Regime')
    plt.ylabel('Cost (Lower is better)')
    plt.xticks(x, [r.replace('_', ' ') for r in regimes])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'routing_costs.png'))
    plt.close()

if __name__ == '__main__':
    plot_results()
