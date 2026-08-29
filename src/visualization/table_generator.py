import json
import os

def generate_markdown_table(results_path='results/summary.json', output_path='docs/results_table.md'):
    if not os.path.exists(results_path):
        return
        
    with open(results_path, 'r') as f:
        data = json.load(f)
        
    regimes = list(data.keys())
    if not regimes:
        return
        
    agents = list(data[regimes[0]].keys())
    
    with open(output_path, 'w') as f:
        f.write("# Final Experimental Results\n\n")
        f.write("This table summarizes the mean reward (negative delay + success bonus) and success rate for each algorithm across different environmental regimes.\n\n")
        
        f.write("| Regime | Algorithm | Mean Reward | Std Dev | Success Rate | Mean Steps |\n")
        f.write("|--------|-----------|-------------|---------|--------------|------------|\n")
        
        for regime in regimes:
            for agent in agents:
                metrics = data[regime][agent]
                reward = f"{metrics['mean_reward']:.2f}"
                std = f"{metrics['std_reward']:.2f}"
                success = f"{metrics['success_rate']*100:.1f}%"
                steps = f"{metrics['mean_steps']:.2f}"
                
                f.write(f"| {regime} | {agent} | {reward} | {std} | {success} | {steps} |\n")

if __name__ == "__main__":
    generate_markdown_table()
