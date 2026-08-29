import sys
import os
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utilities.graph_utils import create_grid_network
from src.environment.traffic_generator import TrafficGenerator
from src.environment.network_env import NetworkRoutingEnv
from src.agents.dqn_agent import DQNAgent
from src.algorithms.baselines import ReactiveDijkstra, PredictiveDijkstra
from src.algorithms.oracle import OracleDijkstra
from src.training.trainer import train_dqn
from src.evaluation.evaluator import evaluate_agents

def run_correlation_sweep():
    # Massive Scale and Translating Wave Settings
    rows, cols = 10, 10
    max_steps = 50
    num_episodes = 1500 # Training budget for DQN
    rhos = [0.0, 0.3, 0.6, 0.9, 1.0]
    eval_seeds = list(range(1000, 1100)) # 100 strict OOD test seeds
    
    results_summary = {}

    for rho in rhos:
        print(f"\n=======================================================")
        print(f"Starting Evaluation for Correlation Regime: Rho = {rho}")
        print(f"=======================================================\n")
        
        # 1. Initialize Graph and Traffic (Regime 3 = Kinematic Wave)
        graph = create_grid_network(rows=rows, cols=cols, seed=42)
        traffic_gen = TrafficGenerator(graph, regime=3, rho=rho, noise_std=0.05, base_multiplier=5.0)
        
        # 2. Environments
        env_H1 = NetworkRoutingEnv(graph, traffic_gen, max_steps=max_steps, history_length=1)
        env_H3 = NetworkRoutingEnv(graph, traffic_gen, max_steps=max_steps, history_length=3)
        
        # 3. Baselines
        reactive = ReactiveDijkstra(env_H1)
        # Holt's Predictive Dijkstra (High alpha/beta for fast wave tracking)
        predictive = PredictiveDijkstra(env_H1, alpha=0.5, beta=0.2)
        oracle = OracleDijkstra(env_H1)
        
        # 4. Train DQN (H=1) - Ablated
        print(f"--- Training DQN (H=1) for {num_episodes} episodes ---")
        dqn_h1 = DQNAgent(env_H1.observation_space.shape[0], env_H1.action_space.n, env_H1, 
                          lr=3e-4, buffer_size=200000, batch_size=128, epsilon_decay=0.997, target_update_freq=1000)
        train_dqn(env_H1, dqn_h1, num_episodes=num_episodes, max_steps=max_steps)
        
        # 5. Train DQN (H=3) - Anticipatory
        print(f"--- Training DQN (H=3) for {num_episodes} episodes ---")
        dqn_h3 = DQNAgent(env_H3.observation_space.shape[0], env_H3.action_space.n, env_H3, 
                          lr=3e-4, buffer_size=200000, batch_size=128, epsilon_decay=0.997, target_update_freq=1000)
        train_dqn(env_H3, dqn_h3, num_episodes=num_episodes, max_steps=max_steps)
        
        # 6. Evaluation
        print(f"\n--- Evaluating all agents on 100 independent seeds ---")
        agents = {
            'Reactive_Dijkstra': reactive,
            'Predictive_Dijkstra': predictive,
            'DQN_H1': dqn_h1,
            'DQN_H3': dqn_h3
        }
        
        # Because different agents need different envs (H=1 vs H=3), we pass both to a custom evaluation wrapper
        # Actually, evaluate_agents assumes a single env. We must evaluate them separately or modify the wrapper.
        
        summary_h1, _ = evaluate_agents(env_H1, {'Reactive_Dijkstra': reactive, 'Predictive_Dijkstra': predictive, 'Oracle_DP': oracle, 'DQN_H1': dqn_h1}, eval_seeds, max_steps)
        summary_h3, _ = evaluate_agents(env_H3, {'DQN_H3': dqn_h3}, eval_seeds, max_steps)
        
        merged_summary = {**summary_h1, **summary_h3}
        results_summary[rho] = merged_summary
        
        for name, metrics in merged_summary.items():
            print(f"  {name:20s}: Success {metrics['success_rate']*100:5.1f}% | Reward: {metrics['mean_reward']:6.2f} +/- {metrics['std_reward']:5.2f}")

    # Save Results
    os.makedirs('results', exist_ok=True)
    with open('results/kinematic_sweep_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=4)
        
    print("\nSweep Complete. Results saved to results/kinematic_sweep_summary.json")

if __name__ == "__main__":
    run_correlation_sweep()
