import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import json
from src.utilities.graph_utils import create_grid_network
from src.environment.traffic_generator import TrafficGenerator
from src.environment.network_env import NetworkRoutingEnv
from src.agents.dqn_agent import DQNAgent
from src.algorithms.baselines import ReactiveDijkstra, PredictiveDijkstra
from src.training.trainer import train_dqn
from src.evaluation.evaluator import evaluate_agents

def run_regime(regime, rho=0.0):
    print(f"\n{'='*50}")
    print(f"Running Experiment - Regime {regime} (rho={rho})")
    print(f"{'='*50}")
    
    # 1. Setup Environment
    # 4x4 grid with diagonals to keep state space small for rapid training
    graph = create_grid_network(rows=4, cols=4, seed=42) 
    
    traffic_gen = TrafficGenerator(graph, regime=regime, rho=rho, noise_std=0.2, base_multiplier=3.0)
    env = NetworkRoutingEnv(graph, traffic_gen, max_steps=30)
    
    # 2. Setup Agents
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print("Initializing RL Agent...")
    dqn_agent = DQNAgent(obs_dim, action_dim, env, lr=2e-4, buffer_size=100000, batch_size=128, epsilon_decay=0.995, target_update_freq=500)
    
    reactive_dijkstra = ReactiveDijkstra(env)
    predictive_dijkstra = PredictiveDijkstra(env, alpha=0.3)
    
    # 3. Train DQN
    print("Training DQN...")
    # 2500 episodes for a stable convergence with history
    train_dqn(env, dqn_agent, num_episodes=2500, max_steps=30)
    
    # 4. Evaluate
    print("Evaluating Agents...")
    eval_seeds = list(range(1000, 1100)) # 100 evaluation seeds
    
    agents = {
        'Reactive_Dijkstra': reactive_dijkstra,
        'Predictive_Dijkstra': predictive_dijkstra,
        'DQN_RL': dqn_agent
    }
    
    summary, raw_results = evaluate_agents(env, agents, eval_seeds, max_steps=30)
    
    for name, metrics in summary.items():
        print(f"{name}:")
        print(f"  Success Rate : {metrics['success_rate']*100:.1f}%")
        print(f"  Mean Reward  : {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Mean Steps   : {metrics['mean_steps']:.2f}")
        
    return summary

if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    all_summaries = {}
    
    # Regime 0: Static
    all_summaries['Regime_0'] = run_regime(regime=0, rho=0.0)
    
    # Regime 1: Independent Noise
    all_summaries['Regime_1'] = run_regime(regime=1, rho=0.0)
    
    # Regime 2: Correlated (rho=0.8)
    all_summaries['Regime_2_rho08'] = run_regime(regime=2, rho=0.8)
    
    # Save results
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(all_summaries, f, indent=4)
