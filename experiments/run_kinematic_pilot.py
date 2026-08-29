import sys
import os
import json
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utilities.graph_utils import create_grid_network
from src.environment.traffic_generator import TrafficGenerator
from src.environment.network_env import NetworkRoutingEnv
from src.agents.dqn_agent import DQNAgent
from src.algorithms.baselines import ReactiveDijkstra, PredictiveDijkstra
from src.training.trainer import train_dqn
from src.evaluation.evaluator import evaluate_agents

def run_kinematic_pilot():
    rows, cols = 10, 10
    print(f"--- 10x10 Kinematic Wave Pilot (Rho = 0.95) ---")
    max_steps = 50
    num_episodes = 1200
    rho = 0.95
    eval_seeds = list(range(1000, 1050)) # 50 seeds

    graph = create_grid_network(rows=rows, cols=cols, seed=42)
    # regime=3 is the new Kinematic Wave
    traffic_gen = TrafficGenerator(graph, regime=3, rho=rho, noise_std=0.05, base_multiplier=5.0)
    
    # We will test Reactive, Predictive, and DQN(H=3)
    env = NetworkRoutingEnv(graph, traffic_gen, max_steps=max_steps, history_length=3)
    
    # Increase network capacity for 10x10
    dqn = DQNAgent(env.observation_space.shape[0], env.action_space.n, env, 
                      lr=3e-4, buffer_size=200000, batch_size=128, epsilon_decay=0.998, target_update_freq=1000)
                      
    print(f"Training DQN (H=3) on {rows}x{cols} grid for {num_episodes} episodes...")
    start_t = time.time()
    train_dqn(env, dqn, num_episodes=num_episodes, max_steps=max_steps)
    end_t = time.time()
    print(f"Training took {end_t - start_t:.1f} seconds")
    
    print("Evaluating...")
    reactive = ReactiveDijkstra(env)
    # Give PredictiveDijkstra high alpha for fast adaptation
    predictive = PredictiveDijkstra(env, alpha=0.5, beta=0.1)
    
    agents = {
        'Reactive': reactive,
        'Predictive': predictive,
        'DQN_H3': dqn
    }
    summary, _ = evaluate_agents(env, agents, eval_seeds, max_steps=max_steps)
    
    for name, metrics in summary.items():
        print(f"  {name:15s}: Success {metrics['success_rate']*100:5.1f}% | Reward {metrics['mean_reward']:7.2f} +/- {metrics['std_reward']:5.2f} | Steps {metrics['mean_steps']:5.1f}")

if __name__ == "__main__":
    run_kinematic_pilot()
