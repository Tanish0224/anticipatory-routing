import sys
import os
import time
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utilities.graph_utils import create_grid_network
from src.environment.traffic_generator import TrafficGenerator
from src.environment.network_env import NetworkRoutingEnv
from src.agents.dqn_agent import DQNAgent
from src.algorithms.baselines import ReactiveDijkstra, PredictiveDijkstra

def benchmark_inference():
    rows, cols = 10, 10
    num_decisions = 1000
    
    print(f"--- Inference Latency Benchmark ({num_decisions} decisions) ---")
    
    graph = create_grid_network(rows=rows, cols=cols, seed=42)
    traffic_gen = TrafficGenerator(graph, regime=3, rho=0.9)
    env = NetworkRoutingEnv(graph, traffic_gen, history_length=3)
    
    reactive = ReactiveDijkstra(env)
    predictive = PredictiveDijkstra(env)
    dqn = DQNAgent(env.observation_space.shape[0], env.action_space.n, env)
    
    state, _ = env.reset(seed=42)
    
    # Warmup torch
    for _ in range(10):
        dqn.act(state, eval_mode=True)
        
    def measure(agent, name):
        start = time.perf_counter()
        for _ in range(num_decisions):
            if name.startswith('DQN'):
                agent.act(state, eval_mode=True)
            else:
                agent.act(state)
        end = time.perf_counter()
        total_time = end - start
        avg_ms = (total_time / num_decisions) * 1000
        print(f"{name:20s}: {avg_ms:.3f} ms / decision")
        
    measure(reactive, "Reactive Dijkstra")
    measure(predictive, "Holt + Dijkstra")
    measure(dqn, "Double DQN (H=3)")

if __name__ == "__main__":
    benchmark_inference()
