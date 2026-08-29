import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utilities.graph_utils import create_grid_network
from src.environment.traffic_generator import TrafficGenerator
from src.environment.network_env import NetworkRoutingEnv
from src.agents.dqn_agent import DQNAgent
from src.training.trainer import train_dqn

print("Running Pilot Experiment (Phase 7) - Sanity Check")

graph = create_grid_network(rows=4, cols=4, seed=42)
traffic_gen = TrafficGenerator(graph, regime=0) # Static
env = NetworkRoutingEnv(graph, traffic_gen, max_steps=30, history_length=3)

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(obs_dim, action_dim, env, lr=2e-4, buffer_size=10000, batch_size=64, epsilon_decay=0.98, target_update_freq=100)

print("Training for 200 episodes...")
history = train_dqn(env, agent, num_episodes=200, max_steps=30)
print(f"Final Average Reward (Last 10): {sum(history[-10:])/10}")
print("Pilot test completed successfully. Environment runs without crashing.")
