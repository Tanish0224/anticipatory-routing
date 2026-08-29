import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from collections import deque
from .traffic_generator import TrafficGenerator

class NetworkRoutingEnv(gym.Env):
    """
    A dynamic routing environment where an agent navigates a graph from a start 
    node to a destination node while edge weights fluctuate over time.
    """
    def __init__(self, graph: nx.DiGraph, traffic_generator: TrafficGenerator, max_steps: int = 100, history_length: int = 3):
        super().__init__()
        self.graph = graph
        self.traffic_gen = traffic_generator
        self.max_steps = max_steps
        self.history_length = history_length
        
        self.num_nodes = graph.number_of_nodes()
        self.nodes = list(graph.nodes())
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        
        self.num_edges = graph.number_of_edges()
        
        # Action space: select the next node index to travel to.
        self.action_space = spaces.Discrete(self.num_nodes)
        
        # Observation space: 
        # [current_node_one_hot (N), dest_node_one_hot (N), edge_weights * history_length (E * H)]
        obs_dim = 2 * self.num_nodes + (self.num_edges * self.history_length)
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.current_node = None
        self.dest_node = None
        self.step_count = 0
        self.weight_history = deque(maxlen=self.history_length)
        
    def _get_obs(self):
        curr_one_hot = np.zeros(self.num_nodes, dtype=np.float32)
        if self.current_node is not None:
            curr_one_hot[self.node_to_idx[self.current_node]] = 1.0
            
        dest_one_hot = np.zeros(self.num_nodes, dtype=np.float32)
        if self.dest_node is not None:
            dest_one_hot[self.node_to_idx[self.dest_node]] = 1.0
            
        # Flatten the history
        history_flat = np.concatenate(self.weight_history) if len(self.weight_history) > 0 else np.zeros(self.num_edges * self.history_length, dtype=np.float32)
        
        return np.concatenate([curr_one_hot, dest_one_hot, history_flat])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset traffic generator (will use self.np_random initialized by gym)
        self.traffic_gen.reset(self.np_random)
        
        # Populate history with initial state
        initial_weights = np.array([self.graph.edges[e]['weight'] for e in self.traffic_gen.edges], dtype=np.float32)
        for _ in range(self.history_length):
            self.weight_history.append(initial_weights)
        
        # Randomly select distinct start and destination nodes
        self.current_node, self.dest_node = self.np_random.choice(self.nodes, 2, replace=False)
        self.step_count = 0
        
        return self._get_obs(), {}

    def step(self, action: int):
        self.step_count += 1
        
        target_node = self.nodes[action]
        
        # Check if the action is a valid outgoing neighbor
        neighbors = list(self.graph.successors(self.current_node))
        
        reward = 0.0
        terminated = False
        truncated = False
        
        if target_node in neighbors:
            # Valid move
            edge = (self.current_node, target_node)
            cost = self.graph.edges[edge]['weight']
            reward = -cost
            self.current_node = target_node
            
            if self.current_node == self.dest_node:
                reward += 100.0  # Big bonus for reaching the destination
                terminated = True
        else:
            # Invalid move - penalize heavily and stay in place
            reward = -50.0
            
        # Advance environment dynamics
        if not terminated:
            self.traffic_gen.step()
            
            # Record new weights in history
            new_weights = np.array([self.graph.edges[e]['weight'] for e in self.traffic_gen.edges], dtype=np.float32)
            self.weight_history.append(new_weights)
            
        if self.step_count >= self.max_steps:
            truncated = True
            
        return self._get_obs(), reward, terminated, truncated, {}
        
    def get_valid_actions(self):
        """Helper for baselines or action masking"""
        neighbors = list(self.graph.successors(self.current_node))
        return [self.node_to_idx[n] for n in neighbors]
