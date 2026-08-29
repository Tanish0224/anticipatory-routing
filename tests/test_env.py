import unittest
import numpy as np
import networkx as nx
from src.utilities.graph_utils import create_grid_network
from src.environment.traffic_generator import TrafficGenerator
from src.environment.network_env import NetworkRoutingEnv

class TestNetworkEnv(unittest.TestCase):
    def setUp(self):
        self.graph = create_grid_network(rows=3, cols=3, seed=42)
        self.traffic_gen = TrafficGenerator(self.graph, regime=0)
        self.env = NetworkRoutingEnv(self.graph, self.traffic_gen, max_steps=20)
        
    def test_environment_reset(self):
        obs, _ = self.env.reset()
        # obs = [curr_node (9), dest_node (9), edges*3 (approx 20*3)]
        self.assertEqual(len(obs), 2 * 9 + self.graph.number_of_edges() * 3)
        self.assertNotEqual(self.env.current_node, self.env.dest_node)
        
    def test_valid_action_progress(self):
        self.env.reset()
        valid_actions = self.env.get_valid_actions()
        self.assertTrue(len(valid_actions) > 0)
        
        # Take a valid action
        action = valid_actions[0]
        _, reward, terminated, truncated, _ = self.env.step(action)
        
        # Reward should be negative cost (or positive 100 if reached dest)
        self.assertTrue(reward <= 0 or reward > 50)
        
    def test_invalid_action_penalty(self):
        self.env.reset()
        valid_actions = self.env.get_valid_actions()
        invalid_action = [i for i in range(9) if i not in valid_actions][0]
        
        _, reward, _, _, _ = self.env.step(invalid_action)
        self.assertEqual(reward, -50.0) # Invalid move penalty

if __name__ == '__main__':
    unittest.main()
