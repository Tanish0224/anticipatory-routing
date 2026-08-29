import unittest
from src.utilities.graph_utils import create_grid_network
from src.environment.traffic_generator import TrafficGenerator
from src.environment.network_env import NetworkRoutingEnv
from src.algorithms.baselines import ReactiveDijkstra, PredictiveDijkstra

class TestBaselines(unittest.TestCase):
    def setUp(self):
        self.graph = create_grid_network(rows=3, cols=3, seed=42)
        self.traffic_gen = TrafficGenerator(self.graph, regime=0) # Static
        self.env = NetworkRoutingEnv(self.graph, self.traffic_gen, max_steps=20)
        
    def test_reactive_dijkstra_static(self):
        # In a static environment, Dijkstra should perfectly solve the graph
        agent = ReactiveDijkstra(self.env)
        self.env.reset(seed=42)
        
        steps = 0
        terminated = False
        while not terminated and steps < 20:
            action = agent.act(None)
            _, _, terminated, _, _ = self.env.step(action)
            steps += 1
            
        self.assertTrue(terminated)
        self.assertEqual(self.env.current_node, self.env.dest_node)

if __name__ == '__main__':
    unittest.main()
