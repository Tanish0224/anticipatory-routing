import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from .networks import QNetwork

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
        
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, obs_dim, action_dim, env, lr=1e-3, gamma=0.99, epsilon_start=1.0, 
                 epsilon_end=0.05, epsilon_decay=0.995, buffer_size=100000, 
                 batch_size=64, target_update_freq=1000):
        self.action_dim = action_dim
        self.env = env # Need env for action masking
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_network = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps_done = 0
        
        # Construct adjacency mask for masking target Q-values during learning
        self.num_nodes = env.num_nodes
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for i in range(self.num_nodes):
            node_name = env.nodes[i]
            neighbors = list(env.graph.successors(node_name))
            for n in neighbors:
                adj[i, env.node_to_idx[n]] = 1.0
        self.adj_mask = torch.FloatTensor(adj).to(self.device)
        
    def act(self, state, eval_mode=False):
        valid_actions = self.env.get_valid_actions()
        if not valid_actions:
            return 0 # Should not happen unless graph has isolated nodes
            
        if not eval_mode and random.random() < self.epsilon:
            return random.choice(valid_actions)
            
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t).squeeze(0).cpu().numpy()
            
            # Action masking: set invalid actions to -infinity
            masked_q_values = np.full(self.action_dim, -np.inf)
            masked_q_values[valid_actions] = q_values[valid_actions]
            
            return int(np.argmax(masked_q_values))
            
    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        self.steps_done += 1
        
        if len(self.memory) > self.batch_size:
            self._learn()
            
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
    def _learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q values
        q_values = self.q_network(states).gather(1, actions)
        
        # Target Q values (using Double DQN for better stability)
        with torch.no_grad():
            # Extract current node index from one-hot encoding in next_states
            next_nodes = next_states[:, :self.num_nodes].argmax(dim=1)
            batch_mask = self.adj_mask[next_nodes]
            
            # Select action with online network (masked)
            online_next_q = self.q_network(next_states)
            online_next_q = online_next_q.masked_fill(batch_mask == 0, -float('inf'))
            next_actions = online_next_q.argmax(1, keepdim=True)
            
            # Evaluate action with target network
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        loss = F.mse_loss(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
