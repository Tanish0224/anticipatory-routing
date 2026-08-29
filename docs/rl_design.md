# RL Agent Design

## 1. Algorithm: Double DQN
We utilize Double Deep Q-Network (Double DQN) to learn the routing policy. 
**Why not standard DQN?** Standard DQN suffers from maximization bias, causing it to systematically overestimate Q-values. In a stochastic routing environment where edge weights fluctuate, overestimating the value of a temporarily "lucky" fast route will destroy the policy. Double DQN decouples action selection from action evaluation to stabilize learning.
**Why not PPO or Actor-Critic?** The action space (next adjacent node) is heavily constrained and state-dependent. While PPO can handle action masking, DQN's off-policy nature allows for massive sample efficiency gains via the Replay Buffer, which is critical given the expensive stochastic environment.

## 2. Architecture and State Representation
The true state of the environment is a fully observable MDP where current edge weights map bijectively to congestion. However, to provide the neural network with explicit finite-difference features (momentum/velocity) without requiring it to infer the environmental transition matrix $P$ from scratch, we use a sliding window of history ($H=3$).

**Input Dimension:** $2N + H \times E$
- One-hot encoding of current node ($N$)
- One-hot encoding of destination node ($N$)
- Flattened edge weights for time steps $[t-2, t-1, t]$ ($3E$)

For the 10x10 grid ($N=100, E=342$), the input size is 1226.
**Network:** MLP with architecture `1226 -> 512 -> 512 -> 100`.

## 3. Structural Action Masking
The agent outputs Q-values for all 100 nodes in the graph, but it can only legally move to adjacent neighbors.
During **inference**, invalid actions are masked with $-\infty$ before the `argmax` operation.
Critically, during **target calculation** in the Bellman update, invalid actions are also masked:
$$Y = R + \gamma \max_{a' \in legal(S')} Q_{target}(S', a')$$
Failing to mask the target network is a common RL failure mode that causes the agent to hallucinate high values for impossible teleportation actions.

## 4. Reward Formulation
The environment returns a dense step penalty:
$$R_t = -W_t(u, v)$$
Where $W_t$ is the travel time of the chosen edge. The episode terminates upon reaching the destination. The cumulative return exactly equals the negative total travel time, aligning the RL objective perfectly with the classical shortest-path objective. No artificial terminal rewards are used, preventing reward hacking.
