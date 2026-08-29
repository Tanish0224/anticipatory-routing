# Limitations

Every scientific investigation must acknowledge its bounds. The following are the explicit limitations of this project:

## 1. Simplified Grid Topology
While the environment utilizes a 10x10 directed grid with cross-edges, real-world road networks (e.g., Manhattan or London) feature irregular geometries, varied speed limits, intersection bottlenecks, and non-uniform branching factors. The uniform grid topology is sufficient for testing the theoretical capacity of RL to anticipate waves, but it does not capture the topological complexity of a real city.

## 2. Discrete Wave Mechanics
The `TrafficGenerator` (Regime 3) simulates translating kinematic waves using a discrete, synchronous max-pooling update. While this successfully creates a persistent "wall" of traffic that sweeps across the map, real traffic waves (as described by Lighthill-Whitham-Richards theory) exhibit continuous shockwave formation, backward propagation (traffic jams propagate backwards relative to traffic flow), and capacity-drop hysteresis. Our model abstracts these micro-kinematics into a macro-level translating penalty.

## 3. The Stationary Distribution Assumption
The evaluation sweeps over the correlation parameter $\rho$, but within any single episode, $\rho$ is stationary. A true real-world environment exhibits *non-stationary* dynamics (e.g., sudden accidents, weather changes, or rush hour onset) where the correlation structure itself shifts unpredictably. The agent was not tested under non-stationary distribution shifts mid-episode.

## 4. Scalability of the RL Observation Space
The current `DQNAgent` accepts a flattened array of all edge weights. For a 10x10 grid ($E=342$), the input size is manageable. However, this dense representation scales at $O(E)$, making it fundamentally unscalable to a city with millions of edges. A production system would require local egocentric observations or a Graph Neural Network (GNN) to achieve resolution invariance. We deliberately avoided GNNs to isolate the temporal anticipation question without introducing confounding architectural complexity.

## 5. Extrapolating the Negative Result
For the problem class and simulator studied here, explicit forecasting combined with classical planning was more effective than end-to-end model-free RL. This negative result for RL should not be generalized to all dynamic routing problems, particularly those where traffic dynamics are highly non-linear or where exact graph topology is unobservable.

## 6. Baseline and Evaluation Constraints
Only Double DQN was investigated as the model-free agent, and only Holt's Linear Trend was considered for the classical forecasting baseline. Holt's model assumes sufficiently smooth temporal structure, and its forecasting quality depends strictly on the traffic regime. Evaluation was restricted to a finite training budget, a fixed set of correlation values, and a fixed set of random seeds. Consequently, these results do not establish universal superiority of classical methods across all forecasting algorithms or RL architectures.
