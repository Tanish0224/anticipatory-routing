# Anticipatory Routing on Dynamic Graphs: A Comparison of Temporal Forecasting and Model-Free Reinforcement Learning

## Abstract
Dynamic routing on networks requires an agent to find the optimal path under dynamically evolving edge costs. A common approach decouples the problem: predict future traffic using time-series forecasting, then run a classical planning algorithm over the predicted weights. Alternatively, end-to-end Model-Free Deep Reinforcement Learning (RL) might bypass compounding prediction errors by implicitly anticipating traffic dynamics. We evaluate these paradigms in a simulated stochastic environment featuring translating kinematic congestion waves. Under strict information symmetry, we benchmark Reactive Dijkstra, Predictive Dijkstra (using Holt's Linear Trend), and Double DQN (with and without historical observations). The experiments indicate that while temporal history improves model-free routing performance under translating-wave conditions, explicit forecasting coupled with classical shortest-path planning substantially outperforms the model-free RL agents in both sample efficiency and variance reduction.

## Research Question
*Under what temporal-correlation conditions does model-free reinforcement learning provide a meaningful advantage over explicit forecasting combined with classical planning for dynamic graph routing?*

## Problem Formulation
We define the routing environment as a directed graph $G = (V, E)$ where $|V|=100$ and $|E|=342$ (a $10 \times 10$ grid with diagonal cross-edges). The dynamic edge cost at time $t$ is $c_e(t) = W_{base}(e)(1 + \beta C_e(t))$, where $C_t \in [0, 1]^{|E|}$ represents the global traffic congestion state. The objective is to minimize the expected cumulative travel time to a target destination.

## Environment Dynamics
Traffic evolves via a discrete translating kinematic-wave abstraction:
$C_{propagation}^{(e)} = \max_{j \in U(e)} C_t^{(j)}$
$C_{t+1}^{(e)} = (1 - S) C_t^{(e)} + S \cdot C_{propagation}^{(e)} + \xi_t^{(e)}$
where $\rho$ controls the spatiotemporal correlation of $\xi_t$. Unlike isotropic diffusion, this directional max-pooling formulation preserves wave amplitude, enforcing an anticipation condition where the wave propagation time is strictly less than the route traversal time ($k < m$).

## Methods
We evaluate the following approaches under strict information symmetry (observing only historical and current edge costs):
1. **Reactive Dijkstra**: Replans dynamically using only current observable costs.
2. **Predictive Dijkstra**: Decoupled forecasting (Double Exponential Smoothing) and planning.
3. **Double DQN (H=1)**: Model-free RL observing only the current state snapshot.
4. **Double DQN (H=3)**: Model-free RL observing a 3-step historical sliding window.
5. **Hindsight Oracle**: Dynamic Programming baseline with access to future traffic states to establish an optimal reference lower bound for travel cost.

## Experimental Protocol
The experimental sweep evaluates performance across five correlation regimes ($\rho \in \{0.0, 0.3, 0.6, 0.9, 1.0\}$). Each RL configuration is trained for 1500 episodes. All methods are then evaluated across 100 fixed, independent, Out-Of-Distribution (OOD) random seeds.

## Results
The experiments demonstrated that at high spatiotemporal correlation ($\rho = 0.9$):
- **Reactive Dijkstra** perfectly solved the deterministic topology but suffered when routing directly into the path of incoming kinematic waves.
- **DQN (H=3)** achieved higher performance than the **DQN (H=1)** ablation, consistent with the hypothesis that temporal information is useful for anticipating the propagating disturbance.
- **Predictive Dijkstra** achieved near-optimal travel times relative to the Hindsight Oracle, significantly outperforming both RL methods.

## Key Findings
For the studied problem class—known graph topology, observable congestion state, and discrete translating-wave dynamics—explicit temporal forecasting combined with classical planning substantially outperformed end-to-end model-free reinforcement learning. These findings suggest that when congestion dynamics exhibit exploitable temporal structure, separating prediction from planning provides a strong alternative to end-to-end learning.

## Limitations
- **Synthetic Environment**: The discrete translating-wave model is a computational abstraction, not a calibrated physical traffic-flow simulation.
- **Topology**: The graph is a uniform grid, lacking the complexities of real-world intersections and capacities.
- **Scalability**: The flattened $O(|E|)$ observation space for the Double DQN architecture scales poorly to massive city-level networks without the adoption of Graph Neural Networks (GNNs).

## Reproducibility
The full experimental pipeline, including environment validation, inference benchmarking, training, evaluation, and statistical plotting, can be reproduced automatically:

```bash
pip install -r requirements.txt
python experiments/run_final_experiment.py
```
Results and summary files are saved to the `results/` directory, while figures are output to `figures/`.

## Repository Structure
- `src/`: Core implementation containing agents, classical algorithms, environment dynamics, and evaluation tools.
- `experiments/`: Scripts for executing sweeps, testing configurations, and running the final experimental pipeline.
- `docs/`: Scientific methodology, environment definitions, and information symmetry constraints.
- `tests/`: Unit tests for graph logic, masking, and environment dynamics.
- `results/`: Empirical JSON outputs from evaluation runs.
- `figures/`: Data visualizations.

## Citation
A formal citation will be added upon manuscript/publication availability.
