# Experimental Protocol

## 1. The Correlation Sweep
To answer the core research question, we evaluate the agents across a continuum of environmental predictability by sweeping the spatial correlation parameter $\rho \in [0.0, 1.0]$.
- **Low $\rho$ (0.0 - 0.3):** Dominated by independent Gaussian noise. Anticipation is theoretically impossible because future states are uncorrelated with current states. Reactive routing should be optimal.
- **High $\rho$ (0.7 - 1.0):** Dominated by the translating kinematic wave. A structured wall of traffic moves directionally across the grid. Anticipatory routing is theoretically possible.

## 2. Generalization and Data Leakage Prevention
To ensure the RL agent learns genuine routing principles rather than memorizing specific traffic realizations, we strictly separate training and evaluation seeds.
- **Training:** The DQN agent trains for thousands of episodes, with the environment seeded dynamically to produce unique traffic scenarios.
- **Evaluation:** All agents (Reactive, Predictive, and DQN) are evaluated on a fixed set of $N=100$ independent unseen random seeds. This guarantees strict Out-of-Distribution (OOD) testing.

## 3. The Temporal Ablation
We ablate the RL agent's observation history to isolate the value of temporal information:
- **DQN (H=1):** Observes only the current global snapshot $W_t$.
- **DQN (H=3):** Observes a sliding window $[W_{t-2}, W_{t-1}, W_t]$.
If $H=3$ outperforms $H=1$ only in the high-$\rho$ regime, it proves the network is actively exploiting the temporal structure of the kinematic wave.
