# Information Symmetry Contract

To guarantee a scientifically fair comparison between Reactive planning, Predictive planning, and Reinforcement Learning, this project enforces strict information symmetry. At no point does the RL agent receive privileged access to hidden global traffic variables that the classical baselines cannot also observe or infer.

## The Environment State
The underlying environment state is the global traffic congestion vector $C_t \in [0, 1]^E$.
The observable edge weights are $W_t = W_{base} (1 + \beta C_t)$.

## The Symmetry Matrix

| Method | Current Observation | Historical Observation | Forecast Horizon | Latent Information Access |
|--------|---------------------|------------------------|-------------------|--------------------------|
| **Reactive Dijkstra** | $W_t$ (Full Graph) | None | 0 steps | None |
| **Predictive Dijkstra (Holt's)** | $W_t$ (Full Graph) | $W_0 \dots W_t$ (Full continuous history) | End of route | None |
| **Double DQN (H=1)** | $W_t$ (Full Graph) | None | Implicit via Q-values | None |
| **Double DQN (H=3)** | $W_t$ (Full Graph) | $W_{t-2}, W_{t-1}, W_t$ (Sliding window) | Implicit via Q-values | None |

## Fairness Justification
1. **No Global PEEKING:** All agents receive the exact same array of edge weights $W_t$. The RL agent's observation space is explicitly flattened from this array.
2. **History Fairness:** The anticipatory RL agent (H=3) receives a sliding window of the last 3 time steps to compute finite-difference features. The Predictive Dijkstra baseline receives *more* temporal history, as Holt's Double Exponential Smoothing runs continuously from $t=0$, granting it the entire history to calculate trend and momentum.
3. **No Unobserved Dynamics:** Since $W_{base}$ and $\beta$ are static constants, $W_t$ maps bijectively to $C_t$. Therefore, both the neural network and the classical forecasting baseline have theoretically perfect information about the current congestion state. Neither has an unfair mathematical advantage.

This structural fairness guarantees that if one method outperforms the other, the advantage stems strictly from the algorithm's planning capability (or its ability to implicitly learn the true transition function $P$), rather than from asymmetric information access.
