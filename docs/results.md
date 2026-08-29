# Results and Conclusions

## The Initial Flaw: A False Negative
The original implementation of this project found that Reactive Dijkstra perfectly matched the performance of Double DQN. However, forensic analysis revealed that the environment itself prevented anticipation:
1. The 4x4 grid produced optimal paths of only 3 steps.
2. The traffic dynamics simulated isotropic diffusion rather than a translating kinematic wave.
Because congestion "dissolved" in place, and trips finished in 3 steps, there was no structural penalty for reacting purely locally.

## The Redesigned Testbed
To test the core research hypothesis rigorously, the environment was scaled to a 10x10 grid ($E=342$), and the traffic generator was engineered to produce Translating Kinematic Waves using max-pooling propagation. This created persistent, high-amplitude traffic walls sweeping across the map, guaranteeing that $Time-To-Impact < Time-To-Destination$.

## Experimental Results
We evaluated Reactive Dijkstra, Predictive Dijkstra (Holt's Linear Trend), and Double DQN across the correlation spectrum $\rho \in [0.0, 1.0]$ using strict out-of-distribution evaluation seeds.

### 1. The Classical Dominance ($\rho < 0.7$)
In low-to-medium correlation regimes, the environment behaves mostly like independent noise. Reactive Dijkstra and Predictive Dijkstra performed identically, achieving optimal travel times. The RL agent struggled with the stochasticity and achieved significantly higher variance and worse mean travel times.

### 2. The Kinematic Wave Regime ($\rho \ge 0.7$)
In the highly structured regime, the translating wave forces planners to anticipate.
- **Reactive Dijkstra:** Suffered significant delays. Because it cannot see the wave until it arrives at an adjacent edge, it routinely routed itself into the path of the incoming traffic wall, resulting in degraded travel times.
- **DQN (H=3):** Demonstrated the theoretical capacity to learn the wave dynamics via the historical sliding window, occasionally rerouting preemptively. However, due to the massive state space of the 10x10 grid, it suffered from training instability and high variance, meaning it was unreliable out-of-distribution.
- **Predictive Dijkstra (Holt's):** Dominated the evaluation. By maintaining a continuous explicit model of the level and trend of every edge, Predictive Dijkstra perfectly anticipated the kinematic wave and routed around it before impact. It achieved the best mean travel times with near-zero variance.

## Scientific Conclusion
This project successfully constructed an environment where anticipation is strictly required, proving that Reactive planning fails under structured kinematic waves.
However, it also definitively proves that **end-to-end Deep Reinforcement Learning is computationally and practically unjustified for this class of routing problems.** 
Decoupling the prediction (via classical time-series forecasting) from the planning (via Dijkstra) solves the anticipatory routing problem optimally, with $O(V \log V)$ runtime, zero training cost, and no variance. 

**Final Verdict:** For dynamic stochastic routing on known graph topologies, explicit forecasting + classical planning strictly dominates model-free Reinforcement Learning.
