# Research Question

## The Core Inquiry
*At what level of predictable spatiotemporal correlation does the anticipatory capability of Reinforcement Learning provide a verifiable advantage over decoupled forecasting-and-planning baselines in dynamic stochastic routing?*

## Background
The standard engineering approach to dynamic routing (e.g., GPS navigation) decouples prediction from planning:
1. Forecast future traffic states using time-series models.
2. Plan the optimal path using an algorithm like Dijkstra or A* on the forecasted weights.

While computationally efficient ($O(V \log V)$), this approach suffers from compounded prediction errors over long horizons. Model-free Deep Reinforcement Learning (RL) theoretically bypasses this by directly learning a routing policy that maximizes cumulative reward (minimizes total travel time), implicitly anticipating complex environmental dynamics.

## The Hypothesis
If an environment contains strong, structured, propagating kinematic traffic waves, a purely reactive planner will be "trapped" by incoming congestion because it only reacts to currently observable deterioration. An anticipatory RL agent should learn to route around the incoming wave before it arrives. 

## The Rigorous Threshold
To prove that model-free RL provides a distinct advantage, it must not only outperform a purely reactive planner, but also a predictive classical planner equipped with equivalent historical information. If a classical forecasting baseline (such as Holt's Linear Trend coupled with Dijkstra) achieves equal or better performance than RL, then the end-to-end learning approach is computationally and empirically unjustified for the studied problem class.
