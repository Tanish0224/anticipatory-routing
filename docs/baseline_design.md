# Baseline Design

To ensure a scientifically rigorous evaluation, this project implements a hierarchy of classical planning baselines. The goal is to aggressively challenge the RL agent: if a simpler method works, the complex method is rejected.

## 1. Reactive Dijkstra
**Mechanism:** At every step, the agent observes the current global edge weights $W_t$ and runs Dijkstra's algorithm to find the shortest path to the destination. It takes the first step of this path, then replans at $t+1$.
**Information Access:** Current edge weights only.
**Strengths:** Optimal for static or purely random (white noise) environments. Extremely fast ($O(V \log V)$).
**Weaknesses:** Completely blind to momentum. If a massive traffic wave is moving toward a critical corridor, Reactive Dijkstra will route directly into the corridor until the wave actually hits it.

## 2. Predictive Dijkstra (Holt's Linear Trend)
**Mechanism:** At every step, the agent updates a stateful Double Exponential Smoothing (Holt's) model for every edge in the graph. Holt's model tracks both the *level* (current congestion) and the *trend* (velocity/momentum) of the congestion.
$$L_t = \alpha W_t + (1-\alpha)(L_{t-1} + T_{t-1})$$
$$T_t = \beta (L_t - L_{t-1}) + (1-\beta) T_{t-1}$$
The planner then forecasts the edge weights $k$ steps into the future, assuming the agent will arrive at an edge $k$ steps from now. It runs Dijkstra on this spatiotemporally forecasted graph.
**Information Access:** Full continuous history of edge weights from $t=0$ to $t_{current}$.
**Strengths:** Explicitly models momentum. If a wave is building on an edge, the trend component $T_t$ will be highly positive, causing the planner to artificially inflate the cost of that edge and route around it before the wave peaks.
**Weaknesses:** Assumes linear trends. May overreact to noise if $\alpha$ and $\beta$ are not perfectly tuned.

## The "Forecasting + Planning" Attack
The inclusion of Predictive Dijkstra represents a rigorous evaluation standard for the RL agent. It directly addresses the fundamental theoretical challenge: *"Why not explicitly forecast the traffic and run Dijkstra?"* If the RL agent cannot beat Predictive Dijkstra, the project concludes that decoupled forecasting is superior to end-to-end learning for this class of problems.
