# Methodology

## Mathematical Formulation

### Network Definition
We define the routing environment as a directed graph $G = (V, E)$. The topology is a $10 \times 10$ uniform grid with diagonal cross-edges, yielding $|V| = 100$ nodes and $|E| = 342$ directed edges.

### Dynamic Edge Costs
Each edge $e \in E$ has a static base cost $W_{base}(e)$ proportional to its Euclidean distance. The dynamic edge cost at time $t$ is defined as:
$$c_e(t) = W_{base}(e) (1 + \beta C_e(t))$$
where $C_e(t) \in [0, 1]$ represents the traffic congestion state on edge $e$, and $\beta$ is a penalty scaling factor (set to $\beta=5.0$).

### Traffic State and Transition
The global traffic state is the vector $C_t \in [0, 1]^{|E|}$.
The transition function $F$ updates the traffic state at each timestep:
$$C_{t+1} = F(C_t, \xi_t; \theta)$$
Under Regime 3 (Kinematic Wave), $F$ applies a directional max-pooling operation over upstream edges to propagate high-amplitude congestion waves without isotropic diffusion. The parameter $\rho \in [0, 1]$ controls the spatiotemporal correlation of the stochastic innovation $\xi_t$.

### Markov Decision Process (MDP) Formulation
The dynamic routing problem is cast as a finite-horizon fully observable MDP.
- **State ($s_t$):** The current traffic state $C_t$ (or bijectively, the current edge costs $W_t$) and the current agent node $v_t$.
- **Action ($a_t$):** Selection of an outgoing edge $e = (v_t, v_{t+1})$, constrained to valid successors $v_{t+1} \in \mathcal{N}(v_t)$.
- **Reward ($r_t$):** The negative scalar travel time of the traversed edge, $r_t = -c_e(t)$.
- **Terminal Condition:** The episode terminates when $v_t = v_{dest}$ or $t = t_{max}$.

### Agent Observation and Policy
To evaluate the impact of temporal features on model-free learning, we define the agent observation $o_t$:
- **H=1:** $o_t = [s_t]$, representing a purely reactive Markov observation.
- **H=3:** $o_t = [s_{t-2}, s_{t-1}, s_t]$, providing explicit finite-difference features to approximate $\partial C / \partial t$.
The routing policy $\pi(a_t | o_t)$ is parameterized by a Double Deep Q-Network (Double DQN) optimized to maximize $\mathbb{E}[\sum_{k=0}^{T} \gamma^k r_{t+k}]$.
