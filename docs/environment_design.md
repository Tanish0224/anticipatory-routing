# Environment Design and Traffic Dynamics

## 1. Topography
The routing environment is modeled as a directed grid network. The final experimental topology is a $10 \times 10$ directed grid with diagonal cross-edges, yielding $|V| = 100$ nodes and $|E| = 342$ directed edges.

### Initial Environment Diagnostic (Version 1)
Initial iterations of this research utilized a $4 \times 4$ grid topology. A diagnostic analysis revealed that on a $4 \times 4$ grid, the optimal path is approximately 3 hops. Because traffic disturbances require multiple timesteps to propagate across the graph, a 3-step trip concludes before upstream congestion can intercept the agent. This structurally prevented temporal anticipation from providing a verifiable advantage, as $m \le k$ (where $m$ is travel time and $k$ is wave propagation time). The expansion to a $10 \times 10$ grid ensures $m > k$, enforcing a strict anticipation condition.

## 2. Traffic Generator (Kinematic Waves)

The core experimental premise requires traffic to exhibit predictable spatiotemporal structure rather than purely independent noise.

### The Diffusion Flaw
An early traffic model propagated congestion by averaging upstream edge weights. Mathematically, this represented a discrete heat equation (isotropic diffusion). Consequently, congestion shocks decayed exponentially and dissipated in place. Reactive planning successfully navigated this regime because the congestion did not effectively translate.

### Translating Kinematic Waves (Version 2 / Regime 3)
To establish a rigorous benchmark for anticipatory routing, the traffic generator was redesigned to approximate directional kinematic waves. The update rule avoids isotropic diffusion by utilizing a directional max-pooling operation over upstream edges to preserve wave amplitude:
$$C_{propagation}^{(e)} = \max_{j \in U(e)} C_t^{(j)}$$
$$C_{t+1}^{(e)} = (1 - S) C_t^{(e)} + S \cdot C_{propagation}^{(e)} + \xi_t^{(e)}$$
Where $U(e)$ denotes the set of topologically upstream edges relative to a propagation vector (e.g., West to East), and $S$ controls the wave translation speed. This generates a persistent, high-amplitude congestion front that sweeps across the topology.

## 3. Correlation Regimes ($\rho$)
The environment stochasticity is parameterized by a correlation coefficient $\rho \in [0.0, 1.0]$:
- $\rho = 0.0$: Pure independent stochastic noise, representing an unstructured environment.
- $\rho \rightarrow 1.0$: Highly structured translating kinematic waves, representing a predictable spatiotemporal environment.
Evaluating algorithms across this sweep identifies the precise structural conditions under which explicit forecasting or model-free learning provides a computational advantage.
