# Statistical Analysis

## Primary Metrics
For every evaluation run, the following metrics are recorded:
1. **Mean Travel Time:** The sum of the edge weights traversed by the agent. Lower is better.
2. **Success Rate:** The percentage of episodes where the agent successfully reaches the destination within the step limit.
3. **Standard Deviation:** The variance in travel time across the $N=100$ independent evaluation seeds.

## Paired Comparisons
Because all agents are evaluated on the exact same set of 100 test seeds (which deterministically control the source, destination, and traffic noise realization), the environments experienced by each algorithm are mathematically identical. This allows for rigorous paired comparisons. If algorithm A beats algorithm B on seed $S$, we know the performance difference is due to algorithmic choice, not a "lucky" traffic realization.

## Confidence and Variance
Reporting only mean travel times can obscure catastrophic failures. RL policies often suffer from high variance, occasionally getting trapped in loops or taking severe detours. The statistical evaluation strictly monitors the standard deviation of rewards. If an RL agent achieves a slightly better mean travel time but suffers a massive standard deviation (due to occasional 0% success rates), it is considered functionally inferior to a stable classical baseline.
