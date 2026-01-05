# Reinforcement learning

## Purpose
Specify reinforcement-learning-based training regimes for players,
primarily for baseline comparison.

## Supported approaches
- Policy gradients
- Self-play
- Actor-critic methods

## Constraints
- RL MUST respect all information-flow constraints
- Reward signals MUST come exclusively from GameEnv
- Exploration MUST NOT leak additional information

## Notes
Reinforcement learning is not the primary focus of QSeaBattle,
but provides useful empirical benchmarks.