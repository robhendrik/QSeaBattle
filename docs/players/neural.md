# Neural players (unassisted)

## Purpose
Define unconstrained neural-network-based players that do not use shared randomness.
These players serve as comparison baselines against assisted strategies.


## Characteristics
- Communication limited to `comms_size`
- No shared randomness
- Fully trainable via reinforcement learning or supervised learning


## Behavioral constraints
- Player A MUST NOT access gun information
- Player B MUST NOT access full field information
- Communication bandwidth MUST be respected


## Notes
Neural players may violate classical assisted performance bounds,
but serve as useful empirical baselines.