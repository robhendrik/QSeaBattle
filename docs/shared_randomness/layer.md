# SharedRandomnessLayer (differentiable implementation)

## Purpose
Provide a differentiable implementation of the **Shared randomness resource (non-classical)** contract,
to enable gradient-based training (imitation, DRU/DIAL, RL variants) while preserving runtime semantics.

This layer is an **implementation** of the shared randomness contract; it does not define new behavior.


## Location
- **Module:** `src/Q_Sea_Battle/shared_randomness_layer.py`
- **Class:** `SharedRandomnessLayer`


## Inputs / Outputs
Implementations MAY support:
- Unbatched `(length,)` inputs and outputs
- Batched `(B, length)` inputs and outputs
- TensorFlow tensors throughout

Outcomes MUST be interpretable as binary at runtime:
- During training: may use continuous relaxations (e.g., logits, straight-through estimators)
- During inference/evaluation: MUST produce discrete `{0,1}` outcomes (or be replaced by Python `SharedRandomness`)


## Behavioral contract
The layer MUST satisfy all requirements from:
- `docs/shared_randomness/shared_randomness.md`

Additionally:
- The differentiable relaxation MUST NOT increase the number of communicated bits.
- Any temperature/annealing schedule MUST be documented and controlled by training utilities.


## Recommended modes (informative)
- `mode="train"`: differentiable relaxation (logits / sigmoid / straight-through)
- `mode="eval"`: discrete sampling consistent with `p_high`


## Failure modes
- Shape mismatch MUST raise `ValueError`.