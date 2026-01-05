# Conventions

## Purpose
This document defines **global notation, terminology, and cross-cutting conventions** used throughout the
QSeaBattle specification. Unless explicitly marked as *informative*, statements in this file are **normative**.


## Notation

### Sizes and shapes
- Let `field_size = n` (board side length). The total number of cells is $n^2$.
- Vectors representing the full field or gun are **flattened** to length $n^2$.
- Batch dimension is denoted by $B$.

**Shape conventions**
- `field`: `(B, n^2)` or `(n^2,)` for a single sample
- `gun`: `(B, n^2)` or `(n^2,)` one-hot
- `comm`: `(B, m)` where `m = comms_size` (often `m=1`)
- `shoot`: `(B, 1)` or scalar `{0,1}`

### Binary values
Unless stated otherwise, binary arrays use values in `{0,1}`.


## Terminology

### Game vs tournament
- A **game** is one execution of the pipeline: generate field+gun â†’ A decides â†’ noise â†’ B decides â†’ reward.
- A **tournament** is a batch of games under identical layout parameters.

### Players and models
- **Players** are decision-makers that implement `PlayerA.decide` and `PlayerB.decide`.
- **Models** are reusable computational components (e.g., neural models) that may back a player.
- A trainable model may store **per-decision state** (e.g., per-layer outcomes) required for Player B.

### Shared randomness (SR)
- **Shared randomness** is a two-party resource used once per layer (Pyr) or once per decision (Lin).
- SR is **not communication**. SR does not increase `comms_size`.


## Determinism, seeding, and reproducibility
- Any utility that samples randomness (data generation, evaluation) MUST accept a `seed` parameter or use a
  reproducible RNG strategy.
- Runtime gameplay MAY be stochastic due to SR and channel noise; evaluation runs SHOULD support deterministic
  replay via fixed seeds.


## Channel noise
- Channel noise flips each bit of `comm` independently with probability `channel_noise`.
- Noise MUST be applied **after** Player A decides and **before** Player B decides.


## Contract language (RFC-style)
The following keywords are normative:
- **MUST / MUST NOT**: required for compliance
- **SHOULD / SHOULD NOT**: recommended; deviations require justification
- **MAY**: optional behavior permitted by the spec


## Interface conventions

### `decide(...)`
- `decide` methods MUST be pure with respect to their explicit inputs, except for documented internal state
  needed for coordination (e.g., per-layer outcomes).
- For `comms_size == 1`, `comm` MUST still be represented as an array/tensor of shape `(B, 1)` (not a scalar).

### Dtypes
- Numpy arrays are acceptable for environment-level logic.
- TensorFlow tensors are acceptable for trainable models and training utilities.
- A given function MUST document whether it expects NumPy or Tensor inputs if ambiguous.


## Error handling
- Shape mismatches MUST raise `ValueError` (preferred) rather than silently reshaping.
- Double-use of shared randomness MUST raise an error (typically `ValueError` or `RuntimeError`), not silently proceed.


## Document status
- Chapters describing components are **normative**, unless explicitly labeled *informative*.
- `docs/algorithms.md` is **informative** and does not override contracts.