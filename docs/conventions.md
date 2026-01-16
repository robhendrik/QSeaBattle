# Conventions

## Purpose
This document defines global notation, terminology, and cross-cutting conventions used throughout the QSeaBattle
specification.

Unless explicitly marked as informative, statements in this file are normative.

## Notation

### Sizes and shapes
- `field_size` is the board side length.
- `n2 = field_size * field_size` is the total number of cells.
- Vectors representing the full field or gun are flattened to length `n2`.
- Batch dimension is denoted by `B`.

Shape conventions
- `field`: `np.ndarray, dtype int {0,1}, shape (n2,)` or `tf.Tensor, dtype float32, shape (B, n2)`
- `gun`: `np.ndarray, dtype int {0,1}, shape (n2,)` (one-hot) or `tf.Tensor, dtype float32, shape (B, n2)`
- `comm`: `np.ndarray, dtype int {0,1}, shape (m,)` or `tf.Tensor, dtype float32, shape (B, m)`
- `shoot`: `int {0,1}` or `tf.Tensor, dtype float32, shape (B, 1)`

### Binary values
Unless stated otherwise:
- NumPy binary arrays use values in `{0,1}`.
- TensorFlow training utilities may use `float32` values in `{0.0, 1.0}`.

## Terminology

### Game vs tournament
- A game is one execution of the pipeline: generate field and gun -> A decides -> noise -> B decides -> reward.
- A tournament is a batch of games under identical layout parameters.

### Players and models
- Players are decision-makers that implement `PlayerA.decide` and `PlayerB.decide`.
- Models are reusable computational components (for example, neural models) that may back a player.
- A trainable model may store per-decision state (for example, per-layer outcomes) required for Player B.

### Shared resource (SR)
- SR (shared resource) is a two-party resource available to both players without communication.
- SR is not communication and MUST NOT increase `comms_size`.
- PRAssistedLayer is a specific type of SR.

## Determinism, seeding, and reproducibility
- Any utility that samples randomness (data generation, evaluation) MUST accept a `seed` parameter or use a
  reproducible RNG strategy.
- Runtime gameplay MAY be stochastic due to SR and channel noise.
- Evaluation runs SHOULD support deterministic replay via fixed seeds.

## Channel noise
- Channel noise flips each bit of `comm` independently with probability `channel_noise`.
- Noise MUST be applied after Player A decides and before Player B decides.

## Contract language
The following keywords are normative:
- MUST / MUST NOT: required for compliance.
- SHOULD / SHOULD NOT: recommended; deviations require justification.
- MAY: optional behavior permitted by the spec.

## Interface conventions

### `decide(...)`
- `decide` methods MUST be pure with respect to their explicit inputs, except for documented internal state needed for
  coordination (for example, per-layer outcomes).
- If `comms_size == 1`, `comm` MUST still be represented as an array or tensor of shape `(m,)` or `(B, 1)`.

### Dtypes
- NumPy arrays are acceptable for environment-level logic.
- TensorFlow tensors are acceptable for trainable models and training utilities.
- A function MUST document whether it expects NumPy arrays or TensorFlow tensors if ambiguous.

## Error handling
- Shape mismatches MUST raise `ValueError` (preferred) rather than silently reshaping.
- Double-use of a shared resource (SR) instance MUST raise an error (typically `ValueError` or `RuntimeError`).

## Module naming conventions

### Teacher and trainable re-export modules
- Pyramid teacher primitives are exported via `Q_Sea_Battle.pyr_teacher_layers`.
- Pyramid trainable models are exported via `Q_Sea_Battle.pyr_trainable_models`.
- Linear teacher primitives are exported via `Q_Sea_Battle.lin_teacher_layers`.
- Linear trainable models are exported via `Q_Sea_Battle.lin_trainable_models`.

### Utilities modules
- Utility modules use the suffix `_utilities.py`.
- Modules SHOULD NOT use the suffix `_utils.py`.

## Document status
- Chapters describing components are normative unless explicitly labeled informative.
- `docs/algorithms.md` is informative and does not override contracts.

## Changelog
- 2026-01-16 - Rob Hendriks: Update SR terminology and module naming conventions.
