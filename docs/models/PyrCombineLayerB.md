# Class PyrCombineLayerB

**Module import path**: `Q_Sea_Battle.pyr_combine_layer_b.PyrCombineLayerB`

> Deterministic (non-trainable) pyramid combine rule for Player B.

!!! note "Parent class"
    Inherits from `tf.keras.layers.Layer`.

!!! note "Terminology"
    This layer uses the parameter name `sr_outcome_batch` for backward compatibility.
    Here **SR = shared resource**; the tensor content is a **PR-assisted outcome** produced by `PRAssistedLayer`.

!!! note "Derived symbols"
    Let `field_size = n`, `n2 = n**2`, and `comms_size = m`.
    In pyramid models, the active state length `L` is halved each level, starting at `L = n2`.

## Overview

`PyrCombineLayerB` updates both:

- the next gun state `G^(ℓ+1)` (length `L/2`), and
- the next communication bit `C^(ℓ+1)` (shape `(B, 1)`),

given current gun state `G^ℓ` (length `L`), PR-assisted outcome `S^ℓ` (length `L/2`), and current comm bit `C^ℓ`.

Per pair formulas (vectorized):

- Next gun state:
  - `G^(ℓ+1)[i] = G^ℓ[2*i] XOR S^ℓ[i]`

- Communication update:
  - `C^(ℓ+1) = C^ℓ XOR ( Σ_i (G^ℓ[2*i + 1] * S^ℓ[i]) mod 2 )`

Intuition: gate the PR-assisted outcomes by whether the gun is in the **odd** position of each pair, then take parity.

## Constructor

### Signature

- `PyrCombineLayerB(name: str | None = None) -> PyrCombineLayerB`

### Arguments

- `name`: `str` or `None`, scalar.
  - Optional Keras layer name.

### Returns

- `PyrCombineLayerB`, scalar.

### Preconditions

- None.

### Postconditions

- Layer is created with `trainable=False`.

### Errors

- None.

## Public Methods

### call

#### Signature

- `call(gun_batch: tf.Tensor, sr_outcome_batch: tf.Tensor, comm_batch: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]`

#### Arguments

- `gun_batch`: `tf.Tensor`, dtype `float32`, shape `(B, L)`.
  - Values are expected to be binary (0 or 1).
  - `L` must be even.
- `sr_outcome_batch`: `tf.Tensor`, dtype `float32`, shape `(B, L/2)`.
  - Values are expected to be binary (0 or 1).
  - Represents the **PR-assisted outcome** aligned to the pairs of `gun_batch`.
- `comm_batch`: `tf.Tensor`, dtype `float32`, shape `(B, 1)`.
  - Values are expected to be binary (0 or 1).

#### Returns

- `(next_gun, next_comm)`:
  - `next_gun`: `tf.Tensor`, dtype `float32`, shape `(B, L/2)`, values in `(0, 1)`.
  - `next_comm`: `tf.Tensor`, dtype `float32`, shape `(B, 1)`, values in `(0, 1)`.

#### Preconditions

- All inputs are rank-2.
- `L % 2 == 0`.
- `sr_outcome_batch.shape[-1] == L/2` (checked at runtime).
- `comm_batch.shape[-1] == 1` (checked at runtime).

#### Postconditions

- `next_gun[..., i]` equals `(gun_batch[..., 2*i] + sr_outcome_batch[..., i]) mod 2`.
- `next_comm` equals `(comm_batch + parity( gun_odd * sr_outcome )) mod 2`.

#### Errors

- Raises `tf.errors.InvalidArgumentError` if `L` is not even (runtime assertion).
- Raises `tf.errors.InvalidArgumentError` if `sr_outcome_batch` length does not match `L/2` (runtime assertion).
- Raises `tf.errors.InvalidArgumentError` if `comm_batch` does not have last dimension 1 (runtime assertion).

!!! example "Minimal usage"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.pyr_combine_layer_b import PyrCombineLayerB

    layer = PyrCombineLayerB()
    gun = tf.zeros((2, 16), dtype=tf.float32)       # B=2, L=16
    outcome = tf.zeros((2, 8), dtype=tf.float32)    # B=2, L/2=8
    comm = tf.zeros((2, 1), dtype=tf.float32)       # B=2, 1
    next_gun, next_comm = layer(gun, outcome, comm)
    ```

## Serialization

- Uses standard Keras tracking/serialization for non-trainable layers.
- No custom `get_config()` is defined; name/weights follow base-class behavior.

## Planned (design-spec)

- None.

## Deviations

- Parameter name `sr_outcome_batch` remains for backward compatibility, but its meaning is PR-assisted outcome.

## Notes for Contributors

- Keep this layer deterministic and `trainable=False` unless the design spec explicitly changes.
- Preserve runtime assertions on `L` evenness, `(B, L/2)` alignment, and comm shape `(B, 1)`.

## Changelog

- 2026-01-11 — Author: Rob Hendriks
