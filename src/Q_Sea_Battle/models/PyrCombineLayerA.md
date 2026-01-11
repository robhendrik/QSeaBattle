# Class PyrCombineLayerA

**Module import path**: `Q_Sea_Battle.pyr_combine_layer_a.PyrCombineLayerA`

> Deterministic (non-trainable) pyramid combine rule for Player A.

!!! note "Parent class"
    Inherits from `tf.keras.layers.Layer`.

!!! note "Terminology"
    This layer uses the parameter name `sr_outcome_batch` for backward compatibility.
    Here **SR = shared resource**; the tensor content is a **PR-assisted outcome** produced by `PRAssistedLayer`.

!!! note "Derived symbols"
    Let `field_size = n`, `n2 = n**2`, and `comms_size = m`.
    In pyramid models, the active state length `L` is halved each level, starting at `L = n2`.

## Overview

`PyrCombineLayerA` implements the Step-1 teacher rule used by the pyramid (Pyr) architecture:

- Input: binary field state `F^ℓ` of length `L` (even).
- Input: PR-assisted outcome `S_A^ℓ` of length `L/2`.
- Output: next field state `F^(ℓ+1)` of length `L/2`, computed as:

`F^(ℓ+1)[i] = F^ℓ[2*i] XOR S_A^ℓ[i]`

This layer is **non-trainable** by design and is used for dataset generation and early integration tests.

## Constructor

### Signature

- `PyrCombineLayerA(name: str | None = None) -> PyrCombineLayerA`

### Arguments

- `name`: `str` or `None`, scalar.
  - Optional Keras layer name.

### Returns

- `PyrCombineLayerA`, scalar.

### Preconditions

- None.

### Postconditions

- Layer is created with `trainable=False`.

### Errors

- None.

## Public Methods

### call

#### Signature

- `call(field_batch: tf.Tensor, sr_outcome_batch: tf.Tensor) -> tf.Tensor`

#### Arguments

- `field_batch`: `tf.Tensor`, dtype `float32`, shape `(B, L)`.
  - Values are expected to be binary (0 or 1).
  - `L` must be even.
- `sr_outcome_batch`: `tf.Tensor`, dtype `float32`, shape `(B, L/2)`.
  - Values are expected to be binary (0 or 1).
  - Represents the **PR-assisted outcome** aligned with the even positions of `field_batch`.

#### Returns

- `next_field_batch`: `tf.Tensor`, dtype `float32`, shape `(B, L/2)`.
  - Values are binary (0 or 1).

#### Preconditions

- `field_batch` is rank-2.
- `sr_outcome_batch` is rank-2.
- `L % 2 == 0`.
- `sr_outcome_batch.shape[-1] == L/2` (checked at runtime).

#### Postconditions

- `next_field_batch[..., i]` equals `(field_batch[..., 2*i] + sr_outcome_batch[..., i]) mod 2`.

#### Errors

- Raises `tf.errors.InvalidArgumentError` if `L` is not even (runtime assertion).
- Raises `tf.errors.InvalidArgumentError` if `sr_outcome_batch` length does not match `L/2` (runtime assertion).

!!! example "Minimal usage"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.pyr_combine_layer_a import PyrCombineLayerA

    layer = PyrCombineLayerA()
    field = tf.zeros((2, 16), dtype=tf.float32)      # B=2, L=16
    outcome = tf.zeros((2, 8), dtype=tf.float32)     # B=2, L/2=8
    next_field = layer(field, outcome)               # shape (2, 8)
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
- Preserve runtime assertions on `L` evenness and alignment with `L/2`.

## Changelog

- 2026-01-11 — Author: Rob Hendriks
