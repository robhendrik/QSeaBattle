# Class PyrMeasurementLayerA

**Module import path**: `Q_Sea_Battle.pyr_measurement_layer_a.PyrMeasurementLayerA`

> Deterministic (non-trainable) pyramid measurement rule for Player A.

!!! note "Parent class"
    Inherits from `tf.keras.layers.Layer`.

!!! note "Derived symbols"
    Let `field_size = n`, `n2 = n**2`, and `comms_size = m`.
    In pyramid models, the active state length `L` is halved each level, starting at `L = n2`.

## Overview

`PyrMeasurementLayerA` implements the Step-1 teacher rule used by the pyramid (Pyr) architecture:

- Input: binary field state `F^ℓ` of length `L` (even).
- Output: measurement vector `M_A^ℓ` of length `L/2`, computed by pairwise XOR:

`M_A^ℓ[i] = F^ℓ[2*i] XOR F^ℓ[2*i + 1]`

This layer is **non-trainable** by design and is used for dataset generation and early integration tests.

## Constructor

### Signature

- `PyrMeasurementLayerA(name: str | None = None) -> PyrMeasurementLayerA`

### Arguments

- `name`: `str` or `None`, scalar.
  - Optional Keras layer name.

### Returns

- `PyrMeasurementLayerA`, scalar.

### Preconditions

- None.

### Postconditions

- Layer is created with `trainable=False`.

### Errors

- None.

## Public Methods

### call

#### Signature

- `call(field_batch: tf.Tensor) -> tf.Tensor`

#### Arguments

- `field_batch`: `tf.Tensor`, dtype `float32`, shape `(B, L)`.
  - Values are expected to be binary (0 or 1).
  - `L` must be even.

#### Returns

- `meas_batch`: `tf.Tensor`, dtype `float32`, shape `(B, L/2)`.
  - Values are binary (0 or 1).

#### Preconditions

- `field_batch` is rank-2.
- `L % 2 == 0`.

#### Postconditions

- `meas_batch[..., i]` equals `(field_batch[..., 2*i] + field_batch[..., 2*i+1]) mod 2`.

#### Errors

- Raises `tf.errors.InvalidArgumentError` if `L` is not even (runtime assertion).

!!! example "Minimal usage"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.pyr_measurement_layer_a import PyrMeasurementLayerA

    layer = PyrMeasurementLayerA()
    field = tf.zeros((2, 16), dtype=tf.float32)  # B=2, L=16
    meas = layer(field)  # shape (2, 8)
    ```

## Serialization

- Uses standard Keras tracking/serialization for non-trainable layers.
- No custom `get_config()` is defined; name/weights follow base-class behavior.

## Planned (design-spec)

- None.

## Deviations

- None.

## Notes for Contributors

- Keep this layer deterministic and `trainable=False` unless the design spec explicitly changes.
- Preserve runtime assertion that `L` is even.

## Changelog

- 2026-01-11 — Author: Rob Hendriks
