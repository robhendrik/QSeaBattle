# Class PyrMeasurementLayerB

**Module import path**: `Q_Sea_Battle.pyr_measurement_layer_b.PyrMeasurementLayerB`

> Deterministic (non-trainable) pyramid measurement rule for Player B.

!!! note "Parent class"
    Inherits from `tf.keras.layers.Layer`.

!!! note "Derived symbols"
    Let `field_size = n`, `n2 = n**2`, and `comms_size = m`.
    In pyramid models, the active state length `L` is halved each level, starting at `L = n2`.

## Overview

`PyrMeasurementLayerB` implements the Step-1 teacher measurement rule used by the pyramid (Pyr) architecture.

Given a binary gun state `G^ℓ` of length `L` (even), the layer produces a measurement vector `M_B^ℓ` of length `L/2`
defined per pair:

- `M_B^ℓ[i] = (NOT G^ℓ[2*i]) AND G^ℓ[2*i + 1]`

This layer is **non-trainable** and is intended for dataset generation and early integration validation.

## Constructor

### Signature

- `PyrMeasurementLayerB(name: str | None = None) -> PyrMeasurementLayerB`

### Arguments

- `name`: `str` or `None`, scalar.
  - Optional Keras layer name.

### Returns

- `PyrMeasurementLayerB`, scalar.

### Preconditions

- None.

### Postconditions

- Layer is created with `trainable=False`.

### Errors

- None.

## Public Methods

### call

#### Signature

- `call(gun_batch: tf.Tensor) -> tf.Tensor`

#### Arguments

- `gun_batch`: `tf.Tensor`, dtype `float32`, shape `(B, L)`.
  - Values are expected to be binary (0 or 1).
  - `L` must be even.

#### Returns

- `meas_batch`: `tf.Tensor`, dtype `float32`, shape `(B, L/2)`.
  - Values are binary (0 or 1).

#### Preconditions

- `gun_batch` is rank-2.
- `L % 2 == 0`.

#### Postconditions

- `meas_batch[..., i]` equals `(1 - gun_batch[..., 2*i]) * gun_batch[..., 2*i + 1]`, clipped to `[0, 1]`.

#### Errors

- Raises `tf.errors.InvalidArgumentError` if `L` is not even (runtime assertion).

!!! example "Minimal usage"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.pyr_measurement_layer_b import PyrMeasurementLayerB

    layer = PyrMeasurementLayerB()
    gun = tf.zeros((2, 16), dtype=tf.float32)  # B=2, L=16
    meas = layer(gun)  # shape (2, 8)
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
