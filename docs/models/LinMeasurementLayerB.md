# Class LinMeasurementLayerB

**Module import path**: `Q_Sea_Battle.lin_measurement_layer_b.LinMeasurementLayerB`

> Learnable mapping from a flattened gun vector to per-cell measurement probabilities in `[0, 1]`.

!!! note "Parent class"
    Inherits from `tf.keras.layers.Layer`.

!!! note "Derived symbols"
    Let `field_size = n`, `n2 = n**2`, and `comms_size = m`.

## Overview

`LinMeasurementLayerB` is a low-level trainable primitive used inside `LinTrainableAssistedModelB`.
It maps a gun vector of length `n2` to per-cell measurement probabilities.

- Input is converted to `tf.Tensor` and cast to `tf.float32` when needed.
- Output is produced by an MLP ending in a sigmoid layer, so values lie in `[0, 1]`.

## Constructor

### Signature

- `LinMeasurementLayerB(n2: int, hidden_units: Sequence[int] = (64,), name: str | None = "LinMeasurementLayerB", **kwargs)`

### Arguments

- `n2`: `int`, scalar.
  - Number of cells in the flattened gun vector.
- `hidden_units`: `Sequence[int]`, shape `(H,)`.
  - Dense-ReLU layer widths, in order.
- `name`: `str` or `None`, scalar.
- `**kwargs`: `dict[str, Any]`, scalar.
  - Forwarded to `tf.keras.layers.Layer`.

### Returns

- `LinMeasurementLayerB`, scalar.

### Preconditions

- `n2 > 0`.

### Postconditions

- `self.n2`: `int`, scalar, equals `n2`.
- `self.hidden_units`: `tuple[int, ...]`, shape `(H,)`.
- MLP layers are created lazily in `build(...)`.

### Errors

- Raises `ValueError` if `n2 <= 0`.

## Public Methods

### build

#### Signature

- `build(input_shape: Any) -> None`

#### Arguments

- `input_shape`: `Any`, scalar.

#### Returns

- `None`.

#### Preconditions

- None.

#### Postconditions

- Creates Dense-ReLU layers per `hidden_units` plus a sigmoid output Dense layer of width `n2`.

#### Errors

- Propagates TensorFlow/Keras build errors.

### call

#### Signature

- `call(guns: tf.Tensor, training: bool = False) -> tf.Tensor`

#### Arguments

- `guns`: `tf.Tensor`, dtype `float32` (or castable to `float32`), shape:
  - `(n2,)`, or
  - `(B, n2)` for batch size `B`.
- `training`: `bool`, scalar.

#### Returns

- `meas_probs`: `tf.Tensor`, dtype `float32`, shape:
  - `(n2,)` if input shape is `(n2,)`, or
  - `(B, n2)` if input shape is `(B, n2)`.
- Value range: `[0.0, 1.0]`.

#### Preconditions

- Input rank is 1 or 2.
- Last dimension equals `n2`.

#### Postconditions

- Output has the same rank and leading dimensions as input.
- Output values are probabilities in `[0.0, 1.0]`.

#### Errors

- Raises `ValueError` if input rank is not 1 or 2.
- Raises `ValueError` if last dimension does not equal `n2`.

!!! example "Minimal usage"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.lin_measurement_layer_b import LinMeasurementLayerB

    n2 = 16
    layer = LinMeasurementLayerB(n2=n2)

    gun = tf.zeros((n2,), dtype=tf.float32)
    probs = layer(gun)  # shape (n2,)
    ```

## Serialization

- Standard Keras layer behavior applies.
- This class does not define `get_config()` explicitly; base-class serialization applies to tracked sublayers and weights.

## Planned (design-spec)

- None identified.

## Deviations

- None identified.

## Notes for Contributors

- Keep strict shape validation: rank must be 1 or 2 and last dim must match `n2`.
- Preserve sigmoid output semantics (probabilities in `[0, 1]`).

## Changelog

- 2026-01-11 — Author: Technical Documentation Team
