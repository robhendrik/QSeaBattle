# PyrMeasurementLayerB

> Role: Keras layer that deterministically computes Player B's per-level pyramid measurement vector using the pairwise rule $M_B^\ell[i] = (1 - G^\ell[2i]) \cdot G^\ell[2i + 1]$.

Location: `Q_Sea_Battle.pyr_measurement_layer_b.PyrMeasurementLayerB`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| name | Optional[str], optional, any string, scalar | Optional Keras layer name. |

Preconditions

- None specified.

Postconditions

- The layer is created with `trainable=False`.

Errors

- Not specified.

Example

```python
import tensorflow as tf
from Q_Sea_Battle.pyr_measurement_layer_b import PyrMeasurementLayerB

layer = PyrMeasurementLayerB(name="pyr_meas_b")
gun_batch = tf.constant([[0, 1, 1, 1]], dtype=tf.float32)  # shape (B=1, L=4)
meas = layer(gun_batch)  # shape (1, 2), expected [[1, 0]]
```

## Public Methods

### call(gun_batch)

Compute pairwise "¬even AND odd" measurements.

Parameters

- gun_batch: tf.Tensor, dtype convertible to float32, values intended in {0,1}, shape (B, L) where L is even.

Returns

- tf.Tensor, dtype float32, values in {0,1}, shape (B, L/2).

Preconditions

- $L$ must be even.

Postconditions

- Output values are clipped to $[0, 1]$.

Errors

- Raises a TensorFlow assertion error if $L \bmod 2 \ne 0$ with message `"Active length L must be even."`.

Example

```python
import tensorflow as tf
from Q_Sea_Battle.pyr_measurement_layer_b import PyrMeasurementLayerB

layer = PyrMeasurementLayerB()
gun_batch = tf.constant(
    [
        [0, 1, 0, 0],  # pairs: (0,1)->1, (0,0)->0
        [1, 1, 0, 1],  # pairs: (1,1)->0, (0,1)->1
    ],
    dtype=tf.float32,
)
meas = layer(gun_batch)
# meas shape (2, 2), values [[1,0],[0,1]]
```

## Data & State

- Trainable state: None (the layer is constructed with `trainable=False`).
- Internal variables/weights: None specified in the module.
- Computation details: Inputs are converted to float32, reshaped to pairs along the last axis, and the measurement is computed as `(1.0 - even) * odd` then clipped to `[0.0, 1.0]`.

## Planned (design-spec)

- None specified.

## Deviations

- No design notes provided; no deviations identified.

## Notes for Contributors

- Keep the layer non-trainable unless the module explicitly evolves beyond "Step 1" teacher-rule encoding.
- If changing the rule, update both the module docstring and the `call` implementation to remain consistent.

## Related

- TensorFlow: `tf.keras.layers.Layer`
- The "Pyr dataset spec" rule described in the module docstring: pairwise "¬even AND odd".

## Changelog

- Version 0.1: Initial implementation of the deterministic (teacher) measurement rule for Player B.