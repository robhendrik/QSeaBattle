# PyrMeasurementLayerA

> Role: Non-trainable Keras layer that computes Player A’s per-level measurement vector via pairwise XOR over a binary field state batch.

Location: `Q_Sea_Battle.pyr_measurement_layer_a.PyrMeasurementLayerA`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| name | Optional[str], constraint: None or non-empty string, shape: scalar | Optional Keras layer name. |

Preconditions

- Not specified.

Postconditions

- The layer is created with `trainable=False`.

Errors

- Not specified.

!!! example "Example"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.pyr_measurement_layer_a import PyrMeasurementLayerA

    layer = PyrMeasurementLayerA(name="pyr_meas_a")
    field_batch = tf.constant([[0, 1, 1, 1],
                               [1, 0, 0, 0]], dtype=tf.float32)  # shape (B=2, L=4)
    meas = layer(field_batch)  # shape (2, 2)
    print(meas.numpy())
    ```

## Public Methods

### call(field_batch)

Compute pairwise XOR measurements.

Parameters

- field_batch: tf.Tensor, dtype convertible to float32, constraint: values in {0, 1}, shape (B, L) where B is batch size and L is active state length and L mod 2 = 0.

Returns

- tf.Tensor, dtype float32, constraint: values in {0, 1}, shape (B, L/2).

Preconditions

- `field_batch` last dimension length L must be even.

Postconditions

- Output equals pairwise XOR computed as $(a + b) \bmod 2$ over adjacent pairs along the last dimension.

Errors

- Raises an assertion failure via `tf.debugging.assert_equal` if $L \bmod 2 \ne 0$ with message `"Active length L must be even."`.

## Data & State

- Trainable state: None; constructor sets `trainable=False`.
- Internal variables/weights: None specified/created by this layer.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- The implementation converts inputs via `tf.convert_to_tensor(..., dtype=tf.float32)` and uses `tf.math.floormod(even + odd, 2.0)` to implement XOR over {0,1}; keep this consistent with the teacher/measurement rule.
- The runtime check for even L is enforced with `tf.debugging.assert_equal`; ensure tests cover both valid even L and invalid odd L cases.

## Related

- TensorFlow: `tf.keras.layers.Layer`

## Changelog

- 0.1: Initial implementation of non-trainable pairwise XOR measurement layer for Player A.