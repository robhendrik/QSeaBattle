# PyrCombineLayerA

> Role: Non-trainable Keras layer that computes Player A’s next pyramid-level binary field state by XOR-combining even-indexed bits with a PR-assisted outcome.
Location: `Q_Sea_Battle.pyr_combine_layer_a.PyrCombineLayerA`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| name | Optional[str], optional, default None, constraints: any valid Keras layer name, shape: scalar | Optional Keras layer name passed to the base `tf.keras.layers.Layer`. |

Preconditions

- None specified.

Postconditions

- The layer is created with `trainable=False`.

Errors

- None raised directly by the constructor.

Example

```python
import tensorflow as tf
from Q_Sea_Battle.pyr_combine_layer_a import PyrCombineLayerA

layer = PyrCombineLayerA(name="pyr_combine_a")
```

## Public Methods

### call(field_batch, sr_outcome_batch)

Combine the current field with the PR-assisted outcome by taking even-indexed field bits and computing elementwise XOR via $(a + b) \bmod 2$.

Parameters

- field_batch: tf.Tensor, dtype float32 (converted internally), constraints: values in {0,1} expected, shape (B, L)
- sr_outcome_batch: tf.Tensor, dtype float32 (converted internally), constraints: values in {0,1} expected, shape (B, L/2)

Returns

- tf.Tensor, dtype float32, constraints: values in {0,1} produced by floormod, shape (B, L/2)

Preconditions

- L (the last dimension of `field_batch`) must be even.
- `sr_outcome_batch` last dimension must equal L/2.

Postconditions

- Output equals `floormod(field_batch[..., ::2] + sr_outcome_batch, 2.0)`.

Errors

- Raises via TensorFlow runtime assertions: `tf.debugging.assert_equal(L % 2, 0, ...)` if L is not even.
- Raises via TensorFlow runtime assertions: `tf.debugging.assert_equal(tf.shape(field[..., ::2])[-1], tf.shape(sr_outcome_batch)[-1], ...)` if `sr_outcome_batch` length does not equal L/2.

Example

```python
import tensorflow as tf
from Q_Sea_Battle.pyr_combine_layer_a import PyrCombineLayerA

layer = PyrCombineLayerA()

field_batch = tf.constant([[1, 0, 1, 1],
                           [0, 1, 0, 1]], dtype=tf.float32)   # shape (B=2, L=4)
sr_outcome_batch = tf.constant([[0, 1],
                                [1, 1]], dtype=tf.float32)    # shape (2, 2)

# even bits are [1,1] and [0,0]; XOR with sr_outcome_batch via mod-2 add
next_field = layer(field_batch, sr_outcome_batch)             # shape (2, 2)
```

## Data & State

- Inherits from `tf.keras.layers.Layer`.
- Trainable state: None (layer is constructed with `trainable=False`).
- Internal state: None specified.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- This layer converts inputs to `tf.float32` and uses `tf.math.floormod(even + sr, 2.0)` to implement XOR; changing dtype or the XOR implementation may affect downstream assumptions about output dtype and value set.
- Input value constraints ({0,1}) are documented but not explicitly asserted beyond shape checks; consider adding value-range assertions only if required by performance and correctness constraints.

## Related

- TensorFlow: `tf.keras.layers.Layer`, `tf.convert_to_tensor`, `tf.debugging.assert_equal`, `tf.math.floormod`.

## Changelog

- 0.1: Initial Step 1 implementation encoding the teacher rule: next field is even-indexed bits XOR PR-assisted outcome.