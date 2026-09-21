# PyrMeasurementLayerB

> Role: Trainable Keras layer that maps a gun-state tensor to measurement logits using a small MLP head.

Location: `Q_Sea_Battle.pyr_measurement_layer_b.PyrMeasurementLayerB`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| hidden_units | int, constraint $\ge 1$, scalar | Width of the hidden Dense layer (ReLU). |
| name | Optional[str], nullable, scalar | Optional Keras layer name. |
| dtype | Optional[tf.dtypes.DType], nullable, scalar | Optional dtype for layer weights and computations. |
| **kwargs | Any, unconstrained | Forwarded to `tf.keras.layers.Layer` base constructor. |

Preconditions

- `hidden_units` is an `int` with value $\ge 1$.

Postconditions

- `self.hidden_units` is set to `int(hidden_units)`.
- `self._dense_hidden`, `self._dense_out`, and `self._built_for_L` are initialized to `None` and are created/set during `build(...)`.

Errors

- `ValueError`: If `hidden_units < 1`.

Example

```python
import tensorflow as tf
from Q_Sea_Battle.pyr_measurement_layer_b import PyrMeasurementLayerB

layer = PyrMeasurementLayerB(hidden_units=64, dtype=tf.float32)
x = tf.zeros((8, 10), dtype=tf.float32)  # L=10 -> output width 5
y = layer(x, training=False)
assert y.shape == (8, 5)
```

## Public Methods

### build

Signature: `build(input_shape: Any) -> None`

Create sublayers based on the input width $L$ (the last dimension of `input_shape`), and set the output dimension to $L/2$.

Parameters

- `input_shape`: Any, Keras/TensorFlow shape-like, must have a statically known last dimension $L$ that is even.

Returns

- `None`: NoneType, no value.

Preconditions

- The last dimension $L$ of `input_shape` is statically known (not `None`).
- $L$ is even ($L \bmod 2 = 0$).

Postconditions

- `self._dense_hidden` is created as `tf.keras.layers.Dense(self.hidden_units, activation="relu")`.
- `self._dense_out` is created as `tf.keras.layers.Dense(L // 2, activation=None)`.
- `self._built_for_L` is set to `int(L)`.
- Base class `build` is called.

Errors

- `ValueError`: If the last dimension $L$ is not statically known.
- `ValueError`: If $L$ is not even.

### call

Signature: `call(gun_batch: tf.Tensor, training: bool = False, **kwargs: Any) -> tf.Tensor`

Run a forward pass producing measurement logits (no sigmoid applied).

Parameters

- `gun_batch`: tf.Tensor, dtype Not specified (converted via `tf.convert_to_tensor(..., dtype=self.dtype or tf.float32)`), shape (B, L).
- `training`: bool, scalar, forwarded to sublayers.
- `**kwargs`: Any, unused (present for Keras compatibility).

Returns

- `meas_logits`: tf.Tensor, dtype equals `self.dtype` if set else `tf.float32`, shape (B, L/2).

Preconditions

- If `gun_batch` has a statically known rank, it must be rank-2.
- At runtime, the last dimension $L$ must be even ($L \bmod 2 = 0$).
- The layer must have been built such that `self._dense_hidden` and `self._dense_out` are not `None`.

Postconditions

- Output is computed as `Dense(L/2)(Dense(hidden_units, relu)(gun_batch))` with logits output.

Errors

- `ValueError`: If `gun_batch` has statically known rank and it is not 2.
- `RuntimeError`: If `self._dense_hidden` or `self._dense_out` is missing (layer not built correctly).
- TensorFlow assertion failure: If runtime $L$ is not even (via `tf.debugging.assert_equal(tf.shape(x)[-1] % 2, 0, ...)`).

Example

```python
import tensorflow as tf
from Q_Sea_Battle.pyr_measurement_layer_b import PyrMeasurementLayerB

layer = PyrMeasurementLayerB(hidden_units=32)
gun_batch = tf.random.uniform((4, 12), minval=-0.5, maxval=0.5)  # scaled domain (convention)
meas_logits = layer(gun_batch, training=True)
print(meas_logits.shape)  # (4, 6)
```

### get_config

Signature: `get_config() -> Dict[str, Any]`

Return the Keras serialization config.

Parameters

- None.

Returns

- `cfg`: Dict[str, Any], unconstrained mapping; includes base layer config plus `"hidden_units": self.hidden_units`.

Example

```python
from Q_Sea_Battle.pyr_measurement_layer_b import PyrMeasurementLayerB

layer = PyrMeasurementLayerB(hidden_units=16)
cfg = layer.get_config()
assert cfg["hidden_units"] == 16
```

## Data & State

- `hidden_units`: int, constraint $\ge 1$, scalar; number of units in the hidden Dense layer.
- `_dense_hidden`: Optional[tf.keras.layers.Dense], nullable; created in `build(...)`, then used in `call(...)`.
- `_dense_out`: Optional[tf.keras.layers.Dense], nullable; created in `build(...)`, then used in `call(...)`.
- `_built_for_L`: Optional[int], nullable; the input width $L$ used to parameterize the layer during `build(...)`.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- The layer relies on a statically known last dimension during `build(...)`; if you change tracing/build behavior, preserve the requirement that $L$ is known to create the output head with dimension $L/2$.
- The output is logits (no sigmoid); if you add squashing/noise behavior, document it explicitly and consider whether it belongs outside this layer.

## Related

- TensorFlow / Keras base class: `tf.keras.layers.Layer`
- Dense sublayers used internally: `tf.keras.layers.Dense`

## Changelog

- Not specified.