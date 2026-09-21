# LinMeasurementLayerA

> Role: Trainable Keras layer that maps rank-2 field logits to rank-2 measurement logits of the same width using a small MLP.

Location: `Q_Sea_Battle.lin_measurement_layer_a.LinMeasurementLayerA`

## Constructor

Parameter | Type | Description
| --- | --- | --- |
| hidden_units | int, constraint $\ge 1$, scalar | Number of units in the hidden `Dense` layer.
| name | Optional[str], scalar | Optional layer name.
| dtype | Optional[tf.dtypes.DType], scalar | Optional layer dtype; also used to set sublayer dtypes in `build()`.
| **kwargs | Any, mapping | Additional keyword arguments forwarded to `tf.keras.layers.Layer`.

Preconditions
- `hidden_units` is int-like and `hidden_units >= 1`.

Postconditions
- `self.hidden_units` is set to `int(hidden_units)`.
- `self._dense_hidden` is `None` and `self._dense_out` is `None` until `build(input_shape)` is called by Keras (or manually).

Errors
- `ValueError`: If `hidden_units < 1`.

Example
```python
import tensorflow as tf
from Q_Sea_Battle.lin_measurement_layer_a import LinMeasurementLayerA

layer = LinMeasurementLayerA(hidden_units=64, dtype=tf.float32)

x = tf.random.normal([8, 32])  # (B, n2)
y = layer(x, training=True)    # (B, n2)
print(y.shape)
```

## Public Methods

### build

Create sublayers once the input feature dimension is known.

Arguments
- input_shape: Any, Keras shape-like, constraint: `tf.TensorShape(input_shape)[-1]` must be statically known (not `None`).

Returns
- None, side-effect only.

Preconditions
- The last dimension of `input_shape` is statically known; let $n2 = \text{int}(\text{input\_shape}[-1])$.

Postconditions
- `self._dense_hidden` is a `tf.keras.layers.Dense`, output dtype `self.dtype`, with `units=self.hidden_units` and `activation="relu"`.
- `self._dense_out` is a `tf.keras.layers.Dense`, output dtype `self.dtype`, with `units=n2` and `activation=None`.
- `super().build(input_shape)` has been called.

Errors
- `ValueError`: If the last dimension of `input_shape` is `None`.

Example
```python
import tensorflow as tf
from Q_Sea_Battle.lin_measurement_layer_a import LinMeasurementLayerA

layer = LinMeasurementLayerA(hidden_units=16)
layer.build((None, 10))  # n2 = 10
```

### call

Forward pass.

Arguments
- field_batch: tf.Tensor, dtype float32 (or `self.dtype` if set), shape (B, n2); constraint: rank must be 2 when statically known.
- training: bool, scalar; whether the call is in training mode.
- **kwargs: Any, mapping; unused extra keyword arguments (kept for Keras compatibility).

Returns
- tf.Tensor, dtype float32 (or `self.dtype` if set), shape (B, n2); measurement logits.

Preconditions
- `field_batch` is convertible to a tensor via `tf.convert_to_tensor`.
- If `field_batch.shape.rank` is statically known, it must equal 2.
- The layer must have been built such that `self._dense_hidden` and `self._dense_out` are not `None`.

Postconditions
- Computes `h = Dense(hidden_units, relu)(field_batch)` and returns `Dense(n2, linear)(h)`.

Errors
- `ValueError`: If `field_batch` has a statically-known rank and it is not 2.
- `RuntimeError`: If `self._dense_hidden is None` or `self._dense_out is None` (layer not built correctly).

Example
```python
import tensorflow as tf
from Q_Sea_Battle.lin_measurement_layer_a import LinMeasurementLayerA

layer = LinMeasurementLayerA(hidden_units=32)
x = tf.random.normal([4, 12])  # (B, n2)
y = layer(x, training=False)
```

### get_config

Return the serializable config for Keras.

Arguments
- None.

Returns
- Dict[str, Any], mapping; a Keras-serializable config including `hidden_units`.

Preconditions
- None.

Postconditions
- The returned dict equals `super().get_config()` updated with `{"hidden_units": self.hidden_units}`.

Errors
- Not specified.

Example
```python
from Q_Sea_Battle.lin_measurement_layer_a import LinMeasurementLayerA

layer = LinMeasurementLayerA(hidden_units=8)
cfg = layer.get_config()
assert cfg["hidden_units"] == 8
```

## Data & State

- hidden_units: int, constraint $\ge 1$, scalar; width of the hidden `Dense` layer.
- _dense_hidden: Optional[tf.keras.layers.Dense], scalar reference; initialized to `None` in `__init__`, created in `build()`.
- _dense_out: Optional[tf.keras.layers.Dense], scalar reference; initialized to `None` in `__init__`, created in `build()`.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- `_ensure_rank2(x, name)` is a module-level helper used by `call()` and raises `ValueError` only when `x.shape.rank` is statically known and not equal to 2; it does not perform a dynamic (runtime) rank assertion.
- `call()` forces `field_batch` through `tf.convert_to_tensor(field_batch, dtype=self.dtype or tf.float32)`, so inputs may be cast to `self.dtype` (or `float32` if `self.dtype` is unset).

## Related

- `tf.keras.layers.Layer`
- `tf.keras.layers.Dense`

## Changelog

- Not specified.