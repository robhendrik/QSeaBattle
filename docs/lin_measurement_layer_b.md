# LinMeasurementLayerB

> Role: Trainable Keras layer mapping gun logits to measurement logits with matching width via a two-layer MLP.

Location: `Q_Sea_Battle.lin_measurement_layer_b.LinMeasurementLayerB`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| hidden_units | int, constraint: >= 1, shape: scalar | Width of the hidden dense layer. |
| name | Optional[str], constraint: any, shape: scalar | Optional Keras layer name. |
| dtype | Optional[tf.dtypes.DType], constraint: any, shape: scalar | Optional Keras dtype for layer variables and computation. |
| **kwargs | Any, constraint: forwarded to `tf.keras.layers.Layer`, shape: N/A | Additional keyword arguments forwarded to `Layer`. |

Preconditions

- `hidden_units` is an `int` with constraint: `hidden_units >= 1`, shape: scalar.

Postconditions

- `self.hidden_units` is set to `int(hidden_units)`.
- `self._dense_hidden` is `Optional[tf.keras.layers.Dense]`, initialized to `None` until `build(...)` is called.
- `self._dense_out` is `Optional[tf.keras.layers.Dense]`, initialized to `None` until `build(...)` is called.

Errors

- Raises `ValueError` if `hidden_units < 1`.

Example

```python
import tensorflow as tf
from Q_Sea_Battle.lin_measurement_layer_b import LinMeasurementLayerB

layer = LinMeasurementLayerB(hidden_units=64)
x = tf.random.normal([8, 16])  # (B, n2)
y = layer(x)                   # (B, n2)
```

## Public Methods

### build

- Signature: `build(self, input_shape: Any) -> None`

Parameter(s)

- `input_shape`: Any, constraint: convertible to `tf.TensorShape` and must have statically known last dimension, shape: N/A.

Return value

- `None`, constraint: N/A, shape: scalar.

Behavior

- Creates two sub-layers after inferring `n2` from `input_shape[-1]`: `Dense(hidden_units, relu)` followed by `Dense(n2, linear)`.

Errors

- Raises `ValueError` if the final dimension of `input_shape` is unknown (`None`).

### call

- Signature: `call(self, gun_batch: tf.Tensor, training: bool = False, **kwargs: Any) -> tf.Tensor`

Parameter(s)

- `gun_batch`: tf.Tensor, dtype: any convertible to layer dtype (defaults to `float32` if `self.dtype` is `None`), shape (B, n2); constraint: must be rank-2 when rank is statically known.
- `training`: bool, constraint: any, shape: scalar; passed through to sub-layer calls.
- `**kwargs`: Any, constraint: unused (present for Keras API compatibility), shape: N/A.

Return value

- tf.Tensor, dtype: matches internal computation dtype (`self.dtype` or `float32`), shape (B, n2); constraint: output width equals `n2` inferred at build time.

Errors

- Raises `ValueError` if `gun_batch` has a statically known rank not equal to 2.
- Raises `RuntimeError` if the layer has not been built correctly (i.e., sub-layers are not initialized).

### get_config

- Signature: `get_config(self) -> Dict[str, Any]`

Parameter(s)

- None.

Return value

- Dict[str, Any], constraint: Keras-serializable configuration, shape: N/A; includes key `"hidden_units"` with value type `int`, shape: scalar.

## Data & State

- `hidden_units`: int, constraint: >= 1, shape: scalar; number of units in the hidden dense layer.
- `_dense_hidden`: Optional[tf.keras.layers.Dense], constraint: `None` before `build(...)`, otherwise a `Dense` with `units=hidden_units` and `activation="relu"`, shape: N/A.
- `_dense_out`: Optional[tf.keras.layers.Dense], constraint: `None` before `build(...)`, otherwise a `Dense` with `units=n2` and `activation=None`, shape: N/A.
- `n2`: int, constraint: `n2 = int(input_shape[-1])` and must be statically known, shape: scalar; inferred at build time and used as output width.

## Planned (design-spec)

- Not specified.

## Deviations

- No design notes provided; no deviations identified.

## Notes for Contributors

- Rank validation in `call(...)` only triggers when the rank is statically known (`x.shape.rank is not None`); dynamic rank mismatches may not raise at this check.
- Sub-layers are created in `build(...)`; calling `call(...)` before the layer is built raises `RuntimeError`.

## Related

- TensorFlow: `tf.keras.layers.Layer`
- TensorFlow: `tf.keras.layers.Dense`

## Changelog

- Not specified.