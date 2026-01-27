# PyrCombineLayerA

> Role: Trainable Keras layer mapping `(field_batch, sr_outcome_batch)` to `next_field` logits for the next field.

Location: `Q_Sea_Battle.pyr_combine_layer_a.PyrCombineLayerA`

## Derived constraints

- Let $L$ be the last dimension of `field_batch`; $L$ must be statically known at build time and must be even, so that $L/2$ is an integer.
- Let $B$ be the batch dimension.
- At runtime, the last dimension of `sr_outcome_batch` must equal $L/2$.

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| hidden_units | int, constraint $\ge 1$, scalar | Number of hidden units in the internal Dense(ReLU) layer. |
| name | Optional[str], scalar | Keras layer name; may be `None`. |
| dtype | Optional[tf.dtypes.DType], scalar | Layer dtype; may be `None` (defaults are applied by Keras and `call()` conversions). |
| **kwargs | Any, variadic | Forwarded to `tf.keras.layers.Layer` constructor. |

Preconditions

- `hidden_units` is an `int` with value $\ge 1$.

Postconditions

- `self.hidden_units` is set to `int(hidden_units)`.
- Sub-layers (`_dense_hidden`, `_dense_out`) are not created until `build()`.

Errors

- Raises `ValueError` if `hidden_units < 1`.

Example

```python
import tensorflow as tf
from Q_Sea_Battle.pyr_combine_layer_a import PyrCombineLayerA

layer = PyrCombineLayerA(hidden_units=64, dtype=tf.float32)
```

## Public Methods

### build

Signature: `build(input_shape: Any) -> None`

Parameters

- `input_shape`: Any, constraint "shape-like accepted by Keras", scalar or nested; expected to describe `field_batch` as `(B, L)` and may also include a second shape for `sr_outcome_batch` that is ignored for building.

Returns

- `None`.

Behavior

- Infers $L$ from the last dimension of the field shape and computes `out_dim = L // 2`.
- Creates two Dense sublayers: a hidden Dense with `hidden_units` and ReLU activation, and an output Dense with `out_dim` and linear activation.
- Records `self._built_for_L = L`.

Preconditions

- The last dimension $L$ of the field shape is statically known (not `None`).
- $L$ is even.

Postconditions

- `self._dense_hidden` and `self._dense_out` are non-`None` instances of `tf.keras.layers.Dense`.
- `self._built_for_L` is set to `int(L)`.

Errors

- Raises `ValueError` if the field shape last dimension is not statically known.
- Raises `ValueError` if $L$ is not even.

### call

Signature: `call(field_batch: tf.Tensor, sr_outcome_batch: tf.Tensor, training: bool = False, **kwargs: Any) -> tf.Tensor`

Parameters

- `field_batch`: tf.Tensor, dtype convertible to `self.dtype` (or `tf.float32` if `self.dtype is None`), shape $(B, L)$, rank 2.
- `sr_outcome_batch`: tf.Tensor, dtype convertible to `self.dtype` (or `tf.float32` if `self.dtype is None`), shape $(B, L/2)$, rank 2.
- `training`: bool, scalar; forwarded to Dense sublayers.
- `**kwargs`: Any, variadic; accepted but not used.

Returns

- `next_field`: tf.Tensor, dtype `self.dtype` if set else `tf.float32`, shape $(B, L/2)$.

Behavior

- Converts inputs to tensors with dtype `self.dtype` (or `tf.float32` if unset).
- Validates both inputs are rank-2.
- Asserts at runtime that `sr_outcome_batch.shape[-1] == field_batch.shape[-1] // 2`.
- Concatenates inputs along the last axis to form shape $(B, 3L/2)$, applies hidden Dense(ReLU), then output Dense(linear) to produce logits.

Preconditions

- Layer has been built so that `_dense_hidden` and `_dense_out` exist.

Postconditions

- No state is created during the call.

Errors

- Raises `ValueError` if either input is not rank-2 (when rank is statically known and not equal to 2).
- Raises a TensorFlow assertion error at runtime if `sr_outcome_batch` last dimension does not equal `field_batch` last dimension divided by 2.
- Raises `RuntimeError` if sublayers are missing (layer not built correctly).

Example

```python
import tensorflow as tf
from Q_Sea_Battle.pyr_combine_layer_a import PyrCombineLayerA

B, L = 4, 10
field = tf.random.uniform((B, L), dtype=tf.float32)
sr = tf.random.uniform((B, L // 2), dtype=tf.float32)

layer = PyrCombineLayerA(hidden_units=32, dtype=tf.float32)
logits = layer(field, sr, training=True)
assert logits.shape == (B, L // 2)
```

### get_config

Signature: `get_config() -> Dict[str, Any]`

Parameters

- None.

Returns

- `config`: Dict[str, Any], mapping; includes base layer config plus key `"hidden_units"` with value `int`.

Behavior

- Extends `tf.keras.layers.Layer.get_config()` with `{"hidden_units": self.hidden_units}`.

## Data & State

- `hidden_units`: int, constraint $\ge 1$, scalar; persisted in config.
- `_dense_hidden`: Optional[tf.keras.layers.Dense], scalar; created in `build()`, otherwise `None`.
- `_dense_out`: Optional[tf.keras.layers.Dense], scalar; created in `build()`, otherwise `None`.
- `_built_for_L`: Optional[int], scalar; set to $L$ in `build()`, otherwise `None`.

## Planned (design-spec)

- Not specified.

## Deviations

- The module docstring describes a contract that `sr_outcome_batch` has shape `(B, L/2)` and that `build()` can use only the field shape; the implementation matches this, but `build()` accepts multiple possible `input_shape` encodings and ignores SR shape even when provided.

## Notes for Contributors

- Keras may call `build()` with only the first input shape for multi-input layers; this implementation intentionally creates all state in `build()` from the field shape alone and validates SR shape at runtime in `call()`.
- Avoid creating new variables or sublayers in `call()`; sublayers are expected to be created in `build()`.

## Related

- TensorFlow: `tf.keras.layers.Layer`, `tf.keras.layers.Dense`
- Tensor ops: `tf.concat`, `tf.convert_to_tensor`, `tf.debugging.assert_equal`

## Changelog

- Not specified.