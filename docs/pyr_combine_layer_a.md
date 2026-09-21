# PyrCombineLayerA

> Role: Trainable Keras layer that combines a per-player field representation with an SR outcome vector to produce next-field logits.

Location: `Q_Sea_Battle.pyr_combine_layer_a.PyrCombineLayerA`

## Derived constraints

- Define $L$ as the last dimension of `field_batch` (field width) and $B$ as the batch size. `build()` requires $L$ to be statically known and even, and the output width is $L/2$.
- At runtime, `sr_outcome_batch` last dimension must equal $L/2$.

## Constructor

| Parameter | Type | Description |
|---|---|---|
| hidden_units | int, constraint $\ge 1$, scalar | Width of the hidden Dense layer. |
| name | Optional[str], scalar | Optional Keras layer name. |
| dtype | Optional[tf.dtypes.DType], scalar | Optional Keras dtype for layer variables and computations. |
| **kwargs | Any, scalar | Forwarded to `tf.keras.layers.Layer`. |

Preconditions

- `hidden_units` is an `int` with constraint $\ge 1$.

Postconditions

- `self.hidden_units` is set to `int(hidden_units)`.
- `self._dense_hidden`, `self._dense_out`, and `self._built_for_L` are initialized to `None` (created/set in `build()`).

Errors

- Raises `ValueError` if `hidden_units < 1`.

Example

```python
import tensorflow as tf
from Q_Sea_Battle.pyr_combine_layer_a import PyrCombineLayerA

layer = PyrCombineLayerA(hidden_units=64)

B, L = 8, 20
field_batch = tf.random.uniform((B, L), dtype=tf.float32)
sr_outcome_batch = tf.random.uniform((B, L // 2), dtype=tf.float32)

y = layer(field_batch, sr_outcome_batch, training=True)
print(y.shape)  # (8, 10)
```

## Public Methods

### build

Signature: `build(input_shape: Any) -> None`

Parameters

- `input_shape`: Any, shape structure; description: shape for `field_batch`, or a multi-input shape structure where the first element corresponds to `field_batch`.

Returns

- `NoneType`, no constraints, scalar.

Preconditions

- The last dimension $L$ of the inferred `field_batch` shape is statically known.
- $L$ is even.

Postconditions

- Creates `self._dense_hidden: tf.keras.layers.Dense` with `units=self.hidden_units`, `activation="relu"`, `dtype=self.dtype`.
- Creates `self._dense_out: tf.keras.layers.Dense` with `units=L/2`, `activation=None`, `dtype=self.dtype`.
- Sets `self._built_for_L` to $L$.

Errors

- Raises `ValueError` if the last dimension $L$ is not statically known.
- Raises `ValueError` if $L$ is not even.

Notes

- Keras may pass only the first input’s shape for multi-input layers; this implementation sizes sublayers using only the inferred `field_batch` width.

### call

Signature: `call(field_batch: tf.Tensor, sr_outcome_batch: tf.Tensor, training: bool = False, **kwargs: Any) -> tf.Tensor`

Parameters

- `field_batch`: tf.Tensor, dtype float32 (or `self.dtype` if set), shape $(B, L)$; field tensor. Rank must be 2 when statically known.
- `sr_outcome_batch`: tf.Tensor, dtype float32 (or `self.dtype` if set), shape $(B, L/2)$; SR outcome logits. Rank must be 2 when statically known.
- `training`: bool, scalar; standard Keras training flag passed to sublayers.
- `**kwargs`: Any, scalar; unused (present for Keras compatibility).

Returns

- `tf.Tensor`, dtype float32 (or `self.dtype` if set), shape $(B, L/2)$; next-field logits (no sigmoid).

Preconditions

- `build()` has been executed successfully such that `self._dense_hidden` and `self._dense_out` are not `None`.
- If static rank is known, both inputs have rank 2.
- Runtime constraint: `sr_outcome_batch` last dimension equals `field_batch` last dimension divided by 2.

Postconditions

- Returns `next_field_logits = Dense(L/2)(Dense(hidden_units, relu)(concat([field_batch, sr_outcome_batch], axis=-1)))`.

Errors

- Raises `ValueError` if static rank is known and either input is not rank 2.
- Raises `tf.errors.InvalidArgumentError` if `sr_outcome_batch` last dimension does not equal `field_batch` last dimension divided by 2 (via `tf.debugging.assert_equal`).
- Raises `RuntimeError` if sublayers were not created in `build()`.

### get_config

Signature: `get_config() -> Dict[str, Any]`

Parameters

- None.

Returns

- `Dict[str, Any]`, unconstrained mapping; includes the base Layer config plus `{"hidden_units": self.hidden_units}`.

## Data & State

- `hidden_units`: int, constraint $\ge 1$, scalar; hidden Dense width set at construction.
- `_dense_hidden`: Optional[tf.keras.layers.Dense], scalar; created in `build()`, `None` before build.
- `_dense_out`: Optional[tf.keras.layers.Dense], scalar; created in `build()`, `None` before build.
- `_built_for_L`: Optional[int], scalar; stores the $L$ used during `build()`, `None` before build.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- `build()` attempts to handle Keras passing only the first input shape for multi-input layers; if you change input handling, keep this compatibility behavior in mind.
- `_ensure_rank2` only enforces rank-2 when the rank is statically known; runtime rank mismatches may not be caught by this check.
- Output is logits by design (`activation=None` on the output Dense); do not add a sigmoid unless the training/inference pipeline is updated accordingly.

## Related

- TensorFlow / Keras: `tf.keras.layers.Layer`, `tf.keras.layers.Dense`
- Internal helpers in the same module: `_ensure_rank2`, `_require_known_last_dim`

## Changelog

- Not specified.