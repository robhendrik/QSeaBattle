# PyrCombineLayerB

> Role: Trainable Keras layer mapping (gun, sr_outcome, comm) -> (next_gun logits, next_comm logits) for the next pyramid level.

Location: `Q_Sea_Battle.pyr_combine_layer_b.PyrCombineLayerB`

## Derived constraints

- Let $L$ be the last dimension of `gun_batch` (the gun feature size). Constraint: $L$ must be statically known at build time and even; the layer outputs gun logits with last dimension $L/2$.
- All inputs to `call(...)` must be rank-2 tensors: shape $(B, D)$ for batch size $B$ and feature dimension $D$.

## Constructor

Parameter | Type | Description
--- | --- | ---
hidden_units | int, constraint $>= 1$ | Width of the intermediate dense layer.
name | str or None, optional | Keras layer name.
dtype | tf.dtypes.DType or None, optional | Layer dtype; if None, `call()` converts inputs to `tf.float32` unless overridden elsewhere.
**kwargs | dict[str, Any] | Forwarded to `tf.keras.layers.Layer` base constructor.

Preconditions

- `hidden_units` must be an integer value $>= 1$.

Postconditions

- `self.hidden_units` is set to `int(hidden_units)`.
- Internal sublayers (`_dense_hidden`, `_dense_gun`, `_dense_comm`) are initialized to `None` and are created later in `build(...)`.

Errors

- Raises `ValueError` if `hidden_units < 1`.

Example

```python
import tensorflow as tf
from Q_Sea_Battle.pyr_combine_layer_b import PyrCombineLayerB

layer = PyrCombineLayerB(hidden_units=64)
```

## Public Methods

### build(input_shape)

Creates sublayers based on the statically known gun dimension $L$.

Parameters

- input_shape: Any, constraints: must be convertible to `tf.TensorShape` and have a statically known last dimension $L$.

Returns

- None

Preconditions

- `input_shape` must have a statically known last dimension $L`.
- $L$ must be even.

Postconditions

- Creates the following sublayers (all with `dtype=self.dtype`): `dense_hidden` (Dense, units=`hidden_units`, activation="relu"), `dense_gun` (Dense, units=$L/2$, activation=None), `dense_comm` (Dense, units=1, activation=None).
- Stores `self._built_for_L = L`.

Errors

- Raises `ValueError` if `gun_batch` last dimension $L$ is not statically known.
- Raises `ValueError` if $L$ is not even.

Example

```python
import tensorflow as tf
from Q_Sea_Battle.pyr_combine_layer_b import PyrCombineLayerB

layer = PyrCombineLayerB(hidden_units=32)
layer.build((None, 8))  # L=8 -> next_gun dim is 4
```

### call(gun_batch, sr_outcome_batch, comm_batch, training=False, **kwargs)

Forward pass combining gun, SR outcome, and communication bit into next-level logits.

Parameters

- gun_batch: tf.Tensor, dtype float32 or `self.dtype`, shape (B, L), constraints: rank-2; last dim $L$; $L$ even (enforced at build time).
- sr_outcome_batch: tf.Tensor, dtype float32 or `self.dtype`, shape (B, L/2), constraints: rank-2; last dim must equal `tf.shape(gun_batch)[-1] // 2`.
- comm_batch: tf.Tensor, dtype float32 or `self.dtype`, shape (B, 1), constraints: rank-2; last dim must equal 1.
- training: bool, constraints: no additional constraints | Passed to sublayers.
- **kwargs: dict[str, Any] | Accepted but not otherwise specified by this implementation.

Returns

- next_gun: tf.Tensor, dtype float32 or `self.dtype`, shape (B, L/2), constraints: logits (activation=None).
- next_comm: tf.Tensor, dtype float32 or `self.dtype`, shape (B, 1), constraints: logits (activation=None).

Preconditions

- The layer must have been built such that internal sublayers exist (typically via first call with known input shapes or explicit `build(...)`).
- All three inputs must be rank-2 tensors.

Postconditions

- Produces `next_gun` and `next_comm` by concatenating inputs along the last axis, applying a hidden Dense layer with ReLU, then projecting to two separate linear heads.

Errors

- Raises `ValueError` if any input has known static rank not equal to 2.
- Raises `tf.errors.InvalidArgumentError` (via `tf.debugging.assert_equal`) if `sr_outcome_batch` last dim is not `L/2` at runtime.
- Raises `tf.errors.InvalidArgumentError` (via `tf.debugging.assert_equal`) if `comm_batch` last dim is not 1 at runtime.
- Raises `RuntimeError` if sublayers are missing (layer not built correctly).

Example

```python
import tensorflow as tf
from Q_Sea_Battle.pyr_combine_layer_b import PyrCombineLayerB

B, L = 4, 8
gun = tf.zeros((B, L), dtype=tf.float32)
sr = tf.zeros((B, L // 2), dtype=tf.float32)
comm = tf.zeros((B, 1), dtype=tf.float32)

layer = PyrCombineLayerB(hidden_units=64)
next_gun, next_comm = layer(gun, sr, comm, training=False)
print(next_gun.shape, next_comm.shape)  # (4, 4) (4, 1)
```

### get_config()

Returns the Keras-serializable configuration.

Parameters

- None

Returns

- cfg: dict[str, Any], constraints: includes base layer config and key `"hidden_units"` with value `int`.

Errors

- Not specified.

Example

```python
from Q_Sea_Battle.pyr_combine_layer_b import PyrCombineLayerB

layer = PyrCombineLayerB(hidden_units=16)
cfg = layer.get_config()
assert cfg["hidden_units"] == 16
```

## Data & State

- hidden_units: int, constraint $>= 1$ | Hyperparameter controlling hidden Dense width.
- _dense_hidden: tf.keras.layers.Dense or None | Created in `build(...)`; units=`hidden_units`, activation="relu".
- _dense_gun: tf.keras.layers.Dense or None | Created in `build(...)`; units=$L/2$, activation=None.
- _dense_comm: tf.keras.layers.Dense or None | Created in `build(...)`; units=1, activation=None.
- _built_for_L: int or None | Stores the gun dimension $L$ used during `build(...)`.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- Do not create trainable state in `call(...)`; sublayers are expected to be created in `build(...)` based on the statically known gun dimension $L$.
- `_ensure_rank2(...)` validates rank only when the rank is statically known; dynamic-rank inputs may bypass this check and rely on downstream TensorFlow errors.

## Related

- TensorFlow Keras `tf.keras.layers.Layer`
- TensorFlow Keras `tf.keras.layers.Dense`

## Changelog

- Not specified.