# PyrCombineLayerB

> Role: Trainable Keras layer that concatenates current gun state, SR outcome logits, and comm logit to produce next-level gun logits and an updated comm logit.

Location: `Q_Sea_Battle.pyr_combine_layer_b.PyrCombineLayerB`

## Derived constraints

- Let $L$ be the last dimension of `gun_batch`; $L$ must be statically known at build time and must be even.
- Let $B$ be the batch size (dynamic); all inputs to `call` must be rank-2 with leading dimension $B$.
- `sr_outcome_batch` last dimension must equal $L/2$ (checked at runtime).
- `comm_batch` last dimension must equal $1$ (checked at runtime).
- Outputs have shapes `(B, L/2)` for `next_gun_logits` and `(B, 1)` for `next_comm_logit`.

## Constructor

Parameter | Type | Description
--- | --- | ---
hidden_units | int, constraint $\ge 1$, scalar | Number of hidden units in the intermediate Dense layer.
name | Optional[str], scalar | Layer name.
dtype | Optional[tf.dtypes.DType], scalar | Layer dtype; inputs are converted to this dtype (or float32 if `None`).
**kwargs | Any, scalar | Passed to the base Keras `Layer` constructor.

Preconditions

- `hidden_units` is an `int` with `hidden_units >= 1`.

Postconditions

- `self.hidden_units` is set to `int(hidden_units)`.
- No sublayers are created until `build()` is called; `_dense_hidden`, `_dense_gun`, `_dense_comm`, and `_built_for_L` are initialized to `None`.

Errors

- `ValueError`: if `hidden_units < 1`.

Example

!!! example "Instantiate the layer"
      ```python
      import tensorflow as tf
      from Q_Sea_Battle.pyr_combine_layer_b import PyrCombineLayerB

      layer = PyrCombineLayerB(hidden_units=64, dtype=tf.float32)
      ```

## Public Methods

### build

- Signature: `build(input_shape: Any) -> None`

Creates sublayers using the statically-known gun width $L$ derived from `input_shape` (Keras may pass only the first input shape for multi-input layers).

Arguments

- `input_shape`: Any, constraint convertible to `tf.TensorShape`, scalar; interpreted as the gun input shape, whose last dimension is $L$.

Returns

- `None`.

Preconditions

- `input_shape` must have a statically known last dimension $L$.
- $L$ must be even.

Postconditions

- Creates and assigns the following sublayers:
  - `_dense_hidden`: `tf.keras.layers.Dense`, units=`hidden_units`, activation=`"relu"`, dtype=`self.dtype`.
  - `_dense_gun`: `tf.keras.layers.Dense`, units=`L/2`, activation=`None`, dtype=`self.dtype`.
  - `_dense_comm`: `tf.keras.layers.Dense`, units=`1`, activation=`None`, dtype=`self.dtype`, `kernel_initializer=RandomNormal(stddev=0.01)`, `bias_initializer="zeros"`.
- Sets `_built_for_L` to `int(L)`.
- Calls `super().build(input_shape)`.

Errors

- `ValueError`: if the last dimension of `input_shape` is not statically known.
- `ValueError`: if $L$ is odd.

### call

- Signature: `call(gun_batch: tf.Tensor, sr_outcome_batch: tf.Tensor, comm_batch: tf.Tensor, training: bool = False, **kwargs: Any) -> Tuple[tf.Tensor, tf.Tensor]`

Runs a forward pass in logit space (no sigmoid). Concatenates inputs, applies a hidden Dense layer, then produces next gun logits and a residual-updated comm logit.

Arguments

- `gun_batch`: `tf.Tensor`, dtype float32 or `self.dtype`, shape $(B, L)$; current gun state (typically scaled values during training).
- `sr_outcome_batch`: `tf.Tensor`, dtype float32 or `self.dtype`, shape $(B, L/2)$; SR outcome logits aligned to the current level.
- `comm_batch`: `tf.Tensor`, dtype float32 or `self.dtype`, shape $(B, 1)$; current communication bit as a logit.
- `training`: `bool`, scalar; passed to Dense layers as their `training` argument.
- `**kwargs`: `Any`, scalar; unused, accepted for Keras compatibility.

Returns

- `(next_gun_logits, next_comm_logit)`: `Tuple[tf.Tensor, tf.Tensor]` where:
  - `next_gun_logits`: `tf.Tensor`, dtype float32 or `self.dtype`, shape $(B, L/2)$; logits for next-level gun representation.
  - `next_comm_logit`: `tf.Tensor`, dtype float32 or `self.dtype`, shape $(B, 1)$; updated comm logit computed as `dense_comm(h) + comm_batch`.

Preconditions

- All three inputs must be rank-2 when statically known.
- Runtime shape requirements must hold: `sr_outcome_batch.shape[-1] == gun_batch.shape[-1] // 2` and `comm_batch.shape[-1] == 1`.
- The layer must have been built such that `_dense_hidden`, `_dense_gun`, and `_dense_comm` are not `None`.

Postconditions

- Inputs are converted via `tf.convert_to_tensor(..., dtype=self.dtype or tf.float32)`.
- Produces outputs as described under Returns.

Errors

- `ValueError`: if any input has a statically known rank that is not 2.
- `tf.errors.InvalidArgumentError`: if runtime assertions on last dimensions fail (`sr_outcome_batch` not $L/2$, or `comm_batch` not 1).
- `RuntimeError`: if sublayers are missing (layer not built correctly).

Example

!!! example "Forward pass"
      ```python
      import tensorflow as tf
      from Q_Sea_Battle.pyr_combine_layer_b import PyrCombineLayerB

      B = 8
      L = 16

      layer = PyrCombineLayerB(hidden_units=64, dtype=tf.float32)
      gun_batch = tf.random.uniform((B, L), minval=-0.5, maxval=0.5)
      sr_outcome_batch = tf.random.normal((B, L // 2))
      comm_batch = tf.random.normal((B, 1))

      next_gun_logits, next_comm_logit = layer(gun_batch, sr_outcome_batch, comm_batch, training=True)
      ```

### get_config

- Signature: `get_config() -> Dict[str, Any]`

Returns the serialized configuration for Keras, including `hidden_units`.

Arguments

- None.

Returns

- `Dict[str, Any]`, scalar mapping; contains base layer config plus key `"hidden_units"` with value `int`.

## Data & State

- `hidden_units`: `int`, constraint $\ge 1$, scalar; number of hidden units in the intermediate Dense layer.
- `_dense_hidden`: `Optional[tf.keras.layers.Dense]`, scalar; created in `build()`, units=`hidden_units`, activation=`relu`.
- `_dense_gun`: `Optional[tf.keras.layers.Dense]`, scalar; created in `build()`, units=`L/2`, activation=`None` (logits).
- `_dense_comm`: `Optional[tf.keras.layers.Dense]`, scalar; created in `build()`, units=`1`, activation=`None` (logits), small-stddev kernel initializer.
- `_built_for_L`: `Optional[int]`, scalar; gun width $L$ used when building, or `None` if not yet built.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- Sublayers must be created in `build()`; `call()` should remain free of state creation to match the stated Keras 3 build note.
- `call()` enforces rank-2 only when rank is statically known; runtime shape checks use `tf.debugging.assert_equal` for last dimensions.
- The comm output is a residual logit update: `dense_comm(h) + comm_batch`; changing this alters downstream behavior.

## Related

- TensorFlow Keras `tf.keras.layers.Layer`
- TensorFlow Keras `tf.keras.layers.Dense`

## Changelog

- Not specified.