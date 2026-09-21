# LinCombineLayerB

> Role: Trainable Keras layer that combines SR outcome logits and communication logits into a single shoot logit, optionally also returning an intermediate flip logit.

Location: `Q_Sea_Battle.lin_combine_layer_b.LinCombineLayerB`

## Derived constraints

- Symbols: $m$ = comms_size (number of communication channels), $n2$ = number of SR outcome bits/features, $B$ = batch size.
- Rank constraints: `outcome_batch` and `comm_batch` must be rank-2 tensors when rank is known.
- Channel usage: only the first communication channel `comm_batch[:, :1]` is used in the current implementation, even if $m > 1$.

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| comms_size | int, constraint $m \ge 1$, scalar | Number of communication channels. Stored for interface compatibility; computation uses only the first channel. |
| hidden_units | int or Sequence[int], constraints each element is int-castable, scalar or shape (L,) | Hidden-layer widths for the parity/flip MLP. If int, treated as a single hidden layer; if sequence, treated as a stack. Default `(64, 64)`. |
| name | Optional[str], constraint any string accepted by Keras, scalar | Optional Keras layer name. |
| dtype | Optional[tf.dtypes.DType], constraint any TensorFlow dtype, scalar | Optional dtype for layer variables and computations; if unset, call-time conversion defaults to `tf.float32`. |
| **kwargs | Any, unconstrained | Forwarded to `tf.keras.layers.Layer`. |

Preconditions

- `comms_size` is int-castable and must satisfy $m \ge 1$.

Postconditions

- `self.comms_size` is set to `int(comms_size)`.
- `self.hidden_units` is normalized to `tuple[int, ...]`.
- Sub-layers are not fully instantiated until `build()` is called; internal placeholders are initialized (`_mlp` empty, `_dense_flip` and `_dense_shoot` set to `None`).

Errors

- Raises `ValueError` if `comms_size < 1`.

Example

```python
import tensorflow as tf
from Q_Sea_Battle.lin_combine_layer_b import LinCombineLayerB

layer = LinCombineLayerB(comms_size=3, hidden_units=(32, 32), dtype=tf.float32)

B, n2, m = 4, 10, 3
outcome_batch = tf.random.normal((B, n2))
comm_batch = tf.random.normal((B, m))

shoot_logit = layer(outcome_batch, comm_batch, training=False)          # shape (B, 1)
shoot_logit2, flip_logit = layer(outcome_batch, comm_batch, return_flip=True)  # both shape (B, 1)
```

## Public Methods

### build

Signature: `build(input_shape: Any) -> None`

Create sub-layers.

Parameters

- input_shape: Any, unconstrained | Input shape metadata passed by Keras; not used to enforce a specific shape contract beyond rank-2 expectations in `call()`.

Returns

- None

Preconditions

- None specified.

Postconditions

- Creates an MLP (`self._mlp`) consisting of `len(self.hidden_units)` Dense layers with ReLU activation.
- Creates `self._dense_flip`: `tf.keras.layers.Dense(1)` for producing `flip_logit`.
- Creates `self._dense_shoot`: `tf.keras.layers.Dense(1)` for producing `shoot_logit` from engineered features `[comm, flip, comm * flip]`.

Errors

- Not specified.

### call

Signature: `call(outcome_batch: tf.Tensor, comm_batch: tf.Tensor, training: bool = False, return_flip: bool = False, **kwargs: Any) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor]`

Run the layer forward pass.

Parameters

- outcome_batch: tf.Tensor, dtype float32 or `self.dtype` after conversion, shape $(B, n2)$ | SR outcome logits.
- comm_batch: tf.Tensor, dtype float32 or `self.dtype` after conversion, shape $(B, m)$ | Communication logits.
- training: bool, scalar | Forwarded to sub-layers to control training behavior.
- return_flip: bool, scalar | If True, returns a 2-tuple `(shoot_logit, flip_logit)`.
- **kwargs: Any, unconstrained | Unused; accepted for Keras call compatibility.

Returns

- If `return_flip` is False: `shoot_logit`: tf.Tensor, dtype float32 or `self.dtype`, shape $(B, 1)$.
- If `return_flip` is True: `(shoot_logit, flip_logit)` where each is tf.Tensor, dtype float32 or `self.dtype`, shape $(B, 1)$.

Preconditions

- `outcome_batch` and `comm_batch` must be rank-2 when their rank is known to TensorFlow (`x.shape.rank is not None`).
- The layer must be built such that `self._dense_flip` and `self._dense_shoot` are not `None`.

Postconditions

- Computes `flip_logit` by applying the hidden Dense stack to `outcome_batch` and then a final Dense(1) head.
- Computes `comm_scalar = comm_batch[:, :1]` (uses only the first channel).
- Computes `shoot_features = concat([comm_scalar, flip_logit, comm_scalar * flip_logit], axis=-1)` with shape $(B, 3)$.
- Computes `shoot_logit` by applying the final Dense(1) head to `shoot_features`.

Errors

- Raises `ValueError` if `outcome_batch` or `comm_batch` has known rank not equal to 2.
- Raises `RuntimeError` if `self._dense_flip` or `self._dense_shoot` is `None` (layer not built correctly).

### get_config

Signature: `get_config() -> Dict[str, Any]`

Return the layer configuration for Keras serialization.

Parameters

- None

Returns

- Dict[str, Any], unconstrained mapping | Configuration dict including `"comms_size"` (int, $m \ge 1$, scalar) and `"hidden_units"` (tuple[int, ...], shape (L,)) in addition to base Layer config.

Preconditions

- None specified.

Postconditions

- The returned dict includes the superclass configuration updated with `comms_size` and `hidden_units`.

Errors

- Not specified.

## Data & State

- comms_size: int, constraint $m \ge 1$, scalar | Number of communication channels (stored; only first is used in `call()`).
- hidden_units: tuple[int, ...], constraints elements int, shape (L,) | Normalized hidden layer sizes for the flip MLP.
- _mlp: list[tf.keras.layers.Layer], shape (L,) | Dense hidden layers created in `build()`.
- _dense_flip: Optional[tf.keras.layers.Dense], constraint either `None` (pre-build) or Dense with units=1 | Flip-logit output head.
- _dense_shoot: Optional[tf.keras.layers.Dense], constraint either `None` (pre-build) or Dense with units=1 | Shoot-logit output head.

## Planned (design-spec)

- Not specified (no design notes provided).

## Deviations

- None identified (no design notes provided to compare).

## Notes for Contributors

- The rank checks in `call()` only trigger when TensorFlow knows the static rank (`x.shape.rank is not None`); dynamic-rank tensors may bypass these checks.
- Only `comm_batch[:, :1]` is consumed; extending to use all $m$ channels would require changing feature engineering and/or the shoot head input.
- The call converts inputs via `tf.convert_to_tensor(..., dtype=self.dtype or tf.float32)`; changing default dtype behavior should consider Keras mixed precision policies.

## Related

- TensorFlow Keras base class: `tf.keras.layers.Layer`
- Dense layers: `tf.keras.layers.Dense`

## Changelog

- Not specified.