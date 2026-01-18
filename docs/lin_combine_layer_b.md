# LinCombineLayerB

> Role: Learnable mapping (outcomes, comm) -> shoot logit (a single logit output).

Location: `Q_Sea_Battle.lin_combine_layer_b.LinCombineLayerB`

## Constructor

Parameter | Type | Description
--- | --- | ---
comms_size | int, constraint: $comms\_size \ge 0$, scalar | Number of communication channels $m$; used for shape checks only (no checks are implemented in code).
hidden_units | int or collections.abc.Sequence[int], constraint: each element castable to int, scalar or shape (k,) | Hidden layer widths; an int becomes a single hidden layer, a sequence becomes multiple layers; each Dense uses ReLU activation.
name | str or None, constraint: any string or None, scalar | Layer name; if None, defaults to `"LinCombineLayerB"`.
**kwargs | dict[str, Unknown], constraint: forwarded to `tf.keras.layers.Layer`, shape N/A | Additional keyword arguments passed to the base Keras Layer.

Preconditions
- `hidden_units` must be an `int` or a sequence of values convertible to `int`.
- `outcomes` and `comm` provided to `call(...)` must be convertible to `tf.Tensor`.
- The last dimension sizes of `outcomes` and `comm` must be compatible for concatenation along axis `-1` after any optional rank-1 expansion.

Postconditions
- `self.comms_size` is set to `int(comms_size)`.
- `self.hidden_units` is normalized to `tuple[int, ...]`.
- An internal MLP is created: `len(self.hidden_units)` Dense layers with ReLU, followed by a final Dense layer with 1 unit and no activation.

Errors
- Raises `TypeError` / `ValueError` if `hidden_units` contains elements that cannot be converted to `int`.
- TensorFlow shape errors may be raised at runtime during concatenation or Dense application if tensor shapes are incompatible.

!!! example "Example"
      ```python
      import tensorflow as tf
      from Q_Sea_Battle.lin_combine_layer_b import LinCombineLayerB

      B, n2, m = 4, 25, 3
      layer = LinCombineLayerB(comms_size=m, hidden_units=(64, 32))

      outcomes = tf.random.uniform((B, n2), dtype=tf.float32)
      comm = tf.random.uniform((B, m), dtype=tf.float32)

      shoot_logit = layer(outcomes, comm, training=True)
      print(shoot_logit.shape)  # (4, 1)
      ```

## Public Methods

### call

`call(outcomes: tf.Tensor, comm: tf.Tensor, training: bool = False) -> tf.Tensor`

Parameters
- outcomes: tf.Tensor, dtype Not specified (converted via `tf.convert_to_tensor`), shape (B, n2) or (n2,) where $n2$ is the outcomes feature size.
- comm: tf.Tensor, dtype Not specified (converted via `tf.convert_to_tensor`), shape (B, m) or (m,) where $m$ is the communication feature size.
- training: bool, constraint: True or False, scalar; forwarded to Dense layers.

Returns
- tf.Tensor, dtype Not specified, shape (B, 1) if `outcomes` rank is 2; shape (1,) if `outcomes` rank is 1 (squeezed along axis 0).

Behavior
- Converts `outcomes` and `comm` to tensors.
- If `outcomes` has rank 1, expands to shape (1, n2) and remembers to squeeze the output back to rank 1.
- If `comm` has rank 1, expands to shape (1, m).
- Concatenates `[outcomes, comm]` along the last axis, applies the hidden Dense layers (ReLU), then applies the final Dense(1) to produce a logit.

Errors
- TensorFlow runtime errors may occur if ranks are not 1 or 2, or if batch dimensions are incompatible for concatenation, or if Dense layers receive incompatible input shapes.

## Data & State

- comms_size: int, constraint: $comms\_size \ge 0$, scalar; stored for shape-check intent (no explicit checks are implemented).
- hidden_units: tuple[int, ...], constraint: each element is an int, shape (k,); normalized from the constructor input.
- _mlp: list[tf.keras.layers.Layer], constraint: Dense layers, shape (k,); the hidden layers (each `tf.keras.layers.Dense(u, activation="relu")`).
- _out: tf.keras.layers.Dense, constraint: units=1 and activation=None, scalar layer instance.

## Planned (design-spec)

- Not specified.

## Deviations

- The module docstring states `comms_size` is "used for shape checks only", but the implementation stores `self.comms_size` without performing any shape validation against `comm`.

## Notes for Contributors

- Keep tensor rank handling consistent: currently only rank-1 inputs are expanded to a batch of 1; other ranks are not explicitly handled.
- If adding shape checks involving `comms_size`, ensure they work for both batched and unbatched inputs and do not break graph execution.

## Related

- `_normalize_hidden_units(hidden_units: int | Sequence[int]) -> tuple[int, ...]` (module-private helper; not part of the public API).

## Changelog

- 0.1: Initial implementation of `LinCombineLayerB` as a small MLP producing a single shoot logit from concatenated `(outcomes, comm)` inputs.