# LinCombineLayerA

> Role: Learnable TensorFlow Keras layer that maps measurement outcomes to communication logits.

Location: `Q_Sea_Battle.lin_combine_layer_a.LinCombineLayerA`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| comms_size | int, constraint: $m \ge 1$, shape: scalar | Number of communication channels ($m$). Stored as `self.comms_size` after `int(...)`. |
| hidden_units | int or Sequence[int], constraint: each value $u \ge 1$, shape: scalar or $(L,)$ | Hidden layer widths for an MLP; an int creates a single hidden layer, a sequence creates a stack of Dense-ReLU layers. Normalized to `tuple[int, ...]` and stored as `self.hidden_units`. |
| name | str or None, constraint: if None then defaults to `"LinCombineLayerA"`, shape: scalar | Keras layer name passed to `tf.keras.layers.Layer.__init__`. |
| **kwargs | dict[str, Any], constraint: must be accepted by `tf.keras.layers.Layer`, shape: mapping | Forwarded to the Keras base layer constructor. |

Preconditions

- `comms_size` must be convertible to `int`.
- `hidden_units` must be an `int` or a `collections.abc.Sequence` of values convertible to `int`.

Postconditions

- `self.comms_size: int` is set.
- `self.hidden_units: tuple[int, ...]` is set.
- `self._mlp: list[tf.keras.layers.Layer]` is created as a list of `tf.keras.layers.Dense(..., activation="relu")` layers.
- `self._out: tf.keras.layers.Dense` is created with `units=self.comms_size` and `activation=None`.

Errors

- Raises `TypeError` if `hidden_units` is not an `int` and not a `Sequence`.
- Raises `ValueError` if any element of `hidden_units` cannot be converted to `int` (or if `comms_size` cannot be converted to `int`), as raised by `int(...)` conversion.
- Any additional errors may be raised by `tf.keras.layers.Layer.__init__` when invalid `name`/`kwargs` are provided.

!!! example "Example"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.lin_combine_layer_a import LinCombineLayerA

    layer = LinCombineLayerA(comms_size=8, hidden_units=(64, 64))

    outcomes = tf.random.uniform(shape=(32, 100), dtype=tf.float32)  # (B, n2)
    logits = layer(outcomes, training=True)  # (B, m)
    ```

## Public Methods

### call

Signature: `call(self, outcomes: tf.Tensor, training: bool = False) -> tf.Tensor`

Parameters

- outcomes: tf.Tensor, dtype: not specified (converted via `tf.convert_to_tensor`), shape $(B, n2)$ or $(n2,)$.
- training: bool, constraint: boolean, shape: scalar.

Returns

- tf.Tensor, dtype: not specified, shape $(B, m)$ if input rank is 2, else shape $(m,)$ if input rank is 1.

Behavior

- Converts `outcomes` to a tensor with `tf.convert_to_tensor`.
- If `outcomes` has rank 1 (shape $(n2,)$), expands to shape $(1, n2)$, runs the MLP and output layer, then squeezes axis 0 to return shape $(m,)$.
- If `outcomes` has rank 2 (shape $(B, n2)$), returns logits of shape $(B, m)$.
- Applies each hidden Dense layer in `self._mlp` with `activation="relu"`, then applies `self._out` Dense with `activation=None` (logits).

Errors

- May raise TensorFlow/Keras shape or rank errors if `outcomes` has rank other than 1 or 2, or if Dense layers cannot be applied to the provided shape/dtype.

## Data & State

- self.comms_size: int, constraint: not validated in code beyond `int(...)`, shape: scalar.
- self.hidden_units: tuple[int, ...], constraint: elements are `int` after normalization, shape: $(L,)$.
- self._mlp: list[tf.keras.layers.Layer], constraint: list length $L = \text{len}(\text{self.hidden_units})$, shape: list.
- self._out: tf.keras.layers.Dense, constraint: units $= m$, shape: layer object.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- `_normalize_hidden_units(hidden_units)` is a private helper that normalizes `hidden_units` to `tuple[int, ...]`; keep it consistent with any future serialization/config logic if added.
- The `call` method explicitly supports rank-1 input by expanding and later squeezing; changes to rank handling should preserve the documented input/output shape conventions $(n2,) \leftrightarrow (m,)$ and $(B, n2) \leftrightarrow (B, m)$.

## Related

- TensorFlow Keras base class: `tf.keras.layers.Layer`
- Dense layers used internally: `tf.keras.layers.Dense`

## Changelog

- 0.1: Initial implementation of `LinCombineLayerA` with configurable hidden MLP and linear output logits.