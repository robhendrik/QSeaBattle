# LinCombineLayerA

> Role: Learnable mapping from measurement outcomes to communication logits via a minimal MLP and final linear projection.

Location: `Q_Sea_Battle.lin_combine_layer_a.LinCombineLayerA`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| comms_size | int, constraint: convertible via `int(comms_size)`; scalar | Number of communication channels ($m$). |
| hidden_units | int \| collections.abc.Sequence[int], constraint: if int then one hidden layer; if sequence then one width per hidden layer; scalar / 1D sequence | Hidden-layer configuration for a Dense+ReLU stack; normalized internally to `tuple[int, ...]`. |
| name | str \| None, constraint: if None defaults to `"LinCombineLayerA"`; scalar | Optional layer name. |
| **kwargs | Any, constraint: forwarded to `tf.keras.layers.Layer`; shape: N/A | Additional keyword arguments passed to the base `Layer` constructor. |

Preconditions

- `comms_size` must be a value acceptable to `int()` and suitable as the `units` argument to `tf.keras.layers.Dense`.
- `hidden_units` must be an `int` or a sequence of values each acceptable to `int()` and suitable as the `units` argument to `tf.keras.layers.Dense`.

Postconditions

- `self.comms_size` is set to `int(comms_size)`.
- `self.hidden_units` is set to a `tuple[int, ...]` derived from `hidden_units`.
- `self._mlp` is a `list[tf.keras.layers.Layer]` of `Dense` layers with `activation="relu"` and widths from `self.hidden_units`.
- `self._out` is a `tf.keras.layers.Dense` with `units=self.comms_size` and `activation=None`.

Errors

- Not specified (constructor may raise exceptions from `int(...)` conversions and from `tf.keras.layers.Dense` initialization if arguments are invalid).

!!! example "Example"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.lin_combine_layer_a import LinCombineLayerA

    layer = LinCombineLayerA(comms_size=8, hidden_units=(64, 64))
    outcomes = tf.random.uniform((32, 10))  # (B, n2)
    comm_logits = layer(outcomes, training=True)  # (B, m)
    ```

## Public Methods

### call

Compute communication logits from measurement outcomes.

Arguments

- outcomes: tf.Tensor, dtype not specified, shape (B, n2) or (n2,), where B is batch size and n2 is the outcomes vector length.
- training: bool, constraint: standard Keras training flag; scalar.

Returns

- tf.Tensor, dtype not specified, shape (B, m) if input was batched, otherwise shape (m,), where $m = \text{comms\_size}$.

Behavior

- Converts `outcomes` via `tf.convert_to_tensor(outcomes)`.
- If `outcomes` is rank-1 (shape (n2,)), promotes to shape (1, n2) for processing and then squeezes the leading dimension to preserve the unbatched output contract.
- Applies each Dense+ReLU layer in `self._mlp` sequentially, then applies `self._out` to produce logits.

Errors

- Not specified (may raise TensorFlow/Keras runtime errors for incompatible shapes, invalid ranks, or layer build issues).

## Data & State

- comms_size: int, constraint: set to `int(comms_size)`; scalar; number of communication channels ($m$).
- hidden_units: tuple[int, ...], constraint: each element derived via `int(u)`; shape (L,), where L is the number of hidden layers.
- _mlp: list[tf.keras.layers.Layer], constraint: each element is a `tf.keras.layers.Dense` with `activation="relu"`; length L.
- _out: tf.keras.layers.Dense, constraint: `units == comms_size` and `activation is None`; scalar object reference.

## Planned (design-spec)

- Not specified.

## Deviations

- No deviations identified between the module docstring "Design agreements" and the implemented behavior.

## Notes for Contributors

- The unbatched input path is implemented by rank check (`x.shape.rank == 1`) and explicit expand/squeeze; ensure any future changes preserve the caller-visible output shape contract for both (B, n2) and (n2,) inputs.
- The helper `_normalize_hidden_units` is internal (name starts with `_`) and normalizes `hidden_units` to `tuple[int, ...]`; changes to its behavior should be reflected in constructor documentation.

## Related

- TensorFlow Keras base class: `tf.keras.layers.Layer`
- Dense layers used internally: `tf.keras.layers.Dense`
- Internal helper: `_normalize_hidden_units` (module-private)

## Changelog

- Not specified.