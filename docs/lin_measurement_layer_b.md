# LinMeasurementLayerB

> Role: Learnable mapping from a flattened gun vector to per-cell measurement probabilities in $[0, 1]$.

Location: `Q_Sea_Battle.lin_measurement_layer_b.LinMeasurementLayerB`

## Derived constraints

- Let `n2` be the flattened field size (number of cells); `n2` must be a positive `int`.
- `call()` accepts rank-1 or rank-2 inputs only; the last dimension must equal `n2` when statically known.

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| n2 | int, constraint: `n2 > 0`, shape: scalar | Number of cells in the flattened representation; determines the output width of the final dense layer. |
| hidden_units | Sequence[int], constraint: each element convertible to `int`, shape: `(L,)` | Sizes of intermediate dense layers; each layer uses ReLU activation. Default: `(64,)`. |
| name | Optional[str], constraint: any valid Keras layer name or `None`, shape: scalar | Layer name passed to `tf.keras.layers.Layer`. Default: `"LinMeasurementLayerB"`. |
| **kwargs | dict[str, Unknown], constraint: passed through to `tf.keras.layers.Layer`, shape: mapping | Additional Keras layer keyword arguments. |

Preconditions: `n2 > 0`.

Postconditions: `self.n2 == int(n2)`; `self.hidden_units` is a `tuple[int, ...]`; internal MLP layer list is initialized empty and is constructed on first `build()`.

Errors: Raises `ValueError` if `n2 <= 0`.

!!! example "Example"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.lin_measurement_layer_b import LinMeasurementLayerB

    n2 = 100
    layer = LinMeasurementLayerB(n2=n2, hidden_units=(64, 32))

    guns_batch = tf.random.uniform(shape=(8, n2), dtype=tf.float32)
    probs_batch = layer(guns_batch, training=True)  # shape (8, n2)

    guns_single = tf.random.uniform(shape=(n2,), dtype=tf.float32)
    probs_single = layer(guns_single)  # shape (n2,)
    ```

## Public Methods

### build(input_shape) -> None

Create weights based on input shape.

- Parameter `input_shape`: Unknown, constraint: Keras-compatible input shape descriptor, shape: Not specified.
- Returns: `None`, constraint: not applicable, shape: not applicable.

Preconditions: None specified.

Postconditions: If not already built, appends `len(hidden_units)` hidden `tf.keras.layers.Dense` layers with ReLU activation and one output `tf.keras.layers.Dense` layer with `units == n2` and sigmoid activation; sets internal built flag; calls `super().build(input_shape)`.

Errors: Not specified.

### call(guns, training: bool = False)

Forward pass.

- Parameter `guns`: `tf.Tensor`-convertible, dtype: any (non-floating will be cast to `tf.float32`), shape: `(B, n2)` or `(n2,)`.
- Parameter `training`: `bool`, constraint: any boolean, shape: scalar.
- Returns: `tf.Tensor`, dtype: floating (cast to `tf.float32` if input is non-floating), constraint: elementwise in $[0, 1]$, shape: same as `guns` (returns `(B, n2)` for batched input and `(n2,)` for rank-1 input).

Preconditions: Input rank must be 1 or 2; last dimension must equal `n2` when statically known.

Postconditions: Applies the internal MLP (Dense/ReLU layers followed by Dense/sigmoid) to produce probabilities; preserves original rank by temporarily expanding rank-1 inputs and squeezing the output back.

Errors: Raises `ValueError` if input rank is not 1 or 2; raises `ValueError` if the statically known last dimension is not `n2`.

## Data & State

- `n2`: `int`, constraint: `n2 > 0`, shape: scalar; output width and expected input last dimension.
- `hidden_units`: `tuple[int, ...]`, constraint: elements are `int`, shape: `(L,)`; hidden layer widths.
- `_mlp`: `list[tf.keras.layers.Layer]`, constraint: contains Keras layers, shape: `(L+1,)` after build; internal sequence of `Dense` layers.
- `_built_mlp`: `bool`, constraint: boolean, shape: scalar; indicates whether `_mlp` has been constructed.

## Planned (design-spec)

Not specified.

## Deviations

Not specified.

## Notes for Contributors

- `build()` is idempotent via `_built_mlp`; if modifying layer construction, keep repeated calls safe.
- `call()` enforces rank 1 or 2 and validates the last dimension only when statically known (`x.shape[-1] is not None`).

## Related

- `tf.keras.layers.Layer`
- `tf.keras.layers.Dense`

## Changelog

- 0.1: Initial implementation as a learnable sigmoid MLP mapping from gun vectors to per-cell probabilities.