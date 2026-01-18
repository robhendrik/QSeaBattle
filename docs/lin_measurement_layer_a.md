# LinMeasurementLayerA

> Role: Learnable mapping from a flattened field vector to per-cell measurement probabilities in $[0, 1]$ via an MLP ending in a sigmoid layer.
Location: `Q_Sea_Battle.lin_measurement_layer_a.LinMeasurementLayerA`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| n2 | int, constraint $n2 > 0$ | Field vector length (number of cells). Used as the output dimension of the final Dense layer and validated against the input last dimension in `call`. |
| hidden_units | Sequence[int], elements constraint each $u$ castable to int | Hidden layer widths for the MLP; each element produces one `tf.keras.layers.Dense(u, activation="relu")` in `build`. Default: `(64,)`. |
| name | Optional[str], constraint either `None` or any string | Layer name passed to `tf.keras.layers.Layer`. Default: `"LinMeasurementLayerA"`. |
| **kwargs | dict[str, Any], not specified | Forwarded to `tf.keras.layers.Layer` superclass constructor. |

Preconditions: `n2 > 0`.  
Postconditions: `self.n2` is set to `int(n2)`; `self.hidden_units` is set to `tuple(int(u) for u in hidden_units)`; `_mlp` is initialized empty and `_built_mlp` is `False`.  
Errors: Raises `ValueError` if `n2 <= 0`.  

!!! example "Example"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.lin_measurement_layer_a import LinMeasurementLayerA
    
    n2 = 100
    layer = LinMeasurementLayerA(n2=n2, hidden_units=(64, 64))
    
    x = tf.zeros((8, n2), dtype=tf.float32)
    y = layer(x, training=True)
    assert y.shape == x.shape
    ```

## Public Methods

### build

- Signature: `build(self, input_shape) -> None`
- Parameters:
  - `input_shape`: Unknown, not specified; passed through to `tf.keras.layers.Layer.build`.
- Returns: None.
- Behavior: Lazily constructs the MLP exactly once; for each `u` in `self.hidden_units`, appends a `tf.keras.layers.Dense(u, activation="relu")`, then appends a final `tf.keras.layers.Dense(self.n2, activation="sigmoid")`; sets `_built_mlp = True` and calls `super().build(input_shape)`.
- Errors: Not specified.

### call

- Signature: `call(self, fields, training: bool = False)`
- Parameters:
  - `fields`: tf.Tensor-like, dtype any numeric; accepted shapes `(B, n2)` or `(n2,)`; converted via `tf.convert_to_tensor(fields)` and cast to `tf.float32` if not floating.
  - `training`: bool, constraint any boolean; passed as `training=training` to each Dense layer call.
- Returns: `tf.Tensor`, dtype float32, shape `(B, n2)` if input rank is 2, else shape `(n2,)` if input rank is 1; values are in $[0, 1]$ due to sigmoid output activation.
- Errors: Raises `ValueError` if input rank is not 1 or 2; raises `ValueError` if the last dimension is known and is not equal to `self.n2`.

## Data & State

- `n2`: int, constraint $n2 > 0$; field vector length and final output width.
- `hidden_units`: tuple[int, ...], constraint each element is an `int` derived from `hidden_units`; defines hidden Dense layers.
- `_mlp`: list[tf.keras.layers.Layer], initial shape `len(_mlp) == 0` before `build`; after `build`, contains `len(hidden_units) + 1` Dense layers in order.
- `_built_mlp`: bool; `False` before `build`, `True` after the first successful `build`.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- `build` is intentionally idempotent via `_built_mlp`; modifying layer construction should preserve that behavior to avoid duplicate sublayers on repeated builds.
- `call` supports rank-1 inputs by temporarily expanding to rank-2 and then squeezing back; maintain the squeeze/unsqueeze contract if changing shape handling.

## Related

- `tf.keras.layers.Layer`
- `tf.keras.layers.Dense`

## Changelog

- 0.1: Initial implementation of a learnable measurement layer mapping field vectors to per-cell probabilities via an MLP with sigmoid output.