# PyrMeasurementLayerB

> Role: Trainable Keras layer mapping a gun state tensor to measurement probabilities constrained to $[0,1]$ via a sigmoid output head.

Location: `Q_Sea_Battle.pyr_measurement_layer_b.PyrMeasurementLayerB`

## Derived constraints

- Let $L$ be the last dimension of `gun_batch`; $L$ must be statically known at build time and must be even.
- Let $B$ be the batch size (dynamic).
- Output last dimension is $L/2$.

## Constructor

Parameter | Type | Description
--- | --- | ---
hidden_units | int, constraint $\ge 1$ | Number of units in the hidden Dense layer.
name | Optional[str], default None | Layer name passed to the Keras base class.
dtype | Optional[tf.dtypes.DType], default None | Layer dtype; used for internal tensor conversion and Dense sublayers.
**kwargs | Any | Additional keyword arguments forwarded to `tf.keras.layers.Layer`.

Preconditions

- `hidden_units >= 1`.

Postconditions

- `self.hidden_units` is set to `int(hidden_units)`.
- Sublayers `_dense_hidden` and `_dense_out` are initialized to `None` and are created later in `build()`.

Errors

- `ValueError`: if `hidden_units < 1`.

!!! example "Example"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.pyr_measurement_layer_b import PyrMeasurementLayerB

    layer = PyrMeasurementLayerB(hidden_units=64, dtype=tf.float32)
    x = tf.random.uniform(shape=(8, 10), dtype=tf.float32)  # L=10 is even
    y = layer(x, training=True)
    print(y.shape)  # (8, 5)
    ```

## Public Methods

### build

- Signature: `build(input_shape: Any) -> None`

Parameters

- input_shape: Any, constraints: convertible to `tf.TensorShape`; shape must have a statically known last dimension $L$ and $L$ must be even.

Returns

- None: NoneType, no value returned.

Behavior

- Creates two Dense sublayers based on the inferred gun dimension $L$ from `input_shape`:
  - Hidden layer: `Dense(self.hidden_units, activation="relu")`
  - Output layer: `Dense(out_dim, activation="sigmoid")` where `out_dim = L // 2`
- Records `self._built_for_L = L`.

Errors

- `ValueError`: if the last dimension of `input_shape` is `None` (not statically known).
- `ValueError`: if $L$ is odd (so $L/2$ is not an integer).

### call

- Signature: `call(gun_batch: tf.Tensor, training: bool = False, **kwargs: Any) -> tf.Tensor`

Parameters

- gun_batch: tf.Tensor, dtype float32 or `self.dtype` (via conversion), shape $(B, L)$; constraints: rank must be 2; last dimension $L$ must be even (checked at runtime).
- training: bool, default False; forwarded to internal Dense sublayers.
- **kwargs: Any; accepted but not specified/used directly in the implementation.

Returns

- meas_b: tf.Tensor, dtype float32 or `self.dtype`, shape $(B, L/2)$; constraints: values in $[0, 1]$ due to sigmoid activation.

Errors

- `ValueError`: if `gun_batch` has statically known rank not equal to 2.
- `tf.errors.InvalidArgumentError`: if runtime check finds $L$ is not even (from `tf.debugging.assert_equal` on `tf.shape(x)[-1] % 2`).
- `RuntimeError`: if the layer is not built correctly (either `_dense_hidden` or `_dense_out` is `None`).

### get_config

- Signature: `get_config() -> Dict[str, Any]`

Parameters

- None.

Returns

- cfg: Dict[str, Any], constraints: includes base layer config plus key `"hidden_units"` with an int value.

## Data & State

- hidden_units: int, constraint $\ge 1$; persisted in config via `get_config()`.
- _dense_hidden: Optional[tf.keras.layers.Dense], initialized to None; created in `build()` with `activation="relu"`.
- _dense_out: Optional[tf.keras.layers.Dense], initialized to None; created in `build()` with `activation="sigmoid"` and units $L/2$.
- _built_for_L: Optional[int], initialized to None; set to $L$ in `build()`.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- This layer enforces probability outputs via a sigmoid head; avoid changing the public contract `call(gun_batch) -> meas_b` and the $[0,1]$ output constraint.
- Do not create state in `call()`; sublayers are created in `build()` based on the statically known last dimension $L$.

## Related

- `tf.keras.layers.Layer`
- `tf.keras.layers.Dense`

## Changelog

- Not specified.