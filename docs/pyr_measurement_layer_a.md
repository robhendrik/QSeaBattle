# PyrMeasurementLayerA

> Role: Trainable Keras layer that maps a rank-2 field batch to rank-2 measurement probabilities with output dimension $L/2$.

Location: `Q_Sea_Battle.pyr_measurement_layer_a.PyrMeasurementLayerA`

## Derived constraints

- Let $L$ be the last dimension of `field_batch`; $L$ must be statically known at build time and must be even.
- Output last dimension is $L/2$; output values are in $[0, 1]$ due to a sigmoid activation.

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| hidden_units | int, constraint $\ge 1$, scalar | Number of units in the hidden Dense layer. |
| name | str \| None, scalar | Layer name passed to `tf.keras.layers.Layer`. |
| dtype | tf.dtypes.DType \| None, scalar | Dtype for the layer and its sublayers; if `None`, call-time conversion defaults to `tf.float32`. |
| **kwargs | Any, variadic mapping | Forwarded to `tf.keras.layers.Layer` constructor. |

### Preconditions

- `hidden_units >= 1`.

### Postconditions

- The instance is created with `trainable=True`.
- The sublayers are not created until `build()` is called; internal sublayer references are initialized to `None`.

### Errors

- `ValueError`: if `hidden_units < 1`.

### Example

!!! example "Constructing the layer"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.pyr_measurement_layer_a import PyrMeasurementLayerA

    layer = PyrMeasurementLayerA(hidden_units=64, dtype=tf.float32)

    x = tf.zeros([8, 10], dtype=tf.float32)  # B=8, L=10 (even)
    y = layer(x, training=False)
    print(y.shape)  # (8, 5)
    ```

## Public Methods

### build

- Signature: `build(input_shape: Any) -> None`

Parameters:

- `input_shape`: Any, constraint: convertible to `tf.TensorShape`, shape: not specified; expected to describe an input with statically known last dimension $L$.

Returns:

- `None`.

Preconditions:

- The last dimension $L$ of `input_shape` is statically known (`shape[-1] is not None`).
- $L$ is even.

Postconditions:

- Creates two sublayers:
  - `Dense(hidden_units, activation="relu", name="dense_hidden")`
  - `Dense(L/2, activation="sigmoid", name="dense_out")`
- Records `_built_for_L = L`.

Errors:

- `ValueError`: if the last dimension is unknown at build time.
- `ValueError`: if $L$ is not even.

### call

- Signature: `call(field_batch: tf.Tensor, training: bool = False, **kwargs: Any) -> tf.Tensor`

Parameters:

- `field_batch`: tf.Tensor, dtype float32 (or `self.dtype` if set), shape (B, L); constraint: rank must be 2 and $L$ must be even.
- `training`: bool, scalar; forwarded to the Dense sublayers.
- `**kwargs`: Any, variadic mapping; accepted but not used by this implementation.

Returns:

- `meas_a`: tf.Tensor, dtype float32 (or `self.dtype` if set), shape (B, L/2); constraint: values in $[0, 1]$ (sigmoid output).

Preconditions:

- `field_batch` must be rank-2.
- The last dimension $L$ must be even.
- The layer must have been built such that internal sublayers exist.

Postconditions:

- Produces measurement probabilities via an MLP: hidden ReLU Dense followed by sigmoid Dense.

Errors:

- `ValueError`: if `field_batch` has a statically known rank that is not 2.
- `tf.errors.InvalidArgumentError` (or framework equivalent): if `tf.shape(field_batch)[-1] % 2 != 0` at runtime due to `tf.debugging.assert_equal`.
- `RuntimeError`: if internal sublayers are missing (layer not built correctly).

### get_config

- Signature: `get_config() -> Dict[str, Any]`

Parameters:

- None.

Returns:

- `Dict[str, Any]`, mapping containing the base Keras layer config plus `{"hidden_units": int}`.

## Data & State

- `hidden_units`: int, constraint $\ge 1$, scalar; stored constructor hyperparameter.
- `_dense_hidden`: tf.keras.layers.Dense \| None; created in `build()`, otherwise `None`.
- `_dense_out`: tf.keras.layers.Dense \| None; created in `build()`, otherwise `None`.
- `_built_for_L`: int \| None; the input last dimension $L$ the layer was built for, set in `build()`.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- Sublayers must be created in `build()` (not in `call()`), and `call()` must not create state.
- The output is probabilities (sigmoid), not logits, to satisfy downstream validation of outcomes in $[0, 1]$.
- The contract requires `field_batch` shape (B, L) and output shape (B, L/2); ensure any future changes preserve this public API.

## Related

- `tf.keras.layers.Layer`
- `tf.keras.layers.Dense`
- `tf.debugging.assert_equal`

## Changelog

- Not specified.