# PyrMeasurementLayerA

> Role: Trainable Keras layer mapping a cropped field state tensor to measurement logits via a 2-layer MLP.

Location: `Q_Sea_Battle.pyr_measurement_layer_a.PyrMeasurementLayerA`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| hidden_units | int, constraint $\ge 1$, scalar | Width of the hidden Dense layer. |
| name | Optional[str], constraint: Keras layer name or None, scalar | Optional Keras layer name. |
| dtype | Optional[tf.dtypes.DType], constraint: valid TensorFlow dtype or None, scalar | Optional dtype for the layer and its sublayers. |
| **kwargs | Any, constraint: forwarded to `tf.keras.layers.Layer`, variadic | Additional keyword arguments forwarded to the Keras base `Layer`. |

### Preconditions

- `hidden_units` is an `int` with constraint $\ge 1$, scalar.

### Postconditions

- `self.hidden_units` is set to `int(hidden_units)`, scalar.
- The sublayers `self._dense_hidden` and `self._dense_out` remain `None` until `build(...)` is called.

### Errors

- Raises `ValueError` if `hidden_units < 1`.

### Example

!!! example "Instantiate the layer"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.pyr_measurement_layer_a import PyrMeasurementLayerA

    layer = PyrMeasurementLayerA(hidden_units=64, dtype=tf.float32)
    ```

## Public Methods

### build

- Signature: `build(input_shape: Any) -> None`

Creates sublayers based on the input width $L$ (the last dimension of `input_shape`), producing an output width $L/2$.

#### Arguments

- `input_shape`: Any, constraint: convertible to `tf.TensorShape` with statically known last dimension $L$, shape (Not specified).

#### Returns

- `None`, constraint: no return value, scalar.

#### Preconditions

- `input_shape` can be converted to `tf.TensorShape`.
- The last dimension $L$ of `input_shape` is statically known.
- $L$ is even so that $L/2$ is an integer.

#### Postconditions

- Creates `self._dense_hidden`: `tf.keras.layers.Dense`, output shape (B, hidden_units) when called on rank-2 input.
- Creates `self._dense_out`: `tf.keras.layers.Dense`, output shape (B, L/2) when called on the hidden activations.
- Sets `self._built_for_L` to `int(L)`, scalar.
- Calls `super().build(input_shape)`.

#### Errors

- Raises `ValueError` if the last dimension $L$ is not statically known.
- Raises `ValueError` if $L$ is not even.

### call

- Signature: `call(field_batch: tf.Tensor, training: bool = False, **kwargs: Any) -> tf.Tensor`

Runs the forward pass: `(B, L) -> (B, hidden_units) -> (B, L/2)` and returns logits (no sigmoid).

#### Arguments

- `field_batch`: tf.Tensor, dtype float32 (if `self.dtype` is None) or `self.dtype` (via `tf.convert_to_tensor`), shape (B, L).
- `training`: bool, constraint: standard Keras training flag, scalar.
- `**kwargs`: Any, constraint: unused (accepted for Keras compatibility), variadic.

#### Returns

- `meas_logits`: tf.Tensor, dtype float32 (if `self.dtype` is None) or `self.dtype`, shape (B, L/2).

#### Preconditions

- If `field_batch` has a statically known rank, it is rank-2 (B, L).
- The last dimension $L$ is even (enforced with a runtime assertion).

#### Postconditions

- Returns `meas_logits = self._dense_out(self._dense_hidden(x))`, where `x` is `field_batch` converted to a tensor with dtype `self.dtype` or `tf.float32`.

#### Errors

- Raises `ValueError` if the input rank is statically known and not 2.
- Raises `RuntimeError` if the layer is missing sublayers (`self._dense_hidden` or `self._dense_out` is `None`).
- May raise TensorFlow assertion errors if the runtime check fails: $L \bmod 2 = 0$.

### get_config

- Signature: `get_config() -> Dict[str, Any]`

Returns the serializable layer configuration, including `hidden_units`.

#### Arguments

- None.

#### Returns

- `cfg`: Dict[str, Any], constraint: Keras-serializable configuration dictionary, shape (Not applicable).

#### Preconditions

- None specified.

#### Postconditions

- The returned dict includes key `"hidden_units"` with value `self.hidden_units`.

#### Errors

- Not specified.

## Data & State

- `hidden_units`: int, constraint $\ge 1$, scalar; number of hidden units in the intermediate Dense layer.
- `_dense_hidden`: Optional[tf.keras.layers.Dense], constraint: `None` before `build(...)`, scalar reference; hidden Dense sublayer created in `build(...)`.
- `_dense_out`: Optional[tf.keras.layers.Dense], constraint: `None` before `build(...)`, scalar reference; output Dense sublayer created in `build(...)`.
- `_built_for_L`: Optional[int], constraint: `None` before `build(...)`, scalar; input width $L$ used to build the layer.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- Sublayers are intentionally created in `build(...)` because the output dimension depends on the input width $L$.
- The output head produces logits (no sigmoid); downstream components are expected to apply any sigmoid/DRU/etc. as needed.
- The runtime even-width constraint is enforced in `call(...)` using `tf.debugging.assert_equal`, which can catch dynamic-shape mismatches even if `build(...)` succeeded.

## Related

- `tf.keras.layers.Layer`
- `tf.keras.layers.Dense`

## Changelog

- Not specified.