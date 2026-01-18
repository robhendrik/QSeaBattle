# PRAssistedLayer

> Role: Stateless Keras layer that produces correlated two-party outcomes from current/previous measurements and previous outcomes, in either expected-value or deterministic stateless-sampling mode.

Location: `Q_Sea_Battle.pr_assisted_layer.PRAssistedLayer`

## Constructor

| Parameter | Type | Description |
|---|---|---|
| `length` | `int`, constraint: `> 0`, shape: scalar | Number of bits in each measurement/outcome vector (the required size of the last dimension of input tensors). |
| `p_high` | `float`, constraint: $0 \le p\_high \le 1$, shape: scalar | Correlation parameter used to compute the probability of matching the previous outcome on the second measurement. |
| `mode` | `str`, constraint: one of `{"expected","sample"}`, shape: scalar | Operation mode: `"expected"` returns deterministic expected outcomes in $[0,1]`; `"sample"` returns sampled binary outcomes in `{0,1}`. |
| `resource_index` | `Optional[int]`, constraint: `None` or any `int`, shape: scalar | Optional identifier mixed into the stateless RNG seed derivation. |
| `seed` | `Optional[int]`, constraint: `None` or any `int`, shape: scalar | Optional base seed for deterministic stateless sampling; if `None`, sampling is process-local and may be non-deterministic across processes. |
| `name` | `Optional[str]`, constraint: `None` or any `str`, shape: scalar | Optional layer name passed to the Keras `Layer` constructor. |
| `**kwargs` | `Any`, constraint: Keras `Layer` kwargs, shape: N/A | Forwarded to `tf.keras.layers.Layer`. |

Preconditions

- `length > 0`.
- `p_high` is within `[0, 1]` (after `float(p_high)` conversion).
- `mode` is either `"expected"` or `"sample"`.

Postconditions

- `self.length: int` is set to `int(length)`.
- `self.p_high: float` is set to `float(p_high)`.
- `self.mode: str` is set to `mode`.
- `self.resource_index: Optional[int]` is set to `None` or `int(resource_index)`.
- `self.seed: Optional[int]` is set to `None` or `int(seed)`.
- No trainable variables are created by this constructor (stateless behavior is by design, but variable absence is not explicitly asserted in code).

Errors

- Raises `ValueError` if `length <= 0`.
- Raises `ValueError` if `p_high` is not in `[0, 1]`.
- Raises `ValueError` if `mode` is not one of `{"expected", "sample"}`.

Example

!!! example "Construct a layer"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.pr_assisted_layer import PRAssistedLayer

    layer = PRAssistedLayer(length=8, p_high=0.9, mode="expected", resource_index=0, seed=123)

    inputs = {
        "current_measurement": tf.zeros((2, 8), dtype=tf.float32),
        "previous_measurement": tf.zeros((2, 8), dtype=tf.float32),
        "previous_outcome": tf.zeros((2, 8), dtype=tf.float32),
        "first_measurement": tf.ones((2, 1), dtype=tf.float32),
    }
    y = layer(inputs)  # tf.Tensor, dtype float32, shape (2, 8)
    ```

## Public Methods

### `get_config() -> Dict[str, Any]`

Return layer config for Keras serialization.

Parameters

- None.

Returns

- `Dict[str, Any]`, constraints: JSON-serializable-by-Keras values for this layer’s constructor fields; shape: mapping with scalar values under keys `{"length","p_high","mode","resource_index","seed"}` plus base Keras config fields.

Errors

- Not specified.

### `call(inputs: Dict[str, tf.Tensor]) -> tf.Tensor`

Compute correlated outcomes.

Parameters

- `inputs`: `Dict[str, tf.Tensor]`, constraints: keys must match exactly `{"current_measurement","previous_measurement","previous_outcome","first_measurement"}`; shapes and dtypes are normalized internally as follows: `current_measurement`, `previous_measurement`, `previous_outcome` are converted to `tf.Tensor, dtype float32, shape (..., length)` with soft range assertions $[0,1]$; `first_measurement` is converted to `tf.Tensor, dtype float32, shape broadcastable to (..., length)` and interpreted as boolean via `first_measurement >= 0.5`.

Returns

- `tf.Tensor`, dtype `float32`, shape `(..., length)`, constraints: in `mode="expected"` values are in $[0,1]$; in `mode="sample"` values are in `{0,1}` after rounding.

Errors

- Raises `ValueError` if `inputs` keys do not match the required set exactly (missing or extra keys).
- May raise `tf.errors.InvalidArgumentError` (via `tf.debugging.assert_equal`) if the last dimension of any of `current_measurement`, `previous_measurement`, or `previous_outcome` is not equal to `length`.
- May raise `tf.errors.InvalidArgumentError` (via `tf.debugging.assert_greater_equal` / `assert_less_equal`) if any of `current_measurement`, `previous_measurement`, or `previous_outcome` contain values outside `[0, 1]`.
- In `mode="sample"`, may raise `tf.errors.InvalidArgumentError` (via `tf.debugging.assert_near`) if `previous_outcome` is not binary (not near its rounded value).

!!! note "Input semantics"
    The layer does not maintain internal state about measurement ordering; the caller must provide `first_measurement` (first vs second) and pass the corresponding `previous_measurement` and `previous_outcome` tensors.

## Data & State

- `length`: `int`, constraint: `> 0`, shape: scalar; the expected size of the last dimension of measurement/outcome vectors.
- `p_high`: `float`, constraint: $0 \le p\_high \le 1$, shape: scalar; correlation parameter used in the second-measurement rule.
- `mode`: `str`, constraint: one of `{"expected","sample"}`, shape: scalar.
- `resource_index`: `Optional[int]`, constraint: `None` or any `int`, shape: scalar; mixed into stateless RNG seed derivation.
- `seed`: `Optional[int]`, constraint: `None` or any `int`, shape: scalar; base seed for stateless RNG, with a process-local fallback when `None`.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- Input validation is strict on keys: `_validate_inputs` requires the input dictionary keys to match exactly; adding optional keys will currently error.
- Shape checks enforce only the last dimension equality to `length`; leading dimensions are unconstrained as long as broadcasting of `first_measurement` to `(..., length)` is possible.
- `mode="sample"` uses `tf.random.stateless_uniform` with a seed derived from `seed`, `resource_index`, and a `stream_id`; when `seed is None` the base seed is drawn from `tf.random.uniform((), ...)`, which is process-local and may vary across processes.

## Related

- TensorFlow Keras Layer base class: `tf.keras.layers.Layer`
- Internal helper dataclass (private): `_PRInputs`

## Changelog

- 0.1: Initial implementation of `PRAssistedLayer` with `"expected"` and `"sample"` modes, strict input validation, and stateless seed derivation.