# PRAssistedLayer

> **Role**: Stateless Keras layer that produces first/second measurement outcomes for a PR-assisted resource.

**Location**: `Q_Sea_Battle.pr_assisted_layer.PRAssistedLayer`

!!! note "Derived constraints"
    Let `field_size = n` and `n2 = n**2`. Let `comms_size = m`.

    `PRAssistedLayer` is independent of `GameLayout`. The only shape constraint is `length > 0` and that the last
    dimension of measurement tensors equals `length`.

## Constructor

| Parameter | Type | Description |
|---|---|---|
| `length` | `int`, scalar | Number of bits in each measurement and outcome vector. Must be `> 0`. |
| `p_high` | `float`, scalar | Correlation parameter in `[0.0, 1.0]`. |
| `mode` | `str`, one of `{"expected","sample"}` | Output mode. See `call(...)`. Default is `"expected"`. |
| `resource_index` | `int` or `None`, scalar | Optional stream identifier to decorrelate multiple resources. |
| `seed` | `int` or `None`, scalar | Optional base seed for stateless RNG in `"sample"` mode. |
| `name` | `str` or `None`, scalar | Optional Keras layer name. |
| `**kwargs` | `dict[str, Any]`, scalar | Forwarded to `tf.keras.layers.Layer`. |

**Preconditions**
- `length > 0`.
- `0.0 <= p_high <= 1.0`.
- `mode` is `"expected"` or `"sample"`.

**Postconditions**
- `self.length`: `int`, scalar, equals `length`.
- `self.p_high`: `float`, scalar.
- `self.mode`: `str`, scalar.
- `self.resource_index`: `int` or `None`, scalar.
- `self.seed`: `int` or `None`, scalar.

**Errors**
- Raises `ValueError` if `length <= 0`.
- Raises `ValueError` if `p_high` is outside `[0.0, 1.0]`.
- Raises `ValueError` if `mode` is not `"expected"` or `"sample"`.

## Methods

### `call`

Compute outcomes for either the first or second measurement.

**Signature**
- `call(inputs: dict[str, tf.Tensor]) -> tf.Tensor`

**Arguments**
- `inputs`: `dict[str, tf.Tensor]`, scalar, with exactly these keys:
  - `current_measurement`: `tf.Tensor`, dtype `float32`, shape `(..., length)`, values in `[0.0, 1.0]`.
  - `previous_measurement`: `tf.Tensor`, dtype `float32`, shape `(..., length)`, values in `[0.0, 1.0]`.
  - `previous_outcome`: `tf.Tensor`, dtype `float32`, shape `(..., length)`, values in `[0.0, 1.0]`.
  - `first_measurement`: `tf.Tensor`, dtype `float32`, shape `(..., 1)` or broadcastable to `(..., length)`,
    where values `>= 0.5` indicate "first measurement".

**Returns**
- `outcome`: `tf.Tensor`, dtype `float32`, shape `(..., length)`.
  - If `mode == "expected"`: values in `[0.0, 1.0]`.
  - If `mode == "sample"`: values in `{0.0, 1.0}` (binary, returned as float32).

**Preconditions**
- `inputs` contains exactly the required keys (no missing or extra keys).
- `tf.shape(current_measurement)[-1] == length`.
- `tf.shape(previous_measurement)[-1] == length`.
- `tf.shape(previous_outcome)[-1] == length`.
- `current_measurement`, `previous_measurement`, `previous_outcome` are in `[0.0, 1.0]`.
- If `mode == "sample"`:
  - `previous_outcome` is binary: `previous_outcome == round(previous_outcome)` (within tolerance).

**Postconditions**
- Returns:
  - For first measurements: uniform random bits in `"sample"` mode, or `0.5` in `"expected"` mode.
  - For second measurements: correlated outcomes governed by `p_high` and the `(prev, current)` settings.
- No internal state is mutated (stateless; safe to call in functional graphs).

**Errors**
- Raises `ValueError` if the input key set is not exactly the required set.
- Raises `tf.errors.InvalidArgumentError` if shape assertions fail.
- Raises `tf.errors.InvalidArgumentError` if range assertions fail.
- Raises `tf.errors.InvalidArgumentError` in `"sample"` mode when `previous_outcome` is not binary.

!!! note "Correlation rule"
    For each index `i`, the probability to keep the previous outcome is:

    - If `(prev_measurement[i], current_measurement[i])` is in `{(0,0), (0,1), (1,0)}`, keep with probability
      `p_high`.
    - If `(1,1)`, keep with probability `1 - p_high`.

### `get_config`

Return layer config for Keras serialization.

**Signature**
- `get_config() -> dict[str, Any]`

**Arguments**
- None.

**Returns**
- `config`: `dict[str, Any]`, scalar, containing at least:
  - `length`: `int`, scalar.
  - `p_high`: `float`, scalar.
  - `mode`: `str`, scalar.
  - `resource_index`: `int` or `None`, scalar.
  - `seed`: `int` or `None`, scalar.

**Preconditions**
- None.

**Postconditions**
- Returned config can be used by Keras to reconstruct the layer.

**Errors**
- None (propagates unexpected Keras base-class errors).

## Data & State

- Attributes (public):
  - `length`: `int`, scalar.
  - `p_high`: `float`, scalar.
  - `mode`: `str`, one of `{"expected","sample"}`.
  - `resource_index`: `int` or `None`, scalar.
  - `seed`: `int` or `None`, scalar.
- Side effects: None (uses stateless RNG for sampling).
- Thread-safety: Stateless; safe for concurrent calls in graph mode.

## Notes for Contributors

- Keep the input key set exact; downstream code relies on strict validation.
- In `"sample"` mode, the layer enforces binary `previous_outcome` to avoid silently sampling from relaxed outcomes.
- `resource_index` and `seed` jointly define the stateless RNG streams. If you add more RNG calls, increment stream IDs
  deterministically to maintain reproducibility.

## Related
- See also: [PRAssisted](PRAssisted.md)

## Changelog
- 2026-01-11 — Author: Rob Hendriks
