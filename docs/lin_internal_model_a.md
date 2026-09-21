# LinInternalModelA

> Role: Depth-1 (fixed) trainable linear internal model mapping field logits to communication logits while exposing measurement and shared-resource outcome logits.

Location: `Q_Sea_Battle.lin_internal_model_a.LinInternalModelA`

## Derived constraints

- Symbols: `n2` = number of field bits (flattened field size); `m` = number of communication bits (comms size); `B` = batch size.
- Fixed depth: `depth = 1`.
- Input contract: field logits are `tf.Tensor, dtype float32, shape (B, n2)`.
- Output contract: communication logits are `tf.Tensor, dtype float32, shape (B, m)` with constraint $m \ge 1$.
- Internal exposure contract: `compute_with_internal(...)` returns `(comm_logits, meas_list, out_list)` where `meas_list` and `out_list` are Python `list` objects of length 1.

## Constructor

| Parameter | Type | Description |
|---|---|---|
| `game_layout` | `Any`, unconstrained, shape N/A | Game layout/config object used to infer `n2` and comms size `m` (stored as `self.M`). |
| `sr_mode` | `str`, default `"replay"`, shape N/A | Shared resource mode; expected values are those supported by `PRAssistedReplay` (examples mentioned: `"replay"`, `"stochastic"`). |
| `p_rule` | `float`, default `1.0`, shape N/A | Probability of using the rule-based component in the PR-assisted shared resource. |
| `beta` | `float`, default `10.0`, shape N/A | Temperature/sharpness parameter used by the PR-assisted shared resource. |
| `alpha` | `float`, default `5.0`, shape N/A | Mixing/strength parameter used by the PR-assisted shared resource. |
| `seed` | `int \| None`, default `None`, shape N/A | Optional seed forwarded to the shared-resource layer. |
| `measure_layers` | `Optional[Sequence[tf.keras.layers.Layer]]`, default `None`, shape N/A | Optional custom measurement layer sequence; if provided, must have length `depth` (=1). |
| `combine_layers` | `Optional[Sequence[tf.keras.layers.Layer]]`, default `None`, shape N/A | Optional custom combine layer sequence; if provided, must have length `depth` (=1). |
| `hidden_units_measure` | `int`, default `64`, shape N/A | Hidden units used when constructing the default measurement layer (`LinMeasurementLayerA`). |
| `hidden_units_combine` | `int`, default `64`, shape N/A | Hidden units used when constructing the default combine layer (`LinCombineLayerA`). |
| `name` | `Optional[str]`, default `None`, shape N/A | Optional Keras model name passed to `tf.keras.Model`. |

Preconditions

- `_infer_n2_and_m(game_layout)` must return `(n2, m)` assignable to `self.n2` and `self.M`.
- Constraint: `m >= 1`.
- If `measure_layers` is provided, it must satisfy `len(measure_layers) == 1`.
- If `combine_layers` is provided, it must satisfy `len(combine_layers) == 1`.

Postconditions

- `self.depth` is set to `1`.
- `self.n2` and `self.M` are set from `_infer_n2_and_m(game_layout)`.
- `self.measure_layers`, `self.combine_layers`, `self.sr_layers` exist and are Python `list` objects of length 1.
- Convenience aliases exist: `self.measure_layer == self.measure_layers[0]`, `self.combine_layer == self.combine_layers[0]`, `self.sr_layer == self.sr_layers[0]`.

Errors

- `ValueError` if inferred `m < 1`.
- `ValueError` if `measure_layers` is provided and `len(measure_layers) != 1`.
- `ValueError` if `combine_layers` is provided and `len(combine_layers) != 1`.

!!! example "Example"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.lin_internal_model_a import LinInternalModelA

    game_layout = ...  # object understood by _infer_n2_and_m
    model = LinInternalModelA(game_layout, sr_mode="replay", p_rule=1.0, beta=10.0, alpha=5.0)

    x = tf.zeros((4, model.n2), dtype=tf.float32)
    y = model(x, training=False)
    assert y.shape[-1] == model.M
    ```

## Public Methods

### set_alpha

`set_alpha(alpha: float) -> None`

- Purpose: Set the PR-assisted shared-resource `alpha` parameter for all SR layers.

Arguments

- `alpha`: `float`, unconstrained, shape N/A.

Returns

- `None`, shape N/A.

Errors

- `AttributeError` if any SR layer lacks a `set_alpha()` method.

### set_p_rule

`set_p_rule(p_rule: float) -> None`

- Purpose: Set the PR-assisted shared-resource `p_rule` parameter for all SR layers.

Arguments

- `p_rule`: `float`, unconstrained, shape N/A.

Returns

- `None`, shape N/A.

Errors

- `AttributeError` if any SR layer lacks a `set_p_rule()` method.

### set_beta

`set_beta(beta: float) -> None`

- Purpose: Set the PR-assisted shared-resource `beta` parameter for all SR layers.

Arguments

- `beta`: `float`, unconstrained, shape N/A.

Returns

- `None`, shape N/A.

Errors

- `AttributeError` if any SR layer lacks a `set_beta()` method.

### set_sr_mode

`set_sr_mode(sr_mode: str) -> None`

- Purpose: Set the SR mode (e.g., `replay` or `stochastic`) for all SR layers.

Arguments

- `sr_mode`: `str`, expected to be supported by the SR layer, shape N/A.

Returns

- `None`, shape N/A.

Errors

- `AttributeError` if any SR layer lacks a `set_sr_mode()` method.

### call

`call(field_scaled: tf.Tensor, training: bool = False, **kwargs: Any) -> tf.Tensor`

- Purpose: Keras forward pass; delegates to `compute_with_internal(...)` and returns only communication logits.

Arguments

- `field_scaled`: `tf.Tensor, dtype float32, shape (B, n2)`; treated as field logits (name reflects upstream code).
- `training`: `bool`, default `False`, shape N/A.
- `**kwargs`: `Any`, unconstrained, shape N/A; unused extra Keras call kwargs.

Returns

- Communication logits: `tf.Tensor, dtype float32, shape (B, m)`.

### compute_with_internal

`compute_with_internal(field_logits: tf.Tensor, replay_out_a_logits_list: Optional[Sequence[tf.Tensor]] = None, harden_between_levels: bool = False, beta_for_hardening: float = 10.0, training: bool = False) -> Tuple[tf.Tensor, List[tf.Tensor], List[tf.Tensor]]`

- Purpose: Compute communication logits and expose internal measurement and shared-resource outcome logits, returning per-level lists of length 1 (depth fixed to 1).

Arguments

- `field_logits`: `tf.Tensor, dtype float32 (via conversion), shape (B, n2)`; must be rank-2 and trailing dimension must match `self.n2` when statically known.
- `replay_out_a_logits_list`: `Optional[Sequence[tf.Tensor]]`, default `None`, shape N/A; if provided must be a `list` or `tuple` of length 1, and element 0 is converted to `tf.Tensor, dtype float32, shape (B, k)` with runtime constraint $k = \text{measurement\_size}$ (enforced via `tf.debugging.assert_equal` on trailing dimension vs measurement logits).
- `harden_between_levels`: `bool`, default `False`, shape N/A; if `True`, replaces input logits with $\pm \text{beta\_for\_hardening}$ based on sign before measurement (API compatibility behavior).
- `beta_for_hardening`: `float`, default `10.0`, shape N/A; magnitude used when hardening.
- `training`: `bool`, default `False`, shape N/A.

Returns

- `comm_logits`: `tf.Tensor, dtype float32, shape (B, m)`.
- `meas_list`: `list[tf.Tensor]`, length 1; element 0 is measurement logits `tf.Tensor, dtype float32, shape (B, k)` where `k` is measurement size produced by the measurement layer (not specified by this module).
- `out_list`: `list[tf.Tensor]`, length 1; element 0 is SR outcome logits `tf.Tensor, dtype float32, shape (B, k)` (same trailing dimension as measurement logits at runtime in replay mode).

Errors

- `ValueError` if `field_logits` is not rank-2.
- `ValueError` if `field_logits` has statically-known trailing dimension and it does not equal `self.n2`.
- `TypeError` if `replay_out_a_logits_list` is provided and is not a `list` or `tuple`.
- `ValueError` if `replay_out_a_logits_list` is provided and `len(...) != 1`.

!!! note "Training kwarg compatibility"
    The measurement and combine layers are called with `training=training` in a `try` block and retried without `training` on `TypeError`, to support layers that do not accept the `training` keyword argument.

### save_weights_to

`save_weights_to(path: str) -> None`

- Purpose: Ensure variables exist (via `_ensure_built()`) then save Keras model weights.

Arguments

- `path`: `str`, unconstrained, shape N/A.

Returns

- `None`, shape N/A.

### load_weights_from

`load_weights_from(path: str) -> None`

- Purpose: Ensure variables exist (via `_ensure_built()`) then load Keras model weights.

Arguments

- `path`: `str`, unconstrained, shape N/A.

Returns

- `None`, shape N/A.

## Data & State

- `n2`: `int`, constraint not specified; number of field bits inferred from `game_layout`, shape N/A.
- `M`: `int`, constraint $m \ge 1$; number of communication bits inferred from `game_layout`, shape N/A.
- `depth`: `int`, constant value `1`, shape N/A.
- `measure_layers`: `list[tf.keras.layers.Layer]`, length 1, shape N/A.
- `combine_layers`: `list[tf.keras.layers.Layer]`, length 1, shape N/A.
- `sr_layers`: `list[PRAssistedReplay]`, length 1, shape N/A.
- `measure_layer`: `tf.keras.layers.Layer`, alias of `measure_layers[0]`, shape N/A.
- `combine_layer`: `tf.keras.layers.Layer`, alias of `combine_layers[0]`, shape N/A.
- `sr_layer`: `PRAssistedReplay`, alias of `sr_layers[0]`, shape N/A.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- The class fixes `depth = 1` and enforces that any provided `measure_layers` or `combine_layers` sequences match this length.
- `_ensure_built()` forces variable creation by running `compute_with_internal(...)` with a dummy replay outcome list whose element has shape `(1, n2)`; if the measurement layer output size differs from `n2`, the replay outcome shape may mismatch and trigger the runtime assertion in SR replay mode.

## Related

- `Q_Sea_Battle.lin_measurement_layer_a.LinMeasurementLayerA`
- `Q_Sea_Battle.lin_combine_layer_a.LinCombineLayerA`
- `Q_Sea_Battle.pr_assisted_replay.PRAssistedReplay`
- `Q_Sea_Battle.pyr_internal_model_a._infer_n2_and_m`

## Changelog

- Not specified.