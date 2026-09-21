# LinInternalModelB

> Role: Depth-1 trainable linear internal model that consumes logits and produces a shoot logit plus Pyramid-compatible internal trace lists.

Location: `Q_Sea_Battle.lin_internal_model_b.LinInternalModelB`

## Derived constraints

- Depth is fixed: `depth = 1`.
- Define symbols used throughout: `n2` = number of gun bits (flattened board size), `m` = communication channel width in logits/bits, `B` = batch size.
- Input/trace list lengths must equal `depth` (therefore exactly 1).
- `m >= 1` is required at construction time.

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | Any, constraints: must be accepted by `_infer_n2_and_m(game_layout)`, shape: N/A | Game layout object used to infer `n2` and `m`. |
| sr_mode | str, constraints: not validated here, shape: scalar | SR mode forwarded to `PRAssistedReplay` (project uses `"replay"` and `"stochastic"`). |
| p_rule | float, constraints: not validated here, shape: scalar | Probability weight for the rule-based branch inside the PR-assisted SR. |
| beta | float, constraints: not validated here, shape: scalar | Logit scale used by the PR-assisted SR. |
| alpha | float, constraints: not validated here, shape: scalar | Mixing/strength parameter used by the PR-assisted SR. |
| seed | int \| None, constraints: any int or None, shape: scalar | Optional seed forwarded to the PR-assisted SR. |
| measure_layers | Sequence[tf.keras.layers.Layer] \| None, constraints: if not None, `len(measure_layers) == depth == 1`, shape: sequence length 1 | Optional custom measurement layers; if None, a default `LinMeasurementLayerB` is created. |
| combine_layers | Sequence[tf.keras.layers.Layer] \| None, constraints: if not None, `len(combine_layers) == depth == 1`, shape: sequence length 1 | Optional custom combine layers; if None, a default `LinCombineLayerB` is created. |
| hidden_units_measure | int, constraints: not validated here, shape: scalar | Hidden size for the default measurement layer. |
| hidden_units_combine | int, constraints: not validated here, shape: scalar | Hidden size for the default combine layer. |
| name | str \| None, constraints: any string or None, shape: scalar | Optional Keras model name. |

Preconditions

- `_infer_n2_and_m(game_layout)` must return `(n2, m)` where `m >= 1`.
- If `measure_layers` is not None, it must be a sequence of length `1`.
- If `combine_layers` is not None, it must be a sequence of length `1`.

Postconditions

- `self.n2: int` and `self.M: int` are set from `_infer_n2_and_m(game_layout)`.
- `self.depth: int` is set to `1`.
- `self.measure_layers: list[tf.keras.layers.Layer]` is length `1`; `self.measure_layer` aliases element `0`.
- `self.combine_layers: list[tf.keras.layers.Layer]` is length `1`; `self.combine_layer` aliases element `0`.
- `self.sr_layers: list[PRAssistedReplay]` is length `1`; `self.sr_layer` aliases element `0`.
- `self.last_flip_logit: tf.Tensor \| None` is initialized to `None`.

Errors

- `ValueError`: if `m < 1`.
- `ValueError`: if `measure_layers` is provided and `len(measure_layers) != 1`.
- `ValueError`: if `combine_layers` is provided and `len(combine_layers) != 1`.

Example

!!! example "Construct and run a forward pass"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.lin_internal_model_b import LinInternalModelB

    game_layout = object()  # must be compatible with _infer_n2_and_m(...)
    model = LinInternalModelB(game_layout)

    B = 2
    gun = tf.zeros((B, model.n2), dtype=tf.float32)
    comm = tf.zeros((B, model.M), dtype=tf.float32)
    prev_meas_list = [tf.zeros((B, model.n2), dtype=tf.float32)]
    prev_out_list = [tf.zeros((B, model.n2), dtype=tf.float32)]

    shoot = model([gun, comm, prev_meas_list, prev_out_list], training=False)
    ```

## Public Methods

### set_alpha(alpha)

Update SR `alpha` for all SR layers.

Arguments

- alpha: float, constraints: not validated here, shape: scalar.

Returns

- None: NoneType, constraints: always None, shape: scalar.

Errors

- AttributeError: if an SR layer does not implement `set_alpha`.

### set_p_rule(p_rule)

Update SR `p_rule` for all SR layers.

Arguments

- p_rule: float, constraints: not validated here, shape: scalar.

Returns

- None: NoneType, constraints: always None, shape: scalar.

Errors

- AttributeError: if an SR layer does not implement `set_p_rule`.

### set_beta(beta)

Update SR `beta` for all SR layers.

Arguments

- beta: float, constraints: not validated here, shape: scalar.

Returns

- None: NoneType, constraints: always None, shape: scalar.

Errors

- AttributeError: if an SR layer does not implement `set_beta`.

### set_sr_mode(sr_mode)

Switch SR mode (e.g., `"replay"` vs `"stochastic"`) for all SR layers.

Arguments

- sr_mode: str, constraints: not validated here, shape: scalar.

Returns

- None: NoneType, constraints: always None, shape: scalar.

Errors

- AttributeError: if an SR layer does not implement `set_sr_mode`.

### call(inputs, training=False, **kwargs)

Keras forward pass that returns only the shoot logit.

Arguments

- inputs: Any, constraints: must be list/tuple in one of the accepted formats described below, shape: N/A.
- training: bool, constraints: any boolean, shape: scalar.
- **kwargs: dict[str, Any], constraints: unused, shape: mapping.

Accepted input formats

- Nested: `[gun, comm, prev_meas_list, prev_out_list]` where `prev_meas_list` and `prev_out_list` are list/tuple each of length `1`.
- Flattened: `[gun, comm, *prev_meas, *prev_out]` where the number of previous tensors equals `depth` (therefore total length `2 + 2 * depth = 4`).

Returns

- shoot_logit: tf.Tensor, dtype float32, constraints: rank-2, shape `(B, 1)`.

Errors

- TypeError: if `inputs` is not a list/tuple.
- ValueError: if `inputs` is flattened but its length is not `2 + 2 * depth`.

!!! note "Delegation"
    This method calls `compute_with_internal(...)` and returns only the first element of its result tuple.

### compute_with_internal(gun_logits, comm_in_logits, prev_meas_list, prev_out_list, harden_between_levels=False, beta_for_hardening=10.0, training=False)

Compute the shoot logit and return Pyramid-compatible internal trace lists (depth-1 traces).

Arguments

- gun_logits: tf.Tensor, dtype float32 (converted via `tf.convert_to_tensor(..., dtype=tf.float32)`), constraints: rank-2, last dimension must be `n2` if statically known, shape `(B, n2)`.
- comm_in_logits: tf.Tensor, dtype float32 (converted via `tf.convert_to_tensor(..., dtype=tf.float32)`), constraints: rank-2, last dimension must be `m` if statically known, shape `(B, m)`.
- prev_meas_list: Sequence[tf.Tensor], constraints: must be list/tuple, length `1`, element convertible to float32 tensor, shape: sequence length 1 containing tensors of shape `(B, w)` where `w` is asserted equal (dynamically) to the measurement width.
- prev_out_list: Sequence[tf.Tensor], constraints: must be list/tuple, length `1`, element convertible to float32 tensor, shape: sequence length 1 containing tensors of shape `(B, w)` where `w` is asserted equal (dynamically) to the measurement width.
- harden_between_levels: bool, constraints: any boolean, shape: scalar; when True, logits are mapped to fixed magnitude by sign between stages.
- beta_for_hardening: float, constraints: used as the fixed magnitude when hardening, shape: scalar.
- training: bool, constraints: any boolean, shape: scalar.

Returns

- result: tuple, constraints: 5-tuple, shape: N/A, contents: `(shoot_logit, meas_b_logits_list, out_b_logits_list, comms_logits_list, gun_logits_list)` where:
  - shoot_logit: tf.Tensor, dtype float32, constraints: rank-2, shape `(B, 1)`.
  - meas_b_logits_list: list[tf.Tensor], constraints: length 1, shape: list length 1 containing a tensor of shape `(B, w)` where `w` is the measurement width produced by `measure_layer`.
  - out_b_logits_list: list[tf.Tensor], constraints: length 1, shape: list length 1 containing a tensor of shape `(B, w)`.
  - comms_logits_list: list[tf.Tensor], constraints: length 2, shape: `[c_logits, shoot_logit]` where `c_logits` is `comm_in_logits` (optionally hardened), shapes `[(B, m), (B, 1)]`.
  - gun_logits_list: list[tf.Tensor], constraints: length 2, shape: `[gun_logits, gun_logits]`, both tensors shape `(B, n2)`.

Errors

- ValueError: if `gun_logits` is not rank-2.
- ValueError: if `gun_logits.shape[-1]` is statically known and not equal to `n2`.
- ValueError: if `comm_in_logits` is not rank-2.
- ValueError: if `comm_in_logits.shape[-1]` is statically known and not equal to `m`.
- TypeError: if `prev_meas_list` or `prev_out_list` is not a list/tuple.
- ValueError: if `len(prev_meas_list) != depth` or `len(prev_out_list) != depth` (depth is 1).
- tf.errors.InvalidArgumentError: may be raised by `tf.debugging.assert_equal` if dynamic trace widths do not match the measurement width.

Side effects

- Updates `self.last_flip_logit: tf.Tensor` to the most recent `flip_logit` returned by the combine layer (dtype float32), or leaves it unchanged if the method fails before assignment.

!!! note "Hardening behavior"
    When `harden_between_levels` is True, `gun_logits`, `comm_in_logits`, and `out_b_logits` are mapped elementwise via `tf.where(x >= 0.0, beta_for_hardening, -beta_for_hardening)`; this is non-differentiable at 0 and acts like a straight-through discretization boundary.

### save_weights_to(path)

Save Keras weights after ensuring variables are built.

Arguments

- path: str, constraints: passed through to `tf.keras.Model.save_weights`, shape: scalar.

Returns

- None: NoneType, constraints: always None, shape: scalar.

Errors

- Not specified: errors propagated from `_ensure_built()` and `tf.keras.Model.save_weights`.

### load_weights_from(path)

Load Keras weights after ensuring variables are built.

Arguments

- path: str, constraints: passed through to `tf.keras.Model.load_weights`, shape: scalar.

Returns

- None: NoneType, constraints: always None, shape: scalar.

Errors

- Not specified: errors propagated from `_ensure_built()` and `tf.keras.Model.load_weights`.

## Data & State

- n2: int, constraints: inferred from `game_layout`, shape: scalar; number of gun bits (flattened board size).
- M: int, constraints: inferred from `game_layout`, must satisfy `M >= 1`, shape: scalar; communication channel width `m`.
- depth: int, constraints: always `1`, shape: scalar.
- measure_layers: list[tf.keras.layers.Layer], constraints: length 1, shape: list length 1; measurement layer stack (depth-indexed).
- combine_layers: list[tf.keras.layers.Layer], constraints: length 1, shape: list length 1; combine layer stack (depth-indexed).
- sr_layers: list[PRAssistedReplay], constraints: length 1, shape: list length 1; PR-assisted SR stack (depth-indexed).
- measure_layer: tf.keras.layers.Layer, constraints: alias of `measure_layers[0]`, shape: N/A.
- combine_layer: tf.keras.layers.Layer, constraints: alias of `combine_layers[0]`, shape: N/A.
- sr_layer: PRAssistedReplay, constraints: alias of `sr_layers[0]`, shape: N/A.
- last_flip_logit: tf.Tensor | None, dtype float32 when not None, constraints: set by `compute_with_internal`, shape unknown from this module (depends on `LinCombineLayerB` output); initial value is None.

## Planned (design-spec)

- Not specified.

## Deviations

- The module docstring contract refers to `comms_size` and uses the symbol `m`; the implementation stores this as `self.M` and enforces `self.M >= 1` with an error message referencing `comms_size>=1`.

## Notes for Contributors

- `call()` supports both nested and flattened input formats; keep `depth` handling consistent if extending beyond depth-1 in the future.
- `_ensure_built()` forces variable creation by calling `compute_with_internal(...)` with dummy tensors; changes to input signatures or required trace widths should be reflected there.
- `compute_with_internal()` performs both static shape checks (rank and known last dimension) and dynamic checks via `tf.debugging.assert_equal`; if measurement width changes away from `n2`, update dummy trace construction in `_ensure_built()` accordingly.

## Related

- `Q_Sea_Battle.lin_measurement_layer_b.LinMeasurementLayerB`
- `Q_Sea_Battle.lin_combine_layer_b.LinCombineLayerB`
- `Q_Sea_Battle.pr_assisted_replay.PRAssistedReplay`
- `Q_Sea_Battle.pyr_internal_model_a._infer_n2_and_m`

## Changelog

- Not specified.