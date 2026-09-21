# PyrInternalModelA

> Role: Logits-only Pyramid internal model (A) that maps field logits $(B, n2)$ to communication logits $(B, 1)$ while optionally exposing per-level intermediate logits for training/teacher forcing.

Location: `Q_Sea_Battle.pyr_internal_model_a.PyrInternalModelA`

## Derived constraints

- Let $n2$ be the flattened field length inferred from `game_layout`; the Pyramid depth is $depth = \log_2(n2)$ and therefore $n2$ MUST be a positive power of two.
- Let $m$ be the communication size inferred from `game_layout` (`comms_size` or legacy `M`); this architecture requires $m = 1$.

## Constructor

Parameter | Type | Description
--- | --- | ---
game_layout | Any, must expose attributes `n2` or `field_size`, and `comms_size` or `M` | Layout-like object used to infer $n2$ and $m$ (stored as `self.n2` and `self.M`).
sr_mode | str, expected in {"replay","stochastic"} | Shared resource mode forwarded to each `PRAssistedReplay` layer.
p_rule | float, not specified | Follow probability forwarded to each `PRAssistedReplay` layer (used in stochastic mode inside `PRAssistedReplay`).
beta | float, not specified | Logit magnitude forwarded to each `PRAssistedReplay` layer.
alpha | float, not specified | PR gate sharpness forwarded to each `PRAssistedReplay` layer.
seed | int \| None, not specified | Optional seed forwarded to each `PRAssistedReplay` layer.
measure_layers | Sequence[tf.keras.layers.Layer] \| None, if provided length must equal `depth` | Optional explicit per-level measurement layers; if `None`, creates `depth` instances of `PyrMeasurementLayerA`.
combine_layers | Sequence[tf.keras.layers.Layer] \| None, if provided length must equal `depth` | Optional explicit per-level combine layers; if `None`, creates `depth` instances of `PyrCombineLayerA`.
name | str \| None, not specified | Optional Keras model name forwarded to `tf.keras.Model`.

Preconditions

- `game_layout` MUST provide enough attributes to infer $n2$ and $m$ (either `n2` or `field_size`, and either `comms_size` or `M`).
- The inferred $m$ MUST equal 1.
- The inferred $n2$ MUST be a positive power of two.
- If `measure_layers` is provided, `len(measure_layers) == depth`.
- If `combine_layers` is provided, `len(combine_layers) == depth`.

Postconditions

- `self.n2: int` is set to the inferred $n2$.
- `self.M: int` is set to the inferred $m$ (and equals 1).
- `self.depth: int` is set to $\log_2(n2)$.
- `self.measure_layers: list[tf.keras.layers.Layer]` has length `depth`.
- `self.combine_layers: list[tf.keras.layers.Layer]` has length `depth`.
- `self.sr_layers: list[PRAssistedReplay]` has length `depth`.
- Backward-compat aliases are set: `self.measure_layer == self.measure_layers[0]` and `self.combine_layer == self.combine_layers[0]`.

Errors

- Raises `ValueError` if inferred `comms_size != 1`.
- Raises `ValueError` if `measure_layers` is provided and its length does not match `depth`.
- Raises `ValueError` if `combine_layers` is provided and its length does not match `depth`.
- Raises `ValueError` if inferred $n2$ is non-positive or not a power of two (via internal validation).

Example

!!! example "Instantiate with inferred layers"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.pyr_internal_model_a import PyrInternalModelA

    class Layout:
        field_size = 4  # n2 = 16, depth = 4
        comms_size = 1

    model = PyrInternalModelA(Layout(), sr_mode="replay", alpha=5.0, beta=10.0, seed=123)

    x = tf.zeros((2, model.n2), dtype=tf.float32)
    y = model(x, training=False)
    assert y.shape == (2, 1)
    ```

## Public Methods

### set_alpha

Set PR gate sharpness for all SR layers.

Signature

- `set_alpha(alpha: float) -> None`

Arguments

- `alpha`: float, not specified, scalar.

Returns

- `None`: NoneType, no value.

Errors

- Raises `AttributeError` if a configured SR layer does not implement `set_alpha`.

### set_p_rule

Set stochastic follow probability for all SR layers.

Signature

- `set_p_rule(p_rule: float) -> None`

Arguments

- `p_rule`: float, not specified, scalar.

Returns

- `None`: NoneType, no value.

Errors

- Raises `AttributeError` if a configured SR layer does not implement `set_p_rule`.

### set_beta

Set hard-logit beta for all SR layers.

Signature

- `set_beta(beta: float) -> None`

Arguments

- `beta`: float, not specified, scalar.

Returns

- `None`: NoneType, no value.

Errors

- Raises `AttributeError` if a configured SR layer does not implement `set_beta`.

### set_sr_mode

Set SR mode for all SR layers.

Signature

- `set_sr_mode(sr_mode: str) -> None`

Arguments

- `sr_mode`: str, not specified, scalar; typically one of `"replay"` or `"stochastic"`.

Returns

- `None`: NoneType, no value.

Errors

- Raises `AttributeError` if a configured SR layer does not implement `set_sr_mode`.

### call

Keras forward pass returning only the final communication logits.

Signature

- `call(field_scaled: tf.Tensor, training: bool = False, **kwargs: Any) -> tf.Tensor`

Arguments

- `field_scaled`: tf.Tensor, dtype float32 (converted), shape $(B, n2)$; treated as field logits (name retained for API compatibility).
- `training`: bool, not specified, scalar; forwarded to sublayers where supported.
- `**kwargs`: Any, unused; accepted for Keras compatibility.

Returns

- `comm_logits`: tf.Tensor, dtype float32, shape $(B, 1)$.

Errors

- Not specified in `call`; input validation is performed inside `compute_with_internal` which `call` invokes.

### compute_with_internal

Compute a full forward pass and return per-level intermediate logits.

Signature

- `compute_with_internal(field_logits: tf.Tensor, replay_out_a_logits_list: Sequence[tf.Tensor] | None = None, harden_between_levels: bool = False, beta_for_hardening: float = 10.0, training: bool = False) -> tuple[tf.Tensor, list[tf.Tensor], list[tf.Tensor]]`

Arguments

- `field_logits`: tf.Tensor, dtype float32 (converted), shape $(B, n2)$.
- `replay_out_a_logits_list`: Sequence[tf.Tensor] \| None, if provided must be a Python `list` or `tuple` of length `depth`; each element is converted to tf.Tensor dtype float32 and MUST satisfy `tf.shape(elem)[-1] == tf.shape(meas_logits)[-1]` at its level (enforced at runtime).
- `harden_between_levels`: bool, not specified, scalar; if `True`, hardens intermediate state logits between levels to $\pm beta\_for\_hardening$ based on sign.
- `beta_for_hardening`: float, not specified, scalar; magnitude used when hardening is enabled.
- `training`: bool, not specified, scalar; forwarded to sublayers where supported.

Returns

- `comm_logits`: tf.Tensor, dtype float32, shape $(B, 1)$; the final field logits interpreted as comm logits for Pyramid layouts.
- `meas_list`: list[tf.Tensor], dtype float32 elements, length `depth`; per-level measurement logits; element shapes are not fully specified in the module docstring (depend on `PyrMeasurementLayerA`), but are used with `tf.zeros_like` and compared in last-dimension width to any teacher-forced replay logits.
- `out_list`: list[tf.Tensor], dtype float32 elements, length `depth`; per-level SR outcome logits returned by `PRAssistedReplay`; element shapes are not fully specified (depend on `PRAssistedReplay` and measurement width at each level).

Errors

- Raises `ValueError` if `field_logits` is not rank-2.
- Raises `ValueError` if `field_logits` last dimension is statically known and not equal to `n2`.
- Raises `TypeError` if `replay_out_a_logits_list` is provided but is not a Python `list` or `tuple`.
- Raises `ValueError` if `replay_out_a_logits_list` is provided and its length is not `depth`.
- May raise `tf.errors.InvalidArgumentError` (or similar TensorFlow runtime error) if teacher forcing replay logits do not match measurement width at a level (enforced via `tf.debugging.assert_equal`).
- Raises `RuntimeError` if internal depth iteration produces no outputs (should be unreachable unless `depth == 0`).

!!! note "Teacher forcing behavior"
    If `replay_out_a_logits_list` is provided, element `[level]` is passed to the SR layer as `replay_outcome_logits`, enabling deterministic per-level outcomes while still producing measurement and combined logits.

### save_weights_to

Save model weights to a file; ensures variables are built first.

Signature

- `save_weights_to(path: str) -> None`

Arguments

- `path`: str, not specified, scalar; destination filepath understood by `tf.keras.Model.save_weights`.

Returns

- `None`: NoneType, no value.

Errors

- Not specified; may raise exceptions from `_ensure_built` or TensorFlow/Keras IO.

### load_weights_from

Load model weights from a file; ensures variables are built first.

Signature

- `load_weights_from(path: str) -> None`

Arguments

- `path`: str, not specified, scalar; source filepath understood by `tf.keras.Model.load_weights`.

Returns

- `None`: NoneType, no value.

Errors

- Not specified; may raise exceptions from `_ensure_built` or TensorFlow/Keras IO.

## Data & State

- `n2`: int, constraint: positive power of two; scalar; flattened field length.
- `M`: int, constraint: equals 1; scalar; communication size (stored under legacy name `M`).
- `depth`: int, constraint: `2**depth == n2`; scalar; number of Pyramid levels.
- `measure_layers`: list[tf.keras.layers.Layer], constraint: length `depth`; per-level measurement layers.
- `combine_layers`: list[tf.keras.layers.Layer], constraint: length `depth`; per-level combine layers.
- `sr_layers`: list[PRAssistedReplay], constraint: length `depth`; per-level PR-assisted shared resource layers.
- `measure_layer`: tf.keras.layers.Layer, alias to `measure_layers[0]`.
- `combine_layer`: tf.keras.layers.Layer, alias to `combine_layers[0]`.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- Input naming: `call` uses parameter name `field_scaled` for API compatibility, but the tensor is treated as logits; adapters for scaling/bit conversion are external to this model.
- Variable creation: weights may not exist until the first forward pass; `save_weights_to` and `load_weights_from` call an internal build helper that runs a minimal `compute_with_internal` using dummy tensors, including a dummy teacher-forcing list of length `depth` with shapes `(1, n2 // (2 ** (d + 1)))`.

## Related

- `Q_Sea_Battle.pyr_measurement_layer_a.PyrMeasurementLayerA`
- `Q_Sea_Battle.pyr_combine_layer_a.PyrCombineLayerA`
- `Q_Sea_Battle.pr_assisted_replay.PRAssistedReplay`

## Changelog

- Not specified.