# PyrInternalModelB

> Role: Pure-logit Pyramid internal model for Player B that consumes logit tensors and emits a final communication logit, with optional per-level internal signal tracing.

Location: `Q_Sea_Battle.pyr_internal_model_b.PyrInternalModelB`

## Derived constraints

- Define $n2$ as the gun/state vector width (number of gun/state bits), and define $m$ as the communication size (`self.M` in code). This class enforces $m = 1$ at construction time.
- Define `depth` as the number of pyramid levels computed from $n2$ via `_validate_power_of_two(n2)`; therefore $n2$ must be a power of two.
- For level index $d \in \{0, \dots, \text{depth}-1\}$, per-level widths are $k_d = \frac{n2}{2^{d+1}}$; therefore `prev_meas_list[d]` and `prev_out_list[d]` must have trailing dimension $k_d$ (enforced dynamically via `tf.debugging.assert_equal` against the computed measurement output width at that level).

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| `game_layout` | `Any`, constraints: interpreted by `_infer_n2_and_m(game_layout)` | Game configuration object used to infer $n2$ and $m$. |
| `sr_mode` | `str`, constraints: documented as one of `{"replay","stochastic"}` | SR mode forwarded to each `PRAssistedReplay` layer. |
| `p_rule` | `float`, constraints: not specified | Probability parameter forwarded to SR layers (interpretation depends on SR mode). |
| `beta` | `float`, constraints: not specified | Hard-logit inverse-temperature forwarded to SR layers. |
| `alpha` | `float`, constraints: not specified | Gate sharpness forwarded to SR layers. |
| `seed` | `int \| None`, constraints: any integer or `None` | Optional random seed forwarded to SR layers. |
| `measure_layers` | `Optional[Sequence[tf.keras.layers.Layer]]`, constraints: if provided, `len(measure_layers) == depth` | Optional pre-constructed per-level measurement layers; otherwise constructed as `PyrMeasurementLayerB()` per level. |
| `combine_layers` | `Optional[Sequence[tf.keras.layers.Layer]]`, constraints: if provided, `len(combine_layers) == depth` | Optional pre-constructed per-level combine layers; otherwise constructed as `PyrCombineLayerB()` per level. |
| `name` | `Optional[str]`, constraints: any string or `None` | Optional Keras model name passed to `tf.keras.Model`. |

Preconditions

- `_infer_n2_and_m(game_layout)` must succeed (behavior not specified in this module).
- $m = 1$ (enforced; otherwise `ValueError`).
- $n2$ must be a power of two (enforced by `_validate_power_of_two(n2)`; exact error behavior not specified in this module).
- If `measure_layers` is provided, it must be a Python sequence with length exactly `depth`.
- If `combine_layers` is provided, it must be a Python sequence with length exactly `depth`.

Postconditions

- `self.n2: int` is set from `_infer_n2_and_m(game_layout)`.
- `self.M: int` is set from `_infer_n2_and_m(game_layout)` and is guaranteed to equal `1`.
- `self.depth: int` is set from `_validate_power_of_two(self.n2)`.
- `self.measure_layers: list[tf.keras.layers.Layer]` has length `depth`.
- `self.combine_layers: list[tf.keras.layers.Layer]` has length `depth`.
- `self.sr_layers: list[PRAssistedReplay]` has length `depth` and is populated with one `PRAssistedReplay(...)` per level using the provided SR hyperparameters.

Errors

- `ValueError`: if `self.M != 1` (message indicates `comms_size==1` requirement).
- `ValueError`: if `measure_layers` is provided and `len(measure_layers) != depth`.
- `ValueError`: if `combine_layers` is provided and `len(combine_layers) != depth`.

Example

```python
import tensorflow as tf
from Q_Sea_Battle.pyr_internal_model_b import PyrInternalModelB

game_layout = ...  # must be compatible with _infer_n2_and_m and yield m=1
model = PyrInternalModelB(game_layout, sr_mode="replay", p_rule=1.0, beta=10.0, alpha=5.0, seed=123)

B = 2
gun_logits = tf.zeros((B, model.n2), dtype=tf.float32)
comm_in_logits = tf.zeros((B, 1), dtype=tf.float32)

prev_meas_list = [tf.zeros((B, model.n2 // (2 ** (d + 1))), dtype=tf.float32) for d in range(model.depth)]
prev_out_list = [tf.zeros((B, model.n2 // (2 ** (d + 1))), dtype=tf.float32) for d in range(model.depth)]

shoot_logit, meas_list, out_list, comms_list, guns_list = model.compute_with_internal(
    gun_logits,
    comm_in_logits,
    prev_meas_list,
    prev_out_list,
    training=False,
)
```

## Public Methods

### `set_alpha(alpha)`

Set PR gate sharpness for all PR-assisted SR layers.

Parameters

- `alpha`: `float`, constraints: not specified, shape: scalar.

Returns

- `None`, constraints: always `None`.

Errors

- `AttributeError`: if any SR layer does not implement `set_alpha`.

### `set_p_rule(p_rule)`

Set `p_rule` for all PR-assisted SR layers.

Parameters

- `p_rule`: `float`, constraints: not specified, shape: scalar.

Returns

- `None`, constraints: always `None`.

Errors

- `AttributeError`: if any SR layer does not implement `set_p_rule`.

### `set_beta(beta)`

Set hard-logit beta for all PR-assisted SR layers.

Parameters

- `beta`: `float`, constraints: not specified, shape: scalar.

Returns

- `None`, constraints: always `None`.

Errors

- `AttributeError`: if any SR layer does not implement `set_beta`.

### `set_sr_mode(sr_mode)`

Set SR mode for all PR-assisted SR layers.

Parameters

- `sr_mode`: `str`, constraints: not specified by this method (constructor documents `{"replay","stochastic"}`), shape: scalar string.

Returns

- `None`, constraints: always `None`.

Errors

- `AttributeError`: if any SR layer does not implement `set_sr_mode`.

### `call(inputs, training=False, **kwargs)`

Keras/Player-facing forward call delegating to `compute_with_internal()`.

Parameters

- `inputs`: `Any`, constraints: must be `list` or `tuple`; accepted shapes: one of the two packings described below.
- `training`: `bool`, constraints: any boolean; shape: scalar.
- `kwargs`: `dict[str, Any]`, constraints: unused; accepted for Keras compatibility.

Accepted input packings

- Flat packing: `inputs = [gun_logits, comm_in_logits, *prev_meas_list, *prev_out_list]` with total length `2 + 2*depth`.
- Nested packing: `inputs = [gun_logits, comm_in_logits, prev_meas_list, prev_out_list]` where `prev_meas_list` and `prev_out_list` are `list`/`tuple`.

Required tensor contracts (enforced in `compute_with_internal`)

- `gun_logits`: `tf.Tensor`, dtype `float32`, shape `(B, n2)`.
- `comm_in_logits`: `tf.Tensor`, dtype `float32`, shape `(B, 1)`.
- `prev_meas_list`: `Sequence[tf.Tensor]`, constraints: length `depth`, each dtype `float32`, shape `(B, k_d)` where $k_d = \frac{n2}{2^{d+1}}$.
- `prev_out_list`: `Sequence[tf.Tensor]`, constraints: length `depth`, each dtype `float32`, shape `(B, k_d)` where $k_d = \frac{n2}{2^{d+1}}$.

Returns

- `shoot_logit`: `tf.Tensor`, dtype `float32`, shape `(B, 1)`.

Errors

- `TypeError`: if `inputs` is not a `list`/`tuple`.
- `ValueError`: if using flat packing and `len(inputs) != 2 + 2*depth`.
- Any errors raised by `compute_with_internal(...)` for shape/type validation.

### `compute_with_internal(gun_logits, comm_in_logits, prev_meas_list, prev_out_list, harden_between_levels=False, beta_for_hardening=10.0, *, training=False)`

Run the pure-logit internal forward pass and return per-level signals.

Parameters

- `gun_logits`: `tf.Tensor`, dtype `float32` (converted via `tf.convert_to_tensor(..., dtype=tf.float32)`), shape `(B, n2)`, constraints: rank must be 2; last dimension must equal `n2` if statically known.
- `comm_in_logits`: `tf.Tensor`, dtype `float32` (converted), shape `(B, 1)`, constraints: rank must be 2; last dimension must equal `1` if statically known.
- `prev_meas_list`: `Sequence[tf.Tensor]`, constraints: must be a Python `list` or `tuple` of length `depth`; each element convertible to `tf.Tensor` and cast to `float32`; shape `(B, k_d)` where $k_d = \frac{n2}{2^{d+1}}$ (enforced dynamically to match the per-level measurement output width).
- `prev_out_list`: `Sequence[tf.Tensor]`, constraints: must be a Python `list` or `tuple` of length `depth`; each element convertible to `tf.Tensor` and cast to `float32`; shape `(B, k_d)$ where $k_d = \frac{n2}{2^{d+1}}$ (enforced dynamically to match the per-level measurement output width).
- `harden_between_levels`: `bool`, constraints: any boolean; shape: scalar; if `True`, intermediate `state_logits`, `c_logit`, and `out_b_logits` are mapped by sign to $\pm \text{beta\_for\_hardening}$ between levels.
- `beta_for_hardening`: `float`, constraints: not specified; shape: scalar; magnitude used when hardening logits.
- `training`: `bool`, constraints: any boolean; shape: scalar; forwarded to sublayers.

Returns

- `shoot_logit`: `tf.Tensor`, dtype `float32`, shape `(B, 1)`, constraints: final communication logit.
- `meas_b_logits_list`: `list[tf.Tensor]`, dtype `float32`, shape: length `depth`, each element shape `(B, k_d)`.
- `out_b_logits_list`: `list[tf.Tensor]`, dtype `float32`, shape: length `depth`, each element shape `(B, k_d)`.
- `comms_logits_list`: `list[tf.Tensor]`, dtype `float32`, shape: length `depth + 1`, element 0 is the input `comm_in_logits` (cast/converted), subsequent elements are per-level next comm logits, each shape `(B, 1)`.
- `gun_logits_list`: `list[tf.Tensor]`, dtype `float32`, shape: length `depth + 1`, element 0 is the input `gun_logits` (cast/converted), subsequent elements are per-level next gun logits, each shape `(B, n2)` (exact internal transformation constraints not specified here; output is cast to `float32`).

Errors

- `ValueError`: if `gun_logits` is not rank 2.
- `ValueError`: if `gun_logits.shape[-1]` is statically known and not equal to `n2`.
- `ValueError`: if `comm_in_logits` is not rank 2, or if its last dimension is statically known and not equal to `1`.
- `TypeError`: if `prev_meas_list` or `prev_out_list` is not a Python `list`/`tuple`.
- `ValueError`: if `len(prev_meas_list) != depth` or `len(prev_out_list) != depth`.
- `tf.errors.InvalidArgumentError` (or other TensorFlow assertion errors): may be raised by `tf.debugging.assert_equal` if per-level trailing dimensions of `prev_meas`/`prev_out` do not match the measurement output trailing dimension.

!!! note "Return order"
    The method returns `(shoot_logit, meas_b_logits_list, out_b_logits_list, comms_logits_list, gun_logits_list)`. A note in code states a legacy implementation returned `gun_logits_list` and `comms_logits_list` in the opposite order.

### `save_weights_to(path)`

Save model weights to a file, ensuring the model is built first.

Parameters

- `path`: `str`, constraints: must be a path understood by `tf.keras.Model.save_weights`; shape: scalar string.

Returns

- `None`, constraints: always `None`.

Errors

- Not specified in this module; may raise errors from TensorFlow/Keras I/O and `save_weights`.

### `load_weights_from(path)`

Load model weights from a file, ensuring the model is built first.

Parameters

- `path`: `str`, constraints: must be a path understood by `tf.keras.Model.load_weights`; shape: scalar string.

Returns

- `None`, constraints: always `None`.

Errors

- Not specified in this module; may raise errors from TensorFlow/Keras I/O and `load_weights`.

## Data & State

- `n2`: `int`, constraints: inferred from `game_layout` by `_infer_n2_and_m`; represents gun/state width.
- `M`: `int`, constraints: must equal `1` (enforced); represents communication size $m$.
- `depth`: `int`, constraints: computed by `_validate_power_of_two(n2)`; number of pyramid levels.
- `measure_layers`: `list[tf.keras.layers.Layer]`, constraints: length `depth`; elements are either provided by `measure_layers` or created as `PyrMeasurementLayerB()`.
- `combine_layers`: `list[tf.keras.layers.Layer]`, constraints: length `depth`; elements are either provided by `combine_layers` or created as `PyrCombineLayerB()`.
- `sr_layers`: `list[PRAssistedReplay]`, constraints: length `depth`; one SR layer per level, created in the constructor.
- Keras model variables: created lazily; `_ensure_built()` runs a dummy forward pass to force variable creation when needed for weight I/O.

## Planned (design-spec)

- Not specified (no design notes provided beyond in-module docstring).

## Deviations

- The module-level docstring describes `compute_with_internal()` returning five values: `shoot_logit, meas_b_list, out_b_list, comms_list, guns_list`, but the implementation currently returns exactly five values in that same order; however, the annotated return type in the signature is `tuple[tf.Tensor, list[tf.Tensor], list[tf.Tensor], list[tf.Tensor], list[tf.Tensor], list[tf.Tensor]]` (six items), which does not match the actual return statement.
- The module-level docstring describes `call()` delegating to `compute_with_internal()` for player-facing compatibility; the implementation does so, but `call()` returns only `shoot_logit` and discards internal lists.

## Notes for Contributors

- Keep the `compute_with_internal()` return type annotation consistent with the actual return tuple length and ordering.
- `compute_with_internal()` includes an inner `harden_logits` helper defined inside the function; if performance is critical, consider implications of Python closure creation (behavioral changes must be validated).
- `_ensure_built()` assumes per-level tensors have trailing dimension `n2 // (2 ** (d + 1))`; changes to measurement/combine layer dimensionality contracts must update this dummy-shape logic accordingly.
- `call()` supports two input packings; any future changes to input structure must preserve backward compatibility or update adapters accordingly.

## Related

- `Q_Sea_Battle.pyr_measurement_layer_b.PyrMeasurementLayerB`
- `Q_Sea_Battle.pyr_combine_layer_b.PyrCombineLayerB`
- `Q_Sea_Battle.pr_assisted_replay.PRAssistedReplay`
- `Q_Sea_Battle.pyr_internal_model_a._infer_n2_and_m`
- `Q_Sea_Battle.pyr_internal_model_a._validate_power_of_two`

## Changelog

- Not specified in module.