# Class PyrTrainableAssistedModelB

**Module import path**: `Q_Sea_Battle.pyr_trainable_assisted_model_b.PyrTrainableAssistedModelB`

> Player-B pyramid assisted model (per-level layers) producing a shoot logit.

!!! note "Parent class"
    Inherits from `tf.keras.Model`.

!!! note "Terminology"
    This model uses `sr_mode` where **SR = shared resource**.
    The per-level shared resource is implemented by `PRAssistedLayer`.

!!! note "Derived symbols"
    Let `field_size = n`, `n2 = n**2`, and `comms_size = m`.
    Pyramid depth is `K = log2(n2)` and requires `n2` to be a power of two.
    Pyramid architecture requires `m = 1`.

## Overview

`PyrTrainableAssistedModelB` is the Player-B model for the pyramid (Pyr) assisted architecture.

It consumes Player A’s per-level internal tensors from `PyrTrainableAssistedModelA.compute_with_internal(...)` and
produces a shoot logit of shape `(B, 1)`.

## Inputs and dependencies

At each pyramid level, Player B uses:

- `PyrMeasurementLayerB` to compute a measurement vector from the current gun state.
- `PRAssistedLayer` as the shared resource to generate PR-assisted outcomes correlated with Player A.
- `PyrCombineLayerB` to update both the gun state and comm bit.

## Architecture pipeline

At each level with active length `L`:

- `state (B, L)` → `measure_layer` → `meas_b (B, L/2)`
- `meas_b (B, L/2)` + Player-A per-level tensors → `PRAssistedLayer(length=L/2, mode=sr_mode)` → `out_b (B, L/2)`
- `(state (B, L), out_b (B, L/2), comm (B, 1))` → `combine_layer` → `(next_state (B, L/2), next_comm (B, 1))`

After `K` levels, `comm` is converted into a **hard shoot logit**.

## Constructor

### Signature

- `PyrTrainableAssistedModelB(`
  `game_layout: Any,`
  `p_high: float = 0.9,`
  `sr_mode: str = "sample",`
  `measure_layers: Sequence[tf.keras.layers.Layer] | None = None,`
  `combine_layers: Sequence[tf.keras.layers.Layer] | None = None,`
  `name: str | None = None`
  `) -> PyrTrainableAssistedModelB`

### Arguments

- `game_layout`: `Any`, scalar.
  - GameLayout-like object providing:
    - `field_size: int`, scalar, or `n2: int`, scalar.
    - `comms_size: int`, scalar.
- `p_high`: `float`, scalar.
  - Correlation parameter forwarded to each `PRAssistedLayer`.
- `sr_mode`: `str`, scalar, one of `('sample', 'expected')`.
- `measure_layers`: `Sequence[tf.keras.layers.Layer]` or `None`, shape `(K,)`.
  - If provided, must have length `K = log2(n2)`.
  - Each layer consumes `(B, L)` and returns `(B, L/2)` for its level.
- `combine_layers`: `Sequence[tf.keras.layers.Layer]` or `None`, shape `(K,)`.
  - If provided, must have length `K = log2(n2)`.
  - Each layer consumes `(state, out, comm)` and returns `(next_state, next_comm)`.
- `name`: `str` or `None`, scalar.

### Returns

- `PyrTrainableAssistedModelB`, scalar.

### Preconditions

- `n2 > 0` and `n2` is a power of two.
- `comms_size == 1`.
- If provided, `len(measure_layers) == K`.
- If provided, `len(combine_layers) == K`.
- `sr_mode in ('sample', 'expected')`.

### Postconditions

- `self.n2`: `int`, scalar.
- `self.M`: `int`, scalar (comms_size).
- `self.depth`: `int`, scalar (`K = log2(n2)`).
- `self.measure_layers`: `list[tf.keras.layers.Layer]`, length `K`.
- `self.combine_layers`: `list[tf.keras.layers.Layer]`, length `K`.
- `self.sr_layers`: `list[PRAssistedLayer]`, length `K`.

### Errors

- Raises `ValueError` if `comms_size != 1`.
- Raises `ValueError` if `n2` is not a power of two.
- Raises `ValueError` if provided layer sequences do not have length `K`.

## Backward compatibility aliases

For legacy code, the following aliases point to the first level’s layers:

- `measure_layer` → `measure_layers[0]`
- `combine_layer` → `combine_layers[0]`

!!! note "Alias scope"
    The forward pass uses the per-level lists. Aliases are provided for older scripts only.

## Public Methods

### call

#### Signature

- `call(inputs: list) -> tf.Tensor`

#### Arguments

- `inputs`: `list`, length `4`, containing:
  - `gun_batch`: `tf.Tensor`, dtype `float32`, shape `(B, n2)`.
  - `comm_batch`: `tf.Tensor`, dtype `float32`, shape `(B, 1)`.
  - `prev_measurements`: `list[tf.Tensor]`, length `K`.
    - Each element: `tf.Tensor`, dtype `float32`, shape `(B, L/2)` for that level.
  - `prev_outcomes`: `list[tf.Tensor]`, length `K`.
    - Each element: `tf.Tensor`, dtype `float32`, shape `(B, L/2)` for that level.
    - Represents Player-A PR-assisted outcomes for that level.

#### Returns

- `shoot_logit`: `tf.Tensor`, dtype `float32`, shape `(B, 1)`.

#### Preconditions

- `inputs` is a list/tuple of length 4.
- `gun_batch` is rank-2 and last dim equals `n2`.
- `comm_batch` is rank-2 and has last dim 1.
- `prev_measurements` and `prev_outcomes` are Python lists/tuples of length `K`.

#### Postconditions

- Returns a hard logit derived from the final comm bit.

#### Errors

- Raises `ValueError` if `inputs` does not have length 4.
- Raises `ValueError` if `gun_batch` shape is not `(B, n2)`.
- Raises `ValueError` if `comm_batch` shape is not `(B, 1)`.
- Raises `TypeError` if `prev_measurements` or `prev_outcomes` is not a Python list/tuple.
- Raises `ValueError` if previous lists do not have length `K`.
- Raises `ValueError` if per-level tensor lengths are inconsistent at any level.

!!! example "Minimal usage"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.pyr_trainable_assisted_model_b import PyrTrainableAssistedModelB

    layout = GameLayout(field_size=4, comms_size=1)  # n2=16 is a power of two
    model = PyrTrainableAssistedModelB(layout, sr_mode="sample")

    B = 2
    gun = tf.zeros((B, 16), dtype=tf.float32)
    comm = tf.zeros((B, 1), dtype=tf.float32)
    # Placeholder lists with correct structure/length K for illustration.
    prev_measurements = [tf.zeros((B, 8), dtype=tf.float32)] * model.depth
    prev_outcomes = [tf.zeros((B, 8), dtype=tf.float32)] * model.depth

    shoot_logit = model([gun, comm, prev_measurements, prev_outcomes])  # shape (B, 1)
    ```

## Serialization

- Standard Keras model behavior applies (tracked sublayers + weights).
- No custom `get_config()` is defined.

!!! warning "Config-based reconstruction"
    If you require full config-based reconstruction, implement `get_config()` and keep it backward compatible.

## Planned (design-spec)

- `get_config()` for stable config-based serialization.

## Deviations

- The parameter name `sr_mode` is retained for backward compatibility, but SR refers to a shared resource.

## Notes for Contributors

- Keep SR terminology consistent: SR = shared resource.
- Keep the per-level layer lists as the source of truth; aliases exist only for legacy code.
- Preserve the requirement `comms_size == 1` unless the pyramid architecture is generalized in the spec.
- Keep the hard bit-to-logit conversion unless a learning-based conversion is introduced intentionally.

## Changelog

- 2026-01-11 — Author: Rob Hendriks
