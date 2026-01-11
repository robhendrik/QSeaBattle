# Class PyrTrainableAssistedModelA

**Module import path**: `Q_Sea_Battle.pyr_trainable_assisted_model_a.PyrTrainableAssistedModelA`

> Player-A pyramid assisted model with per-level measurement/combine layers and a per-level PR-assisted layer.

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

`PyrTrainableAssistedModelA` produces communication logits for Player A by iteratively halving the active state length.
At each level it applies:

1. `PyrMeasurementLayerA` (or a provided measurement layer) to compute `meas^ℓ` of length `L/2`.
2. `PRAssistedLayer` to produce a PR-assisted outcome `out^ℓ` of length `L/2`.
3. `PyrCombineLayerA` (or a provided combine layer) to compute next state of length `L/2`.

After `K` levels, the state has shape `(B, 1)`. The final bit is converted into a **hard logit** for downstream
Bernoulli usage.

## Architecture pipeline

At each level with active length `L`:

- `state (B, L)` → `measure_layer` → `meas (B, L/2)`
- `meas (B, L/2)` → `PRAssistedLayer(length=L/2, mode=sr_mode)` → `out (B, L/2)`
- `(state (B, L), out (B, L/2))` → `combine_layer` → `next_state (B, L/2)`

## Constructor

### Signature

- `PyrTrainableAssistedModelA(`
  `game_layout: Any,`
  `p_high: float = 0.9,`
  `sr_mode: str = "sample",`
  `measure_layers: Sequence[tf.keras.layers.Layer] | None = None,`
  `combine_layers: Sequence[tf.keras.layers.Layer] | None = None,`
  `name: str | None = None`
  `) -> PyrTrainableAssistedModelA`

### Arguments

- `game_layout`: `Any`, scalar.
  - GameLayout-like object providing:
    - `field_size: int`, scalar, or `n2: int`, scalar.
    - `comms_size: int`, scalar.
- `p_high`: `float`, scalar.
  - Correlation parameter forwarded to each `PRAssistedLayer`.
- `sr_mode`: `str`, scalar, one of `('sample', 'expected')`.
  - `"sample"` is required for gameplay and dataset generation.
  - `"expected"` may be used for analysis.
- `measure_layers`: `Sequence[tf.keras.layers.Layer]` or `None`, shape `(K,)`.
  - If provided, must have length `K = log2(n2)`.
  - Each layer consumes `(B, L)` and returns `(B, L/2)` for its level.
- `combine_layers`: `Sequence[tf.keras.layers.Layer]` or `None`, shape `(K,)`.
  - If provided, must have length `K = log2(n2)`.
  - Each layer consumes `(state, out)` and returns `(B, L/2)` for its level.
- `name`: `str` or `None`, scalar.

### Returns

- `PyrTrainableAssistedModelA`, scalar.

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
  - `sr_layers[level]` has `length = L/2` for that level.

### Errors

- Raises `ValueError` if `n2 <= 0`.
- Raises `ValueError` if `n2` is not a power of two.
- Raises `ValueError` if `comms_size != 1`.
- Raises `ValueError` if provided layer sequences do not have length `K`.

## Backward compatibility aliases

These aliases are retained for legacy code:

- `measure_layer` → `measure_layers[0]`
- `combine_layer` → `combine_layers[0]`

!!! note "Alias scope"
    The forward pass uses the per-level lists. Aliases point to the first level only.

## Public Methods

### call

#### Signature

- `call(field_batch: tf.Tensor) -> tf.Tensor`

#### Arguments

- `field_batch`: `tf.Tensor`, dtype `float32` (or castable), shape `(B, n2)`.

#### Returns

- `comm_logits`: `tf.Tensor`, dtype `float32`, shape `(B, 1)`.

#### Preconditions

- `field_batch` is rank-2 and last dim equals `n2`.

#### Postconditions

- Returns the same `comm_logits` as `compute_with_internal(...)[0]`.

#### Errors

- Raises `ValueError` if `field_batch` is not rank-2.

### compute_with_internal

#### Signature

- `compute_with_internal(field_batch: tf.Tensor) -> tuple[tf.Tensor, list[tf.Tensor], list[tf.Tensor]]`

#### Arguments

- `field_batch`: `tf.Tensor`, dtype `float32` (or castable), shape `(B, n2)`.

#### Returns

- `comm_logits`: `tf.Tensor`, dtype `float32`, shape `(B, 1)`.
  - Hard logit conversion from the final bit; values are approximately `-10` or `+10`.
- `measurements`: `list[tf.Tensor]`, length `K`.
  - Each element: `tf.Tensor`, dtype `float32`, shape `(B, L/2)` at that level.
- `outcomes`: `list[tf.Tensor]`, length `K`.
  - Each element: `tf.Tensor`, dtype `float32`, shape `(B, L/2)` at that level.
  - Represents PR-assisted outcomes returned by `PRAssistedLayer`.

#### Preconditions

- `field_batch` is rank-2 and last dim equals `n2`.
- For each level, active length `L` is even.

#### Postconditions

- `len(measurements) == len(outcomes) == K`.
- Lists are aligned by level.
- Final `comm_logits` corresponds to the final state bit.

#### Errors

- Raises `ValueError` if `field_batch` is not rank-2.
- Propagates `tf.errors.InvalidArgumentError` from per-level runtime assertions.

!!! example "Minimal usage"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.pyr_trainable_assisted_model_a import PyrTrainableAssistedModelA

    layout = GameLayout(field_size=4, comms_size=1)  # n2=16 is a power of two
    model = PyrTrainableAssistedModelA(layout, sr_mode="sample")

    field = tf.zeros((2, 16), dtype=tf.float32)  # B=2
    comm_logits, measurements, outcomes = model.compute_with_internal(field)
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
- Keep the final bit-to-logit conversion hard unless a learning-based conversion is introduced intentionally.

## Changelog

- 2026-01-11 — Author: Rob Hendriks
