# Class LinTrainableAssistedModelB

**Module import path**: `Q_Sea_Battle.lin_trainable_assisted_model_b.LinTrainableAssistedModelB`

> Player B model for the linear trainable-assisted baseline.
> Consumes intermediates from Player A and produces a shoot logit.

!!! note "Parent class"
    Inherits from `tf.keras.Model`.

!!! note "Derived symbols"
    Let `field_size = n`, `comms_size = m`, and `n2 = n**2`.

!!! note "Terminology"
    This model uses `sr_mode` where **SR = shared resource** (not "shared randomness").

## Overview

`LinTrainableAssistedModelB` consumes tensors produced by Player A's
`LinTrainableAssistedModelA.compute_with_internal(...)`:

- `prev_meas_list[0]`: `tf.Tensor`, dtype `float32`, shape `(B, n2)`.
- `prev_out_list[0]`: `tf.Tensor`, dtype `float32`, shape `(B, n2)` (PR-assisted outcomes from Player A).

It then performs a second measurement using the PR-assisted layer, and combines the resulting outcomes with the
communication vector to produce a shoot logit.

## Pipeline

- `gun (B, n2)` → `LinMeasurementLayerB` → `meas_probs_b (B, n2)` in `[0, 1]`
- `meas_probs_b + prev tensors` → `PRAssistedLayer` (second measurement) → `outcomes_b (B, n2)`
- `(outcomes_b, comm (B, m))` → `LinCombineLayerB` → `shoot_logit (B, 1)`

## Constructor

### Signature

- `LinTrainableAssistedModelB(field_size: int, comms_size: int, *, sr_mode: str = "expected", seed: int | None = 0,`
  `p_high: float = 0.9, resource_index: int = 0, hidden_units_meas: Sequence[int] = (64,),`
  `hidden_units_combine: int | Sequence[int] = (64, 64), name: str | None = None, **kwargs)`

### Arguments

- `field_size`: `int`, scalar.
  - Defines `n2 = field_size**2`.
- `comms_size`: `int`, scalar.
  - Number of communication channels `m`.
- `sr_mode`: `str`, scalar, one of `{"expected", "sample"}`.
  - Backward-compatible naming where SR = shared resource.
- `seed`: `int` or `None`, scalar.
  - Seed used by the PR-assisted layer in `"sample"` mode.
- `p_high`: `float`, scalar.
  - Correlation parameter for the PR-assisted layer.
- `resource_index`: `int`, scalar.
  - Stream identifier for the PR-assisted layer.
- `hidden_units_meas`: `Sequence[int]`, shape `(H_meas,)`.
  - Hidden widths for `LinMeasurementLayerB`.
- `hidden_units_combine`: `int` or `Sequence[int]`, shape `()` or `(H_comb,)`.
  - Hidden widths for `LinCombineLayerB`.
- `name`: `str` or `None`, scalar.
- `**kwargs`: `dict[str, Any]`, scalar.
  - Forwarded to `tf.keras.Model`.

### Returns

- `LinTrainableAssistedModelB`, scalar.

### Preconditions

- `field_size >= 1` and `n2 = field_size**2`.
- `comms_size >= 1`.
- `sr_mode` is `"expected"` or `"sample"`.
- `0.0 <= p_high <= 1.0`.

### Postconditions

- Sets:
  - `self.field_size`: `int`, scalar.
  - `self.comms_size`: `int`, scalar.
  - `self.n2`: `int`, scalar (`field_size**2`).
- Creates submodules:
  - `self.measurement`: `LinMeasurementLayerB`.
  - `self.pr_assisted`: `PRAssistedLayer`.
  - `self.combine`: `LinCombineLayerB`.
- Backward-compatible aliases are set (see below).

### Errors

- Propagates `ValueError` from submodules for invalid dimensions.
- Propagates `ValueError` from `PRAssistedLayer` for invalid `mode` or parameter ranges.

## Backward compatibility aliases

This model exposes deprecated-but-functional aliases:

- `measure_layer` → `measurement`
- `sr_layer` → `pr_assisted` (deprecated alias retained for older notebooks/scripts)
- `combine_layer` → `combine`

Preferred names:

- `pr_assisted` (or `pr_layer`, `resource_layer`) for theR-assisted resource layer.
- `measurement`, `combine` for the other stages.

!!! warning "Deprecated alias"
    Use `pr_assisted` instead of `sr_layer` in new code.

## Public Methods

### call

#### Signature

- `call(inputs: list[tf.Tensor] | tuple[tf.Tensor, ...], training: bool = False) -> tf.Tensor`

#### Arguments

- `inputs`: `list[tf.Tensor]` or `tuple[tf.Tensor, ...]`, length `4`, containing:
  - `gun`: `tf.Tensor`, dtype `float32` (or castable), shape `(n2,)` or `(B, n2)`.
  - `comm`: `tf.Tensor`, dtype `float32` (or castable), shape `(m,)` or `(B, m)`.
  - `prev_meas_list`: `list[tf.Tensor]` or `tf.Tensor`, where first element (or tensor) is shape `(B, n2)`.
  - `prev_out_list`: `list[tf.Tensor]` or `tf.Tensor`, where first element (or tensor) is shape `(B, n2)`.
- `training`: `bool`, scalar.

#### Returns

- `shoot_logit`: `tf.Tensor`, dtype `float32`, shape:
  - `(1,)` or `(B, 1)` depending on input batching.

#### Preconditions

- `inputs` length is 4.
- `gun` last dim equals `n2` and `comm` last dim equals `m`.
- Previous tensors are provided and are compatible with batch size `B`.

#### Postconditions

- Produces a shoot logit; no internal mutable state is stored.

#### Errors

- Propagates TensorFlow/Keras errors (including shape mismatches).
- May raise `TypeError`/`ValueError` if `inputs` structure is invalid.

!!! example "Minimal usage"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.lin_trainable_assisted_model_b import LinTrainableAssistedModelB

    model = LinTrainableAssistedModelB(field_size=4, comms_size=1, sr_mode="expected")

    B = 2
    n2 = 16
    gun = tf.zeros((B, n2), dtype=tf.float32)
    comm = tf.zeros((B, 1), dtype=tf.float32)
    prev_meas = [tf.zeros((B, n2), dtype=tf.float32)]
    prev_out = [tf.zeros((B, n2), dtype=tf.float32)]

    logit = model([gun, comm, prev_meas, prev_out])  # shape (B, 1)
    ```

## Serialization

- Standard Keras model behavior applies (config + weights).
- This class does not define `get_config()` explicitly; if you require full config-based reconstruction, add `get_config()`.

## Planned (design-spec)

- `get_config()` implementation for stable config-based serialization.

## Deviations

- None identified.

## Notes for Contributors

- Keep PR-assisted terminology consistent: use “PR-assisted resource” / “PR-assisted layer”.
- Preserve backward-compatible aliases, but keep them documented as deprecated.
- If you introduce stricter shape checks, ensure they work for both unbatched `(n2,)` / `(m,)` and batched `(B, ...)`
  inputs.

## Changelog

- 2026-01-11 — Author: Technical Documentation Team
