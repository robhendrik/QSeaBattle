# Class LinCombineLayerB

**Module import path**: `Q_Sea_Battle.lin_combine_layer_b.LinCombineLayerB`

> Learnable mapping `(outcomes, comm) -> shoot_logit`.

!!! note "Parent class"
    Inherits from `tf.keras.layers.Layer`.

!!! note "Derived symbols"
    Let `field_size = n`, `n2 = n**2`, and `comms_size = m`.

## Overview

`LinCombineLayerB` combines:

- `outcomes`: per-cell outcome vector (typically produced by a PR-assisted resource), and
- `comm`: communication vector

into a **single** shoot logit.

- Output is a logit (unbounded real value), not squashed.
- The layer supports both unbatched and batched inputs and preserves rank.

## Constructor

### Signature

- `LinCombineLayerB(comms_size: int, hidden_units: int | Sequence[int] = 64, name: str | None = None, **kwargs)`

### Arguments

- `comms_size`: `int`, scalar.
  - Number of communication channels `m`. Used for shape checks by callers.
- `hidden_units`: `int` or `Sequence[int]`, shape `()` or `(H,)`.
  - Dense-ReLU layer widths, in order.
- `name`: `str` or `None`, scalar.
- `**kwargs`: `dict[str, Any]`, scalar.
  - Forwarded to `tf.keras.layers.Layer`.

### Returns

- `LinCombineLayerB`, scalar.

### Preconditions

- `comms_size >= 1`.

### Postconditions

- `self.comms_size`: `int`, scalar.
- `self.hidden_units`: `tuple[int, ...]`, shape `(H,)`.
- The output Dense layer has width `1` and no activation.

### Errors

- May raise `TypeError`/`ValueError` if `comms_size` or `hidden_units` cannot be converted to integers.

## Public Methods

### call

#### Signature

- `call(outcomes: tf.Tensor, comm: tf.Tensor, training: bool = False) -> tf.Tensor`

#### Arguments

- `outcomes`: `tf.Tensor`, dtype any, shape:
  - `(n2,)`, or
  - `(B, n2)`.
- `comm`: `tf.Tensor`, dtype any, shape:
  - `(m,)`, or
  - `(B, m)`.
- `training`: `bool`, scalar.

#### Returns

- `shoot_logit`: `tf.Tensor`, dtype inferred by TensorFlow, shape:
  - `(1,)` if input outcomes are `(n2,)`, or
  - `(B, 1)` if input outcomes are `(B, n2)`.

#### Preconditions

- Inputs are rank-1 or rank-2 and are batch-compatible after optional expansion.
- Caller is responsible for ensuring `outcomes` last dim equals `n2` and `comm` last dim equals `m`.

#### Postconditions

- Output is a logit suitable for sigmoid/sampling downstream.

#### Errors

- Propagates TensorFlow/Keras errors (e.g., concat shape mismatch).

!!! example "Minimal usage"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.lin_combine_layer_b import LinCombineLayerB

    n2 = 16
    m = 2
    layer = LinCombineLayerB(comms_size=m, hidden_units=(32, 32))

    outcomes = tf.zeros((n2,), dtype=tf.float32)
    comm = tf.zeros((m,), dtype=tf.float32)
    logit = layer(outcomes, comm)  # shape (1,)
    ```

## Serialization

- Standard Keras layer behavior applies.
- This class does not define `get_config()` explicitly; base-class serialization applies to tracked sublayers and weights.

## Planned (design-spec)

- None identified.

## Deviations

- No explicit shape checks for `n2` or `m` are performed inside this layer; callers enforce consistency.

## Notes for Contributors

- Keep output as a logit (no sigmoid here).
- Preserve rank behavior:
  - rank-1 inputs → rank-1 output `(1,)`
  - rank-2 inputs → rank-2 output `(B, 1)`

## Changelog

- 2026-01-11 — Author: Technical Documentation Team
