# gameplay_adapters

> Role: Provide gameplay-facing adapters that translate binary boundary bits (float32 tensors in `{0.0, 1.0}`) to internal hard-logit tensors (and back) for Player A and Player B models.

Location: `Q_Sea_Battle.gameplay_adapters`

## Overview

This module defines gameplay adapters for a "pure-logit" internal Linear and Pyramid model composition. At the gameplay boundary, all inputs/outputs are binary *bits* represented as `tf.float32` tensors with values in `{0.0, 1.0}`; internally, composed models operate on *logits* (real-valued `tf.float32` tensors), where the logical bit value is determined by the sign of the logit. The boundary translation uses a "hard-logit" representation: bit `1` maps to `+beta` and bit `0` maps to `-beta` via `hard_logit`, and internal logits map back to bits via thresholding at `0.0` (`>= 0.0 -> 1.0`, `< 0.0 -> 0.0`).

## Public API

### Functions

#### `hard_logit(bits: tf.Tensor, beta: float) -> tf.Tensor`

Signature: `hard_logit(bits: tf.Tensor, beta: float) -> tf.Tensor`  
Purpose: Map boundary bits in `{0, 1}` to hard logits in `{-beta, +beta}`.  
Arguments: `bits`: Bit tensor (typically float32) with values in `{0, 1}`; `beta`: Logit magnitude to use for 0/1.  
Returns: A `tf.float32` tensor with the same shape as `bits` where `0 -> -beta` and `1 -> +beta`.  
Errors: Not specified.  
Example:
```python
import tensorflow as tf
from Q_Sea_Battle.gameplay_adapters import hard_logit

bits = tf.constant([[0.0, 1.0]], dtype=tf.float32)
logits = hard_logit(bits, beta=10.0)  # [[-10.0, +10.0]]
```

### Constants

Not specified.

### Types

Not specified.

## Dependencies

- `tensorflow` (imported as `tf`)
- Standard library: `dataclasses.dataclass`, `typing.Any`, `typing.Iterable`, `typing.List`, `typing.Sequence`, `typing.Tuple`

## Planned (design-spec)

Not specified.

## Deviations

- The module docstring describes boundary call patterns and contracts for Player A and Player B adapters; these are implemented via `GameplayModelAAdapter.__call__` and `GameplayModelBAdapter.__call__`, while both adapters also expose a deprecated `compute_with_internal` method that prints a warning and forwards to `__call__`.
- Some helper functions exist in the module (prefixed with `_`) but are not part of the documented public API.

## Notes for Contributors

- Boundary validation: Both adapters enforce (by default) that gameplay-facing tensors contain only binary values `{0, 1}` using TensorFlow debugging assertions; these run immediately in eager mode and become `tf.debugging` ops in graph mode.
- Rank handling: Boundary tensors are normalized to rank-2 `(B, D)`; rank-1 `(D,)` inputs are promoted to `(1, D)`.
- Exploration: Both adapters can optionally inject Gaussian noise (`stddev=0.5`) into internal logits before thresholding to bits (`explore=True`), while also optionally returning the raw internal logit (`return_comm_logits` / `return_shoot_logit`).
- Internal model expectations: `GameplayModelAAdapter` expects `internal_model_a.compute_with_internal(field_logits, harden_between_levels=..., beta_for_hardening=...) -> (comm_logits, meas_list, out_list)`; `GameplayModelBAdapter` expects `internal_model_b.compute_with_internal(gun_logits, comm_logits, prev_meas_logits, prev_out_logits, harden_between_levels=..., beta_for_hardening=...) -> (shoot_logit, *rest)`.

## Related

- `GameplayModelAAdapter` (Player A boundary adapter; defined in this module but not documented here as public API per the extraction constraint)
- `GameplayModelBAdapter` (Player B boundary adapter; defined in this module but not documented here as public API per the extraction constraint)

## Changelog

Not specified.