# Q_Sea_Battle.lin_dataset_conversion_utilities

> Role: Utilities for converting canonical linear (depth=1) datasets into training views (layer-wise, internal-model, and full-system/teacher-forced).

Location: `Q_Sea_Battle.lin_dataset_conversion_utilities`

## Overview

This module reshapes and re-encodes a canonical *linear* dataset (fixed depth=1, stored as depth+1 == 2 along axis 1 for traces) into the input/target tuples expected by different training stages. It also provides a representation conversion helper (`apply_rep`) to map bit-like arrays (0/1) into model-facing numeric encodings.

A canonical linear dataset is expected to provide NumPy arrays for field/gun traces, communication traces, measurement input/output traces for players A and B, and a shoot decision bit. All conversions validate required keys and basic shapes before producing outputs.

## Public API

### Functions

#### `apply_rep`

Signature:
```python
def apply_rep(
    x_bits: np.ndarray,
    rep: TrainRep,
    *,
    beta: Union[float, Sequence[float]],
) -> np.ndarray:
```

Purpose: Convert a bit-valued NumPy array (assumed 0/1) into a training representation and return a `float32` array of the same shape.

Arguments:
- `x_bits`: Input NumPy array containing bit values.
- `rep`: Output representation selector (`"bits"`, `"scaled"`, or `"hard_logit"`).
- `beta`: Logit magnitude used when `rep == "hard_logit"`; accepted as a scalar or a 1-element sequence (must be scalar in effect).

Returns:
- A `np.ndarray` of dtype `float32` with the same shape as `x_bits`.

Errors:
- `ValueError`: If `rep` is unknown.
- `ValueError`: If `beta` is provided as a sequence with length not equal to 1 (linear converters require scalar beta).

Example:
```python
import numpy as np
from Q_Sea_Battle.lin_dataset_conversion_utilities import apply_rep

x = np.array([[0, 1, 1]], dtype=np.int32)
y_scaled = apply_rep(x, "scaled", beta=10.0)      # -> [[-0.5, 0.5, 0.5]]
y_logit = apply_rep(x, "hard_logit", beta=2.0)    # -> [[-2.0, 2.0, 2.0]]
```

#### `convert_layer_measure_a`

Signature:
```python
def convert_layer_measure_a(
    ds: CanonicalLinDataset,
    *,
    rep_x: TrainRep = "scaled",
    rep_y: TrainRep = "hard_logit",
    beta: float = 10.0,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
```

Purpose: Create the training view for A's measurement layer, using the field trace at `t=0` as input and A's measurement input at depth=1 as the supervision target.

Arguments:
- `ds`: Canonical linear dataset.
- `rep_x`: Representation for the field input.
- `rep_y`: Representation for the measurement target.
- `beta`: Logit magnitude used for `"hard_logit"` representations.

Returns:
- A mapping `{0: (X, Y)}` where:
- `X = apply_rep(ds["field_bits"][:, 0, :], rep_x, beta=beta)`
- `Y = apply_rep(ds["meas_in_a_bits"][:, 0, :], rep_y, beta=beta)`

Errors:
- `KeyError`: If required keys are missing (via internal validation).
- `ValueError`: If dataset shapes are inconsistent with the linear format (via internal validation).

Example:
```python
from Q_Sea_Battle.lin_dataset_conversion_utilities import convert_layer_measure_a

views = convert_layer_measure_a(ds, rep_x="scaled", rep_y="hard_logit", beta=10.0)
X, Y = views[0]
```

#### `convert_layer_combine_a`

Signature:
```python
def convert_layer_combine_a(
    ds: CanonicalLinDataset,
    *,
    rep_outcome: TrainRep = "hard_logit",
    rep_target: TrainRep = "hard_logit",
    beta: float = 10.0,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
```

Purpose: Create the training view for A's combine layer, training communication bits as a function of A's measurement outcome.

Arguments:
- `ds`: Canonical linear dataset.
- `rep_outcome`: Representation for A's measurement outcome (input).
- `rep_target`: Representation for the communication target.
- `beta`: Logit magnitude used for `"hard_logit"` representations.

Returns:
- A mapping `{0: (outcome_a, comm_target)}` where:
- `outcome_a = apply_rep(ds["meas_out_a_bits"][:, 0, :], rep_outcome, beta=beta)`
- `comm_target = apply_rep(ds["comms_bits"][:, 0, :], rep_target, beta=beta)`

Errors:
- `KeyError`: If required keys are missing (via internal validation).
- `ValueError`: If dataset shapes are inconsistent with the linear format (via internal validation).

Example:
```python
from Q_Sea_Battle.lin_dataset_conversion_utilities import convert_layer_combine_a

views = convert_layer_combine_a(ds, rep_outcome="hard_logit", rep_target="hard_logit", beta=10.0)
outcome_a, comm_target = views[0]
```

#### `convert_layer_measure_b`

Signature:
```python
def convert_layer_measure_b(
    ds: CanonicalLinDataset,
    *,
    rep_x: TrainRep = "scaled",
    rep_y: TrainRep = "hard_logit",
    beta: float = 10.0,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
```

Purpose: Create the training view for B's measurement layer, using the gun trace at `t=0` as input and B's measurement input at depth=1 as the supervision target.

Arguments:
- `ds`: Canonical linear dataset.
- `rep_x`: Representation for the gun input.
- `rep_y`: Representation for the measurement target.
- `beta`: Logit magnitude used for `"hard_logit"` representations.

Returns:
- A mapping `{0: (X, Y)}` where:
- `X = apply_rep(ds["gun_bits"][:, 0, :], rep_x, beta=beta)`
- `Y = apply_rep(ds["meas_in_b_bits"][:, 0, :], rep_y, beta=beta)`

Errors:
- `KeyError`: If required keys are missing (via internal validation).
- `ValueError`: If dataset shapes are inconsistent with the linear format (via internal validation).

Example:
```python
from Q_Sea_Battle.lin_dataset_conversion_utilities import convert_layer_measure_b

views = convert_layer_measure_b(ds)
X, Y = views[0]
```

#### `convert_layer_combine_b`

Signature:
```python
def convert_layer_combine_b(
    ds: CanonicalLinDataset,
    *,
    rep_outcome_b: TrainRep = "hard_logit",
    rep_comm_in: TrainRep = "hard_logit",
    rep_shoot: TrainRep = "hard_logit",
    beta: float = 10.0,
) -> Dict[int, Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]]:
```

Purpose: Create the training view for B's combine layer, where inputs are `(B measurement outcome, comms-in)` and the target is the shoot bit.

Arguments:
- `ds`: Canonical linear dataset.
- `rep_outcome_b`: Representation for B's measurement outcome.
- `rep_comm_in`: Representation for the communication input to B.
- `rep_shoot`: Representation for the shoot target.
- `beta`: Logit magnitude used for `"hard_logit"` representations.

Returns:
- A mapping `{0: ((outcome_b, comm_in), shoot_target)}` where:
- `outcome_b = apply_rep(ds["meas_out_b_bits"][:, 0, :], rep_outcome_b, beta=beta)`
- `comm_in = apply_rep(ds["comms_bits"][:, 0, :], rep_comm_in, beta=beta)`
- `shoot_target = apply_rep(ds["shoot"][:, :], rep_shoot, beta=beta)`

Errors:
- `KeyError`: If required keys are missing (via internal validation).
- `ValueError`: If dataset shapes are inconsistent with the linear format (via internal validation).

Example:
```python
from Q_Sea_Battle.lin_dataset_conversion_utilities import convert_layer_combine_b

views = convert_layer_combine_b(ds, rep_shoot="hard_logit", beta=10.0)
(inputs, shoot_target) = views[0]
(outcome_b, comm_in) = inputs
```

#### `convert_internal_model_a`

Signature:
```python
def convert_internal_model_a(
    ds: CanonicalLinDataset,
    *,
    rep_field: TrainRep = "scaled",
    rep_comm_target: TrainRep = "hard_logit",
    rep_meas_target: TrainRep = "hard_logit",
    rep_out_target: TrainRep = "hard_logit",
    beta: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
```

Purpose: Create inputs/targets for training A's internal model as a single-step (depth=1) view, with measurement/outcome targets wrapped in 1-element lists.

Arguments:
- `ds`: Canonical linear dataset.
- `rep_field`: Representation for the field input (`field_bits` at `t=0`).
- `rep_comm_target`: Representation for the communication target (`comms_bits` at `t=0`).
- `rep_meas_target`: Representation for the measurement-input target (A).
- `rep_out_target`: Representation for the measurement-outcome target (A).
- `beta`: Logit magnitude used for `"hard_logit"` representations.

Returns:
- `(field_0, comm_0, meas_list, out_list)` where:
- `field_0 = apply_rep(ds["field_bits"][:, 0, :], rep_field, beta=beta)`
- `comm_0 = apply_rep(ds["comms_bits"][:, 0, :], rep_comm_target, beta=beta)`
- `meas_list = [apply_rep(ds["meas_in_a_bits"][:, 0, :], rep_meas_target, beta=beta)]`
- `out_list = [apply_rep(ds["meas_out_a_bits"][:, 0, :], rep_out_target, beta=beta)]`

Errors:
- `KeyError`: If required keys are missing (via internal validation).
- `ValueError`: If dataset shapes are inconsistent with the linear format (via internal validation).

Example:
```python
from Q_Sea_Battle.lin_dataset_conversion_utilities import convert_internal_model_a

field_0, comm_0, meas_list, out_list = convert_internal_model_a(ds, beta=10.0)
meas_a_0 = meas_list[0]
out_a_0 = out_list[0]
```

#### `convert_internal_model_b`

Signature:
```python
def convert_internal_model_b(
    ds: CanonicalLinDataset,
    *,
    rep_gun: TrainRep = "scaled",
    rep_comm_in: TrainRep = "hard_logit",
    rep_prev_meas: TrainRep = "hard_logit",
    rep_prev_out: TrainRep = "hard_logit",
    rep_shoot_target: TrainRep = "bits",
    rep_meas_b_target: TrainRep = "bits",
    rep_out_b_target: TrainRep = "bits",
    beta: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray]:
```

Purpose: Create inputs/targets for training B's internal model. B conditions on its gun state at `t=0`, received comm bits at `t=0`, and A's previous-step measurement traces; targets include B's measurement traces and the shoot decision.

Arguments:
- `ds`: Canonical linear dataset.
- `rep_gun`: Representation for the gun input.
- `rep_comm_in`: Representation for the communication input to B.
- `rep_prev_meas`: Representation for A's previous measurement input.
- `rep_prev_out`: Representation for A's previous measurement outcome.
- `rep_shoot_target`: Representation for the shoot target.
- `rep_meas_b_target`: Representation for B's measurement-input target.
- `rep_out_b_target`: Representation for B's measurement-outcome target.
- `beta`: Logit magnitude used for `"hard_logit"` representations.

Returns:
- `(gun_0, comm_0, prev_meas_list, prev_out_list, meas_b_list, out_b_list, shoot_target)` where:
- `gun_0 = apply_rep(ds["gun_bits"][:, 0, :], rep_gun, beta=beta)`
- `comm_0 = apply_rep(ds["comms_bits"][:, 0, :], rep_comm_in, beta=beta)`
- `prev_meas_list = [apply_rep(ds["meas_in_a_bits"][:, 0, :], rep_prev_meas, beta=beta)]`
- `prev_out_list = [apply_rep(ds["meas_out_a_bits"][:, 0, :], rep_prev_out, beta=beta)]`
- `meas_b_list = [apply_rep(ds["meas_in_b_bits"][:, 0, :], rep_meas_b_target, beta=beta)]`
- `out_b_list = [apply_rep(ds["meas_out_b_bits"][:, 0, :], rep_out_b_target, beta=beta)]`
- `shoot_target = apply_rep(ds["shoot"][:, :], rep_shoot_target, beta=beta)`

Errors:
- `KeyError`: If required keys are missing (via internal validation).
- `ValueError`: If dataset shapes are inconsistent with the linear format (via internal validation).

Example:
```python
from Q_Sea_Battle.lin_dataset_conversion_utilities import convert_internal_model_b

gun_0, comm_0, prev_meas_list, prev_out_list, meas_b_list, out_b_list, shoot_target = convert_internal_model_b(ds)
```

#### `convert_full_system`

Signature:
```python
def convert_full_system(
    ds: CanonicalLinDataset,
    *,
    rep_field: TrainRep = "scaled",
    rep_gun: TrainRep = "scaled",
    rep_teacher_comm_trace: TrainRep = "bits",
    rep_teacher_meas_a: TrainRep = "bits",
    rep_teacher_out_a: TrainRep = "bits",
    rep_shoot_target: TrainRep = "bits",
    beta: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], np.ndarray]:
```

Purpose: Create the training view for full-system training with teacher traces, providing initial inputs at `t=0`, a full communication trace over axis 1 (`depth+1 == 2`), teacher-forced A measurement input/outcome for the single depth step, and the shoot target.

Arguments:
- `ds`: Canonical linear dataset.
- `rep_field`: Representation for the initial field input.
- `rep_gun`: Representation for the initial gun input.
- `rep_teacher_comm_trace`: Representation for the teacher communication trace (`comms_bits` over both time steps).
- `rep_teacher_meas_a`: Representation for teacher A measurement input.
- `rep_teacher_out_a`: Representation for teacher A measurement outcome.
- `rep_shoot_target`: Representation for the shoot target.
- `beta`: Logit magnitude used for `"hard_logit"` representations.

Returns:
- `(field_0, gun_0, comm_trace, meas_a_list, out_a_list, shoot_target)` where:
- `field_0 = apply_rep(ds["field_bits"][:, 0, :], rep_field, beta=beta)`
- `gun_0 = apply_rep(ds["gun_bits"][:, 0, :], rep_gun, beta=beta)`
- `comm_trace = apply_rep(ds["comms_bits"][:, :, :], rep_teacher_comm_trace, beta=beta)`
- `meas_a_list = [apply_rep(ds["meas_in_a_bits"][:, 0, :], rep_teacher_meas_a, beta=beta)]`
- `out_a_list = [apply_rep(ds["meas_out_a_bits"][:, 0, :], rep_teacher_out_a, beta=beta)]`
- `shoot_target = apply_rep(ds["shoot"][:, :], rep_shoot_target, beta=beta)`

Errors:
- `KeyError`: If required keys are missing (via internal validation).
- `ValueError`: If dataset shapes are inconsistent with the linear format (via internal validation).

Example:
```python
from Q_Sea_Battle.lin_dataset_conversion_utilities import convert_full_system

field_0, gun_0, comm_trace, meas_a_list, out_a_list, shoot_target = convert_full_system(ds)
```

### Constants

None.

### Types

#### `TrainRep`

Type:
```python
TrainRep = Literal["bits", "scaled", "hard_logit"]
```

Purpose: Selects how bit-valued arrays are presented to models.
- `"bits"`: Values remain in `{0, 1}` (float32).
- `"scaled"`: Values are shifted to `{-0.5, +0.5}` via `x - 0.5`.
- `"hard_logit"`: Values are mapped to logits in `{-beta, +beta}` via `beta * (2x - 1)`.

#### `CanonicalLinDataset`

Type:
```python
class CanonicalLinDataset(TypedDict):
    field_bits: np.ndarray
    gun_bits: np.ndarray
    comms_bits: np.ndarray
    meas_in_a_bits: np.ndarray
    meas_out_a_bits: np.ndarray
    meas_in_b_bits: np.ndarray
    meas_out_b_bits: np.ndarray
    shoot: np.ndarray
```

Purpose: Typed view of the expected canonical linear dataset dictionary.

Required keys and expected shapes:
- `field_bits`: `(N, 2, n2)` player A's field trace; axis 1 is time (depth+1).
- `gun_bits`: `(N, 2, n2)` player B's gun trace; axis 1 is time (depth+1).
- `comms_bits`: `(N, 2, m)` communication trace; axis 1 is time (depth+1).
- `meas_in_a_bits`: `(N, 1, n2)` A measurement input at depth=1.
- `meas_out_a_bits`: `(N, 1, n2)` A measurement outcome at depth=1.
- `meas_in_b_bits`: `(N, 1, n2)` B measurement input at depth=1.
- `meas_out_b_bits`: `(N, 1, n2)` B measurement outcome at depth=1.
- `shoot`: `(N, 1)` shoot decision bit.

## Dependencies

- `numpy` (imported as `np`)
- `typing` (`Any`, `Dict`, `List`, `Literal`, `Sequence`, `Tuple`, `TypedDict`, `Union`)
- `__future__.annotations`

## Planned (design-spec)

Not specified.

## Deviations

Not specified.

## Notes for Contributors

- Most public converters call an internal shape validator before slicing; ensure any new conversion functions validate required keys and the linear depth constraint (`depth+1 == 2`) to keep errors consistent.  
- `apply_rep` accepts `beta` as a scalar or 1-element sequence for compatibility, but enforces an effective scalar; keep this behavior consistent if extending representations.

## Related

- Unknown.

## Changelog

- Not specified.