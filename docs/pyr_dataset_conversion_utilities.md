# dataset_conversion_utility
> Role: Training-view converters for canonical QSeaBattle pyramid datasets; deterministic cropping and representation conversion utilities for producing NumPy arrays expected by trainable layers/models.
Location: `Q_Sea_Battle.pyr_dataset_conversion_utilities`

## Overview

This module provides pure, deterministic conversions from a canonical pyramid dataset dict (typically loaded from `.npz` or produced by a generator) into cropped and representation-converted NumPy arrays suitable for training layers/models. Cropping follows pyramid geometry: at level `d`, state vectors are cropped to active width `L_d`, and measurement vectors for transition `d` are cropped to `k_d`. Supported representations are `"bits"`, `"scaled"` (`x - 0.5`), and `"hard_logit"` (`beta * (2x - 1)`), returning `np.float32` outputs. Mixed-beta conversion (when `beta` is a sequence) is deterministic: contiguous split over samples, per-chunk conversion, and round-robin interleave along axis 0.

## Public API

### Functions

#### `infer_depth_from_dataset(ds: CanonicalPyrDataset) -> int`

**Signature:** `infer_depth_from_dataset(ds: CanonicalPyrDataset) -> int`  
**Purpose:** Infer pyramid depth from a canonical dataset using the measurement-input trace shape.  
**Arguments:**  
- `ds`: Canonical dataset.  
**Returns:** The inferred depth, i.e. `ds["meas_in_a_bits"].shape[1]`.  
**Errors:** Not specified.  
**Example:**
```python
depth = infer_depth_from_dataset(ds)
```

#### `infer_n2_from_dataset(ds: CanonicalPyrDataset) -> int`

**Signature:** `infer_n2_from_dataset(ds: CanonicalPyrDataset) -> int`  
**Purpose:** Infer the base state width `n2` from a canonical dataset using the field trace shape.  
**Arguments:**  
- `ds`: Canonical dataset.  
**Returns:** The inferred base width, i.e. `ds["field_bits"].shape[2]`.  
**Errors:** Not specified.  
**Example:**
```python
n2 = infer_n2_from_dataset(ds)
```

#### `level_sizes(n2: int, d: int) -> tuple[int, int]`

**Signature:** `level_sizes(n2: int, d: int) -> tuple[int, int]`  
**Purpose:** Compute pyramid widths for level `d` using `L_d = n2 // 2**d` and `k_d = L_d // 2`.  
**Arguments:**  
- `n2`: Base width at level 0. Expected to be a power of two.  
- `d`: Level index (0-based).  
**Returns:** A tuple `(L_d, k_d)`.  
**Errors:**  
- `ValueError`: If `d < 0`.  
**Example:**
```python
L_d, k_d = level_sizes(n2=256, d=3)
```

#### `apply_rep(x_bits: np.ndarray, rep: TrainRep, *, beta: Union[float, Sequence[float]]) -> np.ndarray`

**Signature:** `apply_rep(x_bits: np.ndarray, rep: TrainRep, *, beta: Union[float, Sequence[float]]) -> np.ndarray`  
**Purpose:** Convert a bit-valued array to a training representation (`"bits"`, `"scaled"`, `"hard_logit"`) with deterministic mixed-beta behavior when `beta` is a sequence.  
**Arguments:**  
- `x_bits`: Input array; sample axis is axis 0; typically `float32` with values in `{0.0, 1.0}`.  
- `rep`: Output representation name.  
- `beta`: Logit magnitude parameter (scalar) or a sequence for mixed-beta mode.  
**Returns:** Converted array of dtype `np.float32` with the same shape as `x_bits`.  
**Errors:**  
- `ValueError`: If `rep` is unknown or if `beta` is an empty sequence.  
**Example:**
```python
x_scaled = apply_rep(x_bits, "scaled", beta=10.0)
x_logits = apply_rep(x_bits, "hard_logit", beta=[5.0, 10.0, 20.0])
```

#### `convert_layer_measure_a(ds: CanonicalPyrDataset, *, rep_x: TrainRep = "scaled", rep_y: TrainRep = "hard_logit", beta: float | Sequence[float] = 10.0) -> Dict[int, Tuple[np.ndarray, np.ndarray]]`

**Signature:** `convert_layer_measure_a(ds: CanonicalPyrDataset, *, rep_x: TrainRep = "scaled", rep_y: TrainRep = "hard_logit", beta: float | Sequence[float] = 10.0) -> Dict[int, Tuple[np.ndarray, np.ndarray]]`  
**Purpose:** Build per-level training pairs for the Measure-A layer: `X = field_bits[:, d, :L_d]`, `Y = meas_in_a_bits[:, d, :k_d]` with per-array representation conversion.  
**Arguments:**  
- `ds`: Canonical dataset.  
- `rep_x`: Representation for `X`.  
- `rep_y`: Representation for `Y`.  
- `beta`: Logit magnitude (scalar) or mixed-beta sequence.  
**Returns:** Dict mapping level `d` to `(X, Y)` where `X` has shape `(N, L_d)` and `Y` has shape `(N, k_d)`.  
**Errors:** Not specified (may raise from internal validation and conversion helpers).  
**Example:**
```python
pairs_by_level = convert_layer_measure_a(ds, rep_x="scaled", rep_y="hard_logit", beta=10.0)
X0, Y0 = pairs_by_level[0]
```

#### `convert_layer_combine_a(ds: CanonicalPyrDataset, *, rep_field: TrainRep = "scaled", rep_outcome: TrainRep = "hard_logit", rep_target: TrainRep = "hard_logit", beta: float | Sequence[float] = 10.0) -> Dict[int, Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]]`

**Signature:** `convert_layer_combine_a(ds: CanonicalPyrDataset, *, rep_field: TrainRep = "scaled", rep_outcome: TrainRep = "hard_logit", rep_target: TrainRep = "hard_logit", beta: float | Sequence[float] = 10.0) -> Dict[int, Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]]`  
**Purpose:** Build per-level training tuples for the Combine-A layer: inputs `(field_d, out_a_d)` and target `field_d1`, all cropped to `(L_d, k_d, L_{d+1})` and converted to requested representations.  
**Arguments:**  
- `ds`: Canonical dataset.  
- `rep_field`: Representation for the field input at level `d`.  
- `rep_outcome`: Representation for A's measurement outcome at transition `d`.  
- `rep_target`: Representation for the field target at level `d+1`.  
- `beta`: Logit magnitude (scalar) or mixed-beta sequence.  
**Returns:** Dict mapping level `d` to `((field_d, out_a_d), field_d1)` with shapes `field_d: (N, L_d)`, `out_a_d: (N, k_d)`, `field_d1: (N, L_{d+1})`.  
**Errors:** Not specified (may raise from internal validation and conversion helpers).  
**Example:**
```python
tuples_by_level = convert_layer_combine_a(ds, beta=[10.0, 20.0])
((field_d, out_a_d), field_d1) = tuples_by_level[0]
```

#### `convert_layer_measure_b(ds: CanonicalPyrDataset, *, rep_x: TrainRep = "scaled", rep_y: TrainRep = "hard_logit", beta: float | Sequence[float] = 10.0) -> Dict[int, Tuple[np.ndarray, np.ndarray]]`

**Signature:** `convert_layer_measure_b(ds: CanonicalPyrDataset, *, rep_x: TrainRep = "scaled", rep_y: TrainRep = "hard_logit", beta: float | Sequence[float] = 10.0) -> Dict[int, Tuple[np.ndarray, np.ndarray]]`  
**Purpose:** Build per-level training pairs for the Measure-B layer: `X = gun_bits[:, d, :L_d]`, `Y = meas_in_b_bits[:, d, :k_d]` with representation conversion.  
**Arguments:**  
- `ds`: Canonical dataset.  
- `rep_x`: Representation for `X`.  
- `rep_y`: Representation for `Y`.  
- `beta`: Logit magnitude (scalar) or mixed-beta sequence.  
**Returns:** Dict mapping level `d` to `(X, Y)` where `X` has shape `(N, L_d)` and `Y` has shape `(N, k_d)`.  
**Errors:** Not specified (may raise from internal validation and conversion helpers).  
**Example:**
```python
pairs_by_level = convert_layer_measure_b(ds, rep_x="scaled", rep_y="hard_logit", beta=10.0)
X0, Y0 = pairs_by_level[0]
```

#### `convert_layer_combine_b(ds: CanonicalPyrDataset, *, rep_gun: TrainRep = "scaled", rep_outcome_b: TrainRep = "hard_logit", rep_comm_in: TrainRep = "hard_logit", rep_gun_next: TrainRep = "hard_logit", rep_comm_next: TrainRep = "hard_logit", beta: float | Sequence[float] = 10.0) -> Dict[int, Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]`

**Signature:** `convert_layer_combine_b(ds: CanonicalPyrDataset, *, rep_gun: TrainRep = "scaled", rep_outcome_b: TrainRep = "hard_logit", rep_comm_in: TrainRep = "hard_logit", rep_gun_next: TrainRep = "hard_logit", rep_comm_next: TrainRep = "hard_logit", beta: float | Sequence[float] = 10.0) -> Dict[int, Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]`  
**Purpose:** Build per-level training tuples for the Combine-B layer: inputs `(gun_d, out_b_d, comm_d)` and targets `(gun_d1, comm_d1)` with pyramid cropping and representation conversion.  
**Arguments:**  
- `ds`: Canonical dataset.  
- `rep_gun`: Representation for the gun input at level `d`.  
- `rep_outcome_b`: Representation for B's measurement outcome at transition `d`.  
- `rep_comm_in`: Representation for the comm input at level `d`.  
- `rep_gun_next`: Representation for the gun target at level `d+1`.  
- `rep_comm_next`: Representation for the comm target at level `d+1`.  
- `beta`: Logit magnitude (scalar) or mixed-beta sequence.  
**Returns:** Dict mapping level `d` to `((gun_d, out_b_d, comm_d), (gun_d1, comm_d1))` with shapes `gun_d: (N, L_d)`, `out_b_d: (N, k_d)`, `comm_d: (N, 1)`, `gun_d1: (N, L_{d+1})`, `comm_d1: (N, 1)`.  
**Errors:** Not specified (may raise from internal validation and conversion helpers).  
**Example:**
```python
tuples_by_level = convert_layer_combine_b(ds, beta=10.0)
((gun_d, out_b_d, comm_d), (gun_d1, comm_d1)) = tuples_by_level[0]
```

#### `convert_internal_model_a(ds: CanonicalPyrDataset, *, rep_field: TrainRep = "scaled", rep_comm_target: TrainRep = "hard_logit", rep_meas_target: TrainRep = "hard_logit", rep_out_target: TrainRep = "hard_logit", beta: float = 10.0) -> tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]`

**Signature:** `convert_internal_model_a(ds: CanonicalPyrDataset, *, rep_field: TrainRep = "scaled", rep_comm_target: TrainRep = "hard_logit", rep_meas_target: TrainRep = "hard_logit", rep_out_target: TrainRep = "hard_logit", beta: float = 10.0) -> tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]`  
**Purpose:** Build the supervised training view for internal model A, returning initial field state, the (final) communication target, and per-transition lists of A measurement-input and measurement-outcome targets.  
**Arguments:**  
- `ds`: Canonical dataset.  
- `rep_field`: Representation for the initial field state (level 0).  
- `rep_comm_target`: Representation for the comm target.  
- `rep_meas_target`: Representation for A measurement-input targets.  
- `rep_out_target`: Representation for A measurement-outcome targets.  
- `beta`: Logit magnitude (scalar only).  
**Returns:** Tuple `(field_0, comm_a_target, meas_targets_list, out_targets_list)` where `field_0: (N, n2)`, `comm_a_target: (N, 1)`, and each list element has shape `(N, k_d)` for transition `d`.  
**Errors:** Not specified (may raise from internal validation and conversion helpers).  
**Example:**
```python
field_0, comm_a_target, meas_list, out_list = convert_internal_model_a(ds, beta=10.0)
```

#### `convert_internal_model_b(ds: CanonicalPyrDataset, *, rep_gun: TrainRep = "scaled", rep_comm_in: TrainRep = "hard_logit", rep_prev_meas: TrainRep = "hard_logit", rep_prev_out: TrainRep = "hard_logit", rep_shoot_target: TrainRep = "bits", rep_meas_b_target: TrainRep = "bits", rep_out_b_target: TrainRep = "bits", beta: float = 10.0) -> tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray]`

**Signature:** `convert_internal_model_b(ds: CanonicalPyrDataset, *, rep_gun: TrainRep = "scaled", rep_comm_in: TrainRep = "hard_logit", rep_prev_meas: TrainRep = "hard_logit", rep_prev_out: TrainRep = "hard_logit", rep_shoot_target: TrainRep = "bits", rep_meas_b_target: TrainRep = "bits", rep_out_b_target: TrainRep = "bits", beta: float = 10.0) -> tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray]`  
**Purpose:** Build the supervised training view for internal model B, returning initial gun state, teacher comm input, lists of previous A-side measurements/outcomes, B-side measurement targets/outcome targets, and final shoot label.  
**Arguments:**  
- `ds`: Canonical dataset.  
- `rep_gun`: Representation for the initial gun state (level 0).  
- `rep_comm_in`: Representation for the teacher comm input at level 0.  
- `rep_prev_meas`: Representation for previous A-side measurement inputs.  
- `rep_prev_out`: Representation for previous A-side measurement outcomes.  
- `rep_shoot_target`: Representation for the final shoot label.  
- `rep_meas_b_target`: Representation for B measurement-input targets.  
- `rep_out_b_target`: Representation for B measurement-outcome targets.  
- `beta`: Logit magnitude (scalar only).  
**Returns:** Tuple `(gun_0, teacher_comm_0, prev_meas_list, prev_out_list, meas_b_list, out_b_list, shoot_target)` with shapes `gun_0: (N, n2)`, `teacher_comm_0: (N, 1)`, each list element `(N, k_d)`, and `shoot_target: (N, 1)`.  
**Errors:** Not specified (may raise from internal validation and conversion helpers).  
**Example:**
```python
gun_0, teacher_comm_0, prev_meas, prev_out, meas_b, out_b, shoot = convert_internal_model_b(ds, beta=10.0)
```

#### `convert_full_system(ds: CanonicalPyrDataset, *, rep_field: TrainRep = "scaled", rep_gun: TrainRep = "scaled", rep_teacher_comm_trace: TrainRep = "bits", rep_teacher_meas_a: TrainRep = "bits", rep_teacher_out_a: TrainRep = "bits", rep_shoot_target: TrainRep = "bits", beta: float = 10.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], np.ndarray]`

**Signature:** `convert_full_system(ds: CanonicalPyrDataset, *, rep_field: TrainRep = "scaled", rep_gun: TrainRep = "scaled", rep_teacher_comm_trace: TrainRep = "bits", rep_teacher_meas_a: TrainRep = "bits", rep_teacher_out_a: TrainRep = "bits", rep_shoot_target: TrainRep = "bits", beta: float = 10.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], np.ndarray]`  
**Purpose:** Build a full-system (A→B) training view with teacher traces: initial states for A and B, teacher comm trace, teacher A-side measurement/outcome traces, and shoot target.  
**Arguments:**  
- `ds`: Canonical dataset.  
- `rep_field`: Representation for the initial field state (level 0).  
- `rep_gun`: Representation for the initial gun state (level 0).  
- `rep_teacher_comm_trace`: Representation for the full comm trace (depth+1).  
- `rep_teacher_meas_a`: Representation for teacher A measurement inputs.  
- `rep_teacher_out_a`: Representation for teacher A measurement outcomes.  
- `rep_shoot_target`: Representation for the final shoot label.  
- `beta`: Logit magnitude (scalar only).  
**Returns:** Tuple `(field_0, gun_0, teacher_comms_trace, teacher_meas_a_list, teacher_out_a_list, shoot_target)` with shapes `field_0: (N, n2)`, `gun_0: (N, n2)`, `teacher_comms_trace: (N, depth+1, 1)`, list elements `(N, k_d)`, and `shoot_target: (N, 1)`.  
**Errors:** Not specified (may raise from internal validation and conversion helpers).  
**Example:**
```python
field_0, gun_0, comms, meas_a, out_a, shoot = convert_full_system(ds, beta=10.0)
```

#### `convert_all_traces(ds: CanonicalPyrDataset, *, rep_field: TrainRep = "bits", rep_gun: TrainRep = "bits", rep_comms: TrainRep = "bits", rep_meas_in_a: TrainRep = "bits", rep_meas_out_a: TrainRep = "bits", rep_meas_in_b: TrainRep = "bits", rep_meas_out_b: TrainRep = "bits", rep_shoot: TrainRep = "bits", beta: float | Sequence[float] = 10.0) -> dict[str, Any]`

**Signature:** `convert_all_traces(ds: CanonicalPyrDataset, *, rep_field: TrainRep = "bits", rep_gun: TrainRep = "bits", rep_comms: TrainRep = "bits", rep_meas_in_a: TrainRep = "bits", rep_meas_out_a: TrainRep = "bits", rep_meas_in_b: TrainRep = "bits", rep_meas_out_b: TrainRep = "bits", rep_shoot: TrainRep = "bits", beta: float | Sequence[float] = 10.0) -> dict[str, Any]`  
**Purpose:** Convert and return a complete per-level view of all canonical traces with pyramid-consistent cropping and per-trace representation selection; supports deterministic mixed-beta and maintains aligned ordering across all returned arrays.  
**Arguments:**  
- `ds`: Canonical dataset.  
- `rep_field`: Representation for the field state trace.  
- `rep_gun`: Representation for the gun state trace.  
- `rep_comms`: Representation for the comm trace.  
- `rep_meas_in_a`: Representation for A measurement-input trace.  
- `rep_meas_out_a`: Representation for A measurement-outcome trace.  
- `rep_meas_in_b`: Representation for B measurement-input trace.  
- `rep_meas_out_b`: Representation for B measurement-outcome trace.  
- `rep_shoot`: Representation for the final shoot label.  
- `beta`: Logit magnitude (scalar) or mixed-beta sequence.  
**Returns:** Dict containing keys `"N"`, `"depth"`, `"n2"`, `"L"`, `"k"`, `"field"`, `"gun"`, `"comms"`, `"meas_in_a"`, `"meas_out_a"`, `"meas_in_b"`, `"meas_out_b"`, `"shoot"` with shapes and list lengths as described in the function docstring.  
**Errors:** Not specified (may raise from internal validation and conversion helpers).  
**Example:**
```python
view = convert_all_traces(ds, rep_field="scaled", rep_comms="hard_logit", beta=[5.0, 10.0])
field_level0 = view["field"][0]   # (N, L_0)
shoot = view["shoot"]             # (N, 1)
```

### Constants

None.

### Types

#### `TrainRep`

**Definition:** `Literal["bits", "scaled", "hard_logit"]`  
**Purpose:** Enumerates allowed training representations accepted by converters.

#### `CanonicalPyrDataset`

**Definition:** `TypedDict` with required keys mapping to `np.ndarray`.  
**Purpose:** Typed view of the canonical pyramid dataset used by converters.  
**Keys (required):**  
- `field_bits`: State trace for player A's field. Shape `(N, depth+1, n2)`.  
- `gun_bits`: State trace for player B's gun. Shape `(N, depth+1, n2)`.  
- `comms_bits`: Communication trace. Shape `(N, depth+1, 1)`.  
- `meas_in_a_bits`: Measurement inputs for A at each transition. Shape `(N, depth, n2)`.  
- `meas_out_a_bits`: Measurement outcomes for A at each transition. Shape `(N, depth, n2)`.  
- `meas_in_b_bits`: Measurement inputs for B at each transition. Shape `(N, depth, n2)`.  
- `meas_out_b_bits`: Measurement outcomes for B at each transition. Shape `(N, depth, n2)`.  
- `shoot`: Final shoot/decision label. Shape `(N, 1)`.

## Dependencies

- `numpy` (imported as `np`)
- `typing` (`Literal`, `TypedDict`, `Dict`, `Tuple`, `List`, `Sequence`, `Any`, `Union`)

## Planned (design-spec)

Not specified.

## Deviations

- Several helper functions exist in the module but are not documented here because they are private by naming convention (leading underscore): `_as_float32`, `_require_keys`, `_validate_basic_shapes`, `_crop_field_like`, `_crop_meas_like`, `_normalize_betas`, `_contiguous_splits`, `_interleave_round_robin`.
- `convert_internal_model_b` contains a docstring note stating: "The return type annotation in the signature is narrower than the actual returned tuple." The signature in the provided text returns a 7-tuple and the implementation returns 7 values; any further mismatch details are not specified.

## Notes for Contributors

- Maintain determinism for mixed-beta mode: any refactor must preserve contiguous splitting and round-robin interleaving semantics along sample axis 0.
- Preserve converter signatures labeled as "frozen signatures" in the source comments.
- Keep cropping rules consistent with pyramid geometry: state traces use `L_d`, transition traces use `k_d`.

## Related

- `converters_spec.md` (mentioned in module docstring; location not specified)

## Changelog

Unknown.