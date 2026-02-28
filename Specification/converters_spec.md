# Dataset Converters Specification (Updated — Logit-State Default, Logit Targets)

This document defines the **frozen, normative interfaces** for converting the **canonical**
QSeaBattle pyramid dataset into **training-ready** views.

It is updated for the **logit-state internal model** approach:

- Internal models pass **state as logits between levels** (field/gun wires are logits after level 0).
- The canonical dataset remains **bit storage**.
- Converters therefore default to generating **logit-style** inputs **and targets** (`hard_logit`) so that
  supervised training matches runtime/RL distributions as closely as possible.

The converters are **pure utilities**:
- no model dependencies
- no randomness / no RNG
- deterministic mapping from canonical dataset → training arrays

> **All per-level vectors returned by converters are CROPPED to the active width (`L_d` or `k_d`).**

---

## 1. Canonical Dataset

All converters consume the same canonical dataset (storage format).

```python
class CanonicalPyrDataset(TypedDict):
    field_bits: np.ndarray        # (N, depth+1, n2)
    gun_bits: np.ndarray          # (N, depth+1, n2)
    comms_bits: np.ndarray        # (N, depth+1, 1)
    meas_in_a_bits: np.ndarray    # (N, depth,   n2)
    meas_out_a_bits: np.ndarray   # (N, depth,   n2)
    meas_in_b_bits: np.ndarray    # (N, depth,   n2)
    meas_out_b_bits: np.ndarray   # (N, depth,   n2)
    shoot: np.ndarray             # (N, 1)
```

### 1.1 Invariants (canonical storage)
- dtype: `float32`
- values: `{0.0, 1.0}`
- right padding beyond the active prefix is **zero**
- `depth = ds["meas_in_a_bits"].shape[1]`
- state traces (`field_bits`, `gun_bits`, `comms_bits`) have length `depth+1`
- transition traces (`meas_*_bits`) have length `depth`

---

## 2. Representation Modes

```python
TrainRep = Literal["bits", "scaled", "hard_logit"]
```

| Mode | Mapping |
|---|---|
| `bits` | keep `{0,1}` |
| `scaled` | `x → x - 0.5` (so values are `{-0.5, +0.5}`) |
| `hard_logit` | `x → beta * (2x - 1)` (so values are `{−beta, +beta}`) |

---

## 3. Beta Parameter

Converters accept:

- `beta: float | Sequence[float]` with default `10.0`.

Normalization (normative):
- If `beta` is a scalar: `betas = [float(beta)]`
- If `beta` is a sequence: `betas = [float(b) for b in beta]`
- If `beta` is a sequence and `len(betas) == 0`: **raise `ValueError`**
- If `len(betas) == 1`: behavior must be **bitwise identical** to the scalar case

### 3.1 When beta affects values
- `beta` affects numeric values **only** when a field is represented as `hard_logit`.
- For `bits` and `scaled`, numeric values are independent of `beta`.

### 3.2 Mixed-beta dataset views (normative)
If `len(betas) = K > 1`, converters must generate a **mixed hardness** view as follows:

1) **Deterministic contiguous split** of the sample axis:
   - Let `N = number of samples`.
   - Split indices `[0..N-1]` into `K` contiguous chunks with sizes differing by at most 1.
   - Equivalent to: `np.array_split(np.arange(N), K)`.

2) **Per-chunk conversion**:
   - For chunk `j`, apply conversion with `beta = betas[j]`.
   - This affects only those tensors whose `rep == "hard_logit"`.

3) **Deterministic interleaving (final order)**:
   - Concatenate outputs by **round-robin interleaving** across chunks:
     - emit element 0 of chunk 0, element 0 of chunk 1, …, element 0 of chunk K-1 (if present),
     - then element 1 of chunk 0, …,
     - skipping exhausted chunks,
     - until all `N` elements are emitted.
   - Apply the same interleaving consistently to **every array** returned for that converter and level `d`.

**Note:** When `K>1`, the final output order differs from the original dataset order due to interleaving,
even for tensors represented as `bits` or `scaled` (values unchanged, row order interleaved).

---

## 4. Pyramid Level Geometry (Frozen)

Let `n2 = ds["field_bits"].shape[2]` and `depth = log2(n2)`.

For level `d ∈ [0 .. depth]`:

- Active state width: `L_d = n2 / 2^d`
- Active measurement width (transitions `d ∈ [0 .. depth-1]`): `k_d = n2 / 2^(d+1)`

### 4.1 Cropping rule (normative)
- Field/Gun at level `d`: `v[:L_d]`
- Measurement in/out at transition `d`: `v[:k_d]`
- Comms are `(N,1)` and never cropped

---

## 5. Converter Indexing Rule (Frozen)

- Layer/transition converters produce keys `d ∈ [0 .. depth-1]`.
- State access may use indices `0 .. depth`.
- No converter produces a training sample for `d = depth`.

---

## 6. Logit-State Defaults (Normative)

For the logit-state internal model approach, converters default to:

- **Inputs** are `hard_logit` where they represent internal wires (state, comm, outcomes, measurements).
- **Targets** are also `hard_logit` to match “everything is logits” inside internal models.
- `bits` remains available for label-style training where desired, but is not the default here.

`scaled` remains available as a **legacy** representation for older scaled-state internal wiring.

---

## 7. Converter — Measure A Layer (Updated defaults)

```python
def convert_layer_measure_a(
    ds: CanonicalPyrDataset,
    *,
    rep_x: TrainRep = "hard_logit",
    rep_y: TrainRep = "hard_logit",
    beta: float | Sequence[float] = 10.0,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
```

Output: `{ d: (X, Y) }` for `d ∈ [0 .. depth-1]`

- `X`: `(N, L_d)` from `field_bits[:, d, :L_d]`
- `Y`: `(N, k_d)` from `meas_in_a_bits[:, d, :k_d]`

---

## 8. Converter — Combine A Layer (Updated defaults)

```python
def convert_layer_combine_a(
    ds: CanonicalPyrDataset,
    *,
    rep_field: TrainRep = "hard_logit",
    rep_outcome: TrainRep = "hard_logit",
    rep_target: TrainRep = "hard_logit",
    beta: float | Sequence[float] = 10.0,
) -> dict[int, tuple[tuple[np.ndarray, np.ndarray], np.ndarray]]:
```

Output: `{ d: ((field_d, out_a_d), field_d1) }`

---

## 9. Converter — Measure B Layer (Updated defaults)

```python
def convert_layer_measure_b(
    ds: CanonicalPyrDataset,
    *,
    rep_x: TrainRep = "hard_logit",
    rep_y: TrainRep = "hard_logit",
    beta: float | Sequence[float] = 10.0,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
```

---

## 10. Converter — Combine B Layer (Updated defaults)

```python
def convert_layer_combine_b(
    ds: CanonicalPyrDataset,
    *,
    rep_gun: TrainRep = "hard_logit",
    rep_outcome_b: TrainRep = "hard_logit",
    rep_comm_in: TrainRep = "hard_logit",
    rep_gun_next: TrainRep = "hard_logit",
    rep_comm_next: TrainRep = "hard_logit",
    beta: float | Sequence[float] = 10.0,
) -> dict[int, tuple[tuple[np.ndarray, np.ndarray, np.ndarray],
                     tuple[np.ndarray, np.ndarray]]]:
```

---

## 11–13. Internal-model and full-system converters

All converters that accept `beta` must accept:

- `beta: float | Sequence[float] = 10.0`

and must follow §3.2 when `len(betas)>1`.

---

## 14. Helper Functions (Frozen)

```python
def infer_depth_from_dataset(ds: CanonicalPyrDataset) -> int:
    """Returns depth = ds["meas_in_a_bits"].shape[1]."""

def level_sizes(n2: int, d: int) -> tuple[int, int]:
    """Returns (L_d, k_d) with L_d = n2 // (2**d), k_d = L_d // 2."""
```

---

**Status:** Frozen (Updated defaults for logit-state internal models; mixed-beta behavior is normative)
