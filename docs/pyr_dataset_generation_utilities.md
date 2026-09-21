# QSeaBattle: Pyramid per-game dataset generation utilities
> Role: Generate per-game and stacked pyramid-task datasets (binary float32 traces) using NumPy-only “teacher” logic, including validation and NPZ persistence.
Location: `Q_Sea_Battle.pyr_dataset_generation_utilities`

## Overview

This module generates *per-game* traces (one full game per sample) and stacked datasets for the QSeaBattle “pyramid” task in a canonical on-disk format. All arrays are `float32` with binary values `{0.0, 1.0}`, are dense, and are right-padded with zeros to a fixed width `n2` (which must be a power of two). Pyramid sizes are derived from `n2` such that for level `d` (0-indexed): `L[d] = n2 / 2**d` is the meaningful prefix length for `field`/`gun` state vectors, and `k[d] = L[d] / 2` is the meaningful prefix length for measurement inputs/outcomes at that level.

Canonical stacked dataset shapes for `N` games: `field_bits (N, depth+1, n2)`, `gun_bits (N, depth+1, n2)`, `comms_bits (N, depth+1, 1)`, `meas_in_a_bits (N, depth, n2)`, `meas_out_a_bits (N, depth, n2)`, `meas_in_b_bits (N, depth, n2)`, `meas_out_b_bits (N, depth, n2)`, `shoot (N, 1)`.

## Public API

### Functions

#### `teacher_measure_a(field: np.ndarray) -> np.ndarray`

**Signature:** `teacher_measure_a(field: np.ndarray) -> np.ndarray`  
**Purpose:** Compute A-side measurement inputs from the current field prefix as pairwise XOR between even- and odd-indexed elements.  
**Arguments:** `field` (np.ndarray): 1D float array containing binary values `{0,1}` in its meaningful prefix.  
**Returns:** 1D `float32` binary array of length `len(field) / 2`.  
**Errors:** Not specified (no explicit validation; relies on NumPy operations and input shape/contents).  
**Example:**
```python
import numpy as np
from Q_Sea_Battle.pyr_dataset_generation_utilities import teacher_measure_a

field_prefix = np.array([1, 0, 1, 1], dtype=np.float32)
meas_in_a = teacher_measure_a(field_prefix)  # length 2
```

#### `teacher_combine_a(field: np.ndarray, sr_outcome: np.ndarray) -> np.ndarray`

**Signature:** `teacher_combine_a(field: np.ndarray, sr_outcome: np.ndarray) -> np.ndarray`  
**Purpose:** Compute the next reduced field prefix from the A-side SR outcome: `next_field = even(field) XOR sr_outcome`.  
**Arguments:** `field` (np.ndarray): 1D float array for the current meaningful prefix; `sr_outcome` (np.ndarray): 1D float array of binary outcomes aligned with `even(field)`.  
**Returns:** 1D `float32` binary array of length `len(field) / 2`.  
**Errors:** Not specified (no explicit validation).  
**Example:**
```python
import numpy as np
from Q_Sea_Battle.pyr_dataset_generation_utilities import teacher_combine_a

field_prefix = np.array([1, 0, 1, 1], dtype=np.float32)     # even = [1, 1]
out_a = np.array([0, 1], dtype=np.float32)
next_field = teacher_combine_a(field_prefix, out_a)          # [1^0, 1^1] -> [1, 0]
```

#### `teacher_measure_b(gun: np.ndarray) -> np.ndarray`

**Signature:** `teacher_measure_b(gun: np.ndarray) -> np.ndarray`  
**Purpose:** Compute B-side measurement inputs from the current gun prefix as `(NOT even(gun)) AND odd(gun)`.  
**Arguments:** `gun` (np.ndarray): 1D float array containing a one-hot vector in its meaningful prefix.  
**Returns:** 1D `float32` binary array of length `len(gun) / 2`.  
**Errors:** Not specified (no explicit validation).  
**Example:**
```python
import numpy as np
from Q_Sea_Battle.pyr_dataset_generation_utilities import teacher_measure_b

gun_prefix = np.array([0, 1, 0, 0], dtype=np.float32)  # one-hot at index 1
meas_in_b = teacher_measure_b(gun_prefix)              # length 2
```

#### `teacher_combine_b(gun: np.ndarray, sr_outcome: np.ndarray, comm: np.ndarray) -> tuple[np.ndarray, np.ndarray]`

**Signature:** `teacher_combine_b(gun: np.ndarray, sr_outcome: np.ndarray, comm: np.ndarray) -> tuple[np.ndarray, np.ndarray]`  
**Purpose:** Reduce the gun prefix and update the comm bit: `next_gun = XOR(even(gun), odd(gun))`; `next_comm = comm XOR sr_outcome[argmax(next_gun)]`.  
**Arguments:** `gun` (np.ndarray): 1D float array containing a one-hot vector in its meaningful prefix; `sr_outcome` (np.ndarray): 1D float array of binary outcomes aligned with `even(gun)`; `comm` (np.ndarray): `float32` array of shape `(1,)` representing the current comm bit.  
**Returns:** Tuple `(next_gun, next_comm)` where `next_gun` is a 1D `float32` binary vector of length `len(gun) / 2` and `next_comm` is a `float32` array of shape `(1,)`.  
**Errors:** Not specified (no explicit validation; may error if shapes are incompatible).  
**Example:**
```python
import numpy as np
from Q_Sea_Battle.pyr_dataset_generation_utilities import teacher_combine_b

gun_prefix = np.array([0, 1, 0, 0], dtype=np.float32)
out_b = np.array([1, 0], dtype=np.float32)
comm = np.array([0.0], dtype=np.float32)

next_gun, next_comm = teacher_combine_b(gun_prefix, out_b, comm)
```

#### `pyr_sizes(n2: int) -> PyrSizes`

**Signature:** `pyr_sizes(n2: int) -> PyrSizes`  
**Purpose:** Derive pyramid sizing metadata from padded width `n2` (power of two), including reduction depth and per-level meaningful prefix lengths.  
**Arguments:** `n2` (int): Total padded width; must be power of two and `>= 2`.  
**Returns:** `PyrSizes` instance with fields `n2`, `depth`, `L`, and `k`.  
**Errors:** `ValueError` if `n2 < 2`, `n2` is not a power of two, or the final reduction is invalid (expects `k[-1] == 1`).  
**Example:**
```python
from Q_Sea_Battle.pyr_dataset_generation_utilities import pyr_sizes

s = pyr_sizes(8)
# s.depth == 3
# s.L == (8, 4, 2)
# s.k == (4, 2, 1)
```

#### `generate_one_game_trace_pyr(n2: int, *, seed: int = 0, validate: bool = True) -> Dict[str, np.ndarray]`

**Signature:** `generate_one_game_trace_pyr(n2: int, *, seed: int = 0, validate: bool = True) -> Dict[str, np.ndarray]`  
**Purpose:** Generate one complete per-game pyramid trace, including initial field/gun state, per-level measurement inputs/outcomes for both sides, comm trace, and a `shoot` label (one-shot hit).  
**Arguments:** `n2` (int): Total padded width (power of two, `>= 2`); `seed` (int): RNG seed; `validate` (bool): if True, validate output via `validate_one_game_trace`.  
**Returns:** Dict of unstacked arrays: `field_bits (depth+1, n2)`, `gun_bits (depth+1, n2)`, `comms_bits (depth+1, 1)`, `meas_in_a_bits (depth, n2)`, `meas_out_a_bits (depth, n2)`, `meas_in_b_bits (depth, n2)`, `meas_out_b_bits (depth, n2)`, `shoot (1,)`.  
**Errors:** `ValueError` on sizing issues, unexpected teacher function output shapes, or validation failures (when `validate=True`).  
**Example:**
```python
from Q_Sea_Battle.pyr_dataset_generation_utilities import generate_one_game_trace_pyr

trace = generate_one_game_trace_pyr(16, seed=123, validate=True)
shoot = trace["shoot"]          # shape (1,)
field_bits = trace["field_bits"]  # shape (depth+1, 16)
```

#### `validate_one_game_trace(n2: int, *, field_bits: np.ndarray, gun_bits: np.ndarray, comms_bits: np.ndarray, meas_in_a_bits: np.ndarray, meas_out_a_bits: np.ndarray, meas_in_b_bits: np.ndarray, meas_out_b_bits: np.ndarray, shoot: np.ndarray) -> None`

**Signature:** `validate_one_game_trace(n2: int, *, field_bits: np.ndarray, gun_bits: np.ndarray, comms_bits: np.ndarray, meas_in_a_bits: np.ndarray, meas_out_a_bits: np.ndarray, meas_in_b_bits: np.ndarray, meas_out_b_bits: np.ndarray, shoot: np.ndarray) -> None`  
**Purpose:** Validate a single-game trace against core invariants: canonical shapes, binary domains, per-level gun one-hot constraint, and strict zero padding beyond each level’s meaningful prefix.  
**Arguments:** `n2` (int): Total padded width; `field_bits`, `gun_bits`, `comms_bits`, `meas_in_a_bits`, `meas_out_a_bits`, `meas_in_b_bits`, `meas_out_b_bits`, `shoot`: arrays in the canonical per-game layout.  
**Returns:** `None`.  
**Errors:** `ValueError` if any invariant is violated (shape mismatch, non-binary values, gun not one-hot, or non-zero padding).  
**Example:**
```python
from Q_Sea_Battle.pyr_dataset_generation_utilities import generate_one_game_trace_pyr, validate_one_game_trace

trace = generate_one_game_trace_pyr(8, seed=0, validate=False)
validate_one_game_trace(8, **trace)
```

#### `generate_pyr_dataset(n2: int, num_games: int, *, seed: int = 0, validate: bool = True) -> Dict[str, np.ndarray]`

**Signature:** `generate_pyr_dataset(n2: int, num_games: int, *, seed: int = 0, validate: bool = True) -> Dict[str, np.ndarray]`  
**Purpose:** Generate a stacked dataset of `num_games` pyramid games, using `seed + i` per game.  
**Arguments:** `n2` (int): Total padded width; `num_games` (int): number of games/samples (`> 0`); `seed` (int): base seed; `validate` (bool): validate each generated trace.  
**Returns:** Dict of stacked arrays with leading dimension `N == num_games`: `field_bits`, `gun_bits`, `comms_bits`, `meas_in_a_bits`, `meas_out_a_bits`, `meas_in_b_bits`, `meas_out_b_bits`, `shoot`.  
**Errors:** `ValueError` if `num_games <= 0` or if per-game generation/validation fails.  
**Example:**
```python
from Q_Sea_Battle.pyr_dataset_generation_utilities import generate_pyr_dataset

ds = generate_pyr_dataset(16, num_games=100, seed=0, validate=True)
print(ds["field_bits"].shape)  # (100, depth+1, 16)
```

#### `save_npz(path: str, ds: Dict[str, np.ndarray]) -> None`

**Signature:** `save_npz(path: str, ds: Dict[str, np.ndarray]) -> None`  
**Purpose:** Save a generated dataset dict to a compressed `.npz` file via `numpy.savez_compressed`.  
**Arguments:** `path` (str): output file path; `ds` (Dict[str, np.ndarray]): dataset dict (e.g., from `generate_pyr_dataset`).  
**Returns:** `None`.  
**Errors:** Not specified (I/O errors may be raised by NumPy/Python runtime).  
**Example:**
```python
from Q_Sea_Battle.pyr_dataset_generation_utilities import generate_pyr_dataset, save_npz

ds = generate_pyr_dataset(8, num_games=10, seed=42)
save_npz("pyr_8_n10.npz", ds)
```

### Constants

Not specified.

### Types

#### `PyrSizes`

**Kind:** `@dataclass(frozen=True)`  
**Purpose:** Container for derived pyramid sizes for a given `n2`.  
**Fields:** `n2` (int): total padded width; `depth` (int): number of reduction steps (`log2(n2)`); `L` (Tuple[int, ...]): per-level meaningful prefix widths for field/gun state vectors (length `depth`, `L[d] = n2 / 2**d`); `k` (Tuple[int, ...]): per-level meaningful prefix widths for measurement vectors (length `depth`, `k[d] = L[d] / 2`).  

## Dependencies

- `numpy` (as `np`)
- Python standard library: `dataclasses.dataclass`, `typing.Dict`, `typing.Tuple`
- `from __future__ import annotations`

## Planned (design-spec)

Unknown (no design notes provided).

## Deviations

- The module docstring notes that where external specifications disagree with behavior, the “teacher” logic implemented here is authoritative; external deviations are not specified in the provided text.

## Notes for Contributors

- Keep this module dependency-light (NumPy only) to support standalone generation scripts.  
- Preserve binary float32 conventions `{0.0, 1.0}` and strict right-padding with zeros; `validate_one_game_trace` encodes critical invariants for on-disk compatibility.  
- When modifying teacher logic, ensure that per-level shapes match `k[d]` and that reduction consistency holds (`pyr_sizes` expects `k[-1] == 1`).  

## Related

- NumPy documentation for random generation (`numpy.random.default_rng`) and saving NPZ files (`numpy.savez_compressed`).

## Changelog

Unknown (not specified).