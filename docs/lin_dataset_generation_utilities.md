# QSeaBattle.lin_dataset_generation_utilities

> Role: Generate and validate canonical linear (depth=1) per-game and per-dataset teacher traces for QSeaBattle, using binary float32 NumPy arrays.

Location: `Q_Sea_Battle.lin_dataset_generation_utilities`

## Overview

This module generates and validates the canonical storage format for linear teacher traces (single-step / depth=1). Each generated game is represented as binary float32 arrays (logical bits encoded as `{0.0, 1.0}`) describing hidden field bits, a one-hot gun position, shared communication bits, per-player measurement inputs/outputs, and the resulting `shoot` label (hit/miss bit at the gun index).

Canonical shapes (single game vs. dataset) are as described in the module docstring: per-game arrays have no leading batch dimension, and dataset arrays have a leading `N` dimension for `N` games.

## Public API

### Functions

#### `lin_sizes(n2: int, m: int) -> LinSizes`

**Signature:** `lin_sizes(n2: int, m: int) -> LinSizes`  
**Purpose:** Create a validated `LinSizes` instance for linear (depth=1) traces.  
**Arguments:** `n2` (int): Number of field/gun bits; must be `>= 1`. `m` (int): Number of communication bits; must be `>= 1`.  
**Returns:** `LinSizes`: A sizes bundle with `depth=1`.  
**Errors:** `ValueError` if `n2 < 1` or `m < 1`.  
**Example:**
```python
from Q_Sea_Battle.lin_dataset_generation_utilities import lin_sizes

s = lin_sizes(n2=16, m=4)
assert s.n2 == 16 and s.m == 4 and s.depth == 1
```

#### `generate_one_game_trace_lin(n2: int, m: int, *, seed: int = 0, validate: bool = True) -> Dict[str, np.ndarray]`

**Signature:** `generate_one_game_trace_lin(n2: int, m: int, *, seed: int = 0, validate: bool = True) -> Dict[str, np.ndarray]`  
**Purpose:** Generate a single linear (depth=1) teacher trace as a dict of binary float32 NumPy arrays with canonical per-game shapes.  
**Arguments:** `n2` (int): Number of field/gun bits. `m` (int): Number of communication bits. `seed` (int): RNG seed for this game. `validate` (bool): If `True`, run `validate_one_game_trace_lin` on the generated trace.  
**Returns:** `Dict[str, np.ndarray]`: A dict containing keys `field_bits`, `gun_bits`, `comms_bits`, `meas_in_a_bits`, `meas_out_a_bits`, `meas_in_b_bits`, `meas_out_b_bits`, `shoot`, each with the per-game shapes documented in the module docstring.  
**Errors:** `ValueError` if validation is enabled and the generated trace fails validation. `ValueError` may also be raised by `lin_sizes` if `n2 < 1` or `m < 1`.  
**Example:**
```python
import numpy as np
from Q_Sea_Battle.lin_dataset_generation_utilities import generate_one_game_trace_lin

g = generate_one_game_trace_lin(n2=8, m=3, seed=123, validate=True)
assert g["field_bits"].shape == (2, 8)
assert g["gun_bits"].shape == (2, 8)
assert g["comms_bits"].shape == (2, 3)
assert g["meas_in_a_bits"].shape == (1, 8)
assert g["meas_out_a_bits"].shape == (1, 8)
assert g["meas_in_b_bits"].shape == (1, 8)
assert g["meas_out_b_bits"].shape == (1, 8)
assert g["shoot"].shape == (1,)
assert g["field_bits"].dtype == np.float32
```

#### `validate_one_game_trace_lin(n2: int, m: int, *, field_bits: np.ndarray, gun_bits: np.ndarray, comms_bits: np.ndarray, meas_in_a_bits: np.ndarray, meas_out_a_bits: np.ndarray, meas_in_b_bits: np.ndarray, meas_out_b_bits: np.ndarray, shoot: np.ndarray) -> None`

**Signature:** `validate_one_game_trace_lin(n2: int, m: int, *, field_bits: np.ndarray, gun_bits: np.ndarray, comms_bits: np.ndarray, meas_in_a_bits: np.ndarray, meas_out_a_bits: np.ndarray, meas_in_b_bits: np.ndarray, meas_out_b_bits: np.ndarray, shoot: np.ndarray) -> None`  
**Purpose:** Validate shapes and bit encodings for a single linear (depth=1) trace, including one-hot gun encoding for both players.  
**Arguments:** `n2` (int): Number of field/gun bits. `m` (int): Number of communication bits. `field_bits` (`np.ndarray`): Shape `(2, n2)`. `gun_bits` (`np.ndarray`): Shape `(2, n2)`; each row must be one-hot. `comms_bits` (`np.ndarray`): Shape `(2, m)`. `meas_in_a_bits` (`np.ndarray`): Shape `(1, n2)`. `meas_out_a_bits` (`np.ndarray`): Shape `(1, n2)`. `meas_in_b_bits` (`np.ndarray`): Shape `(1, n2)`. `meas_out_b_bits` (`np.ndarray`): Shape `(1, n2)`. `shoot` (`np.ndarray`): Shape `(1,)`.  
**Returns:** `None`.  
**Errors:** `ValueError` if any shape constraint is violated, if any array contains non-binary values (not in `{0.0, 1.0}`), or if either player’s gun vector is not one-hot.  
**Example:**
```python
from Q_Sea_Battle.lin_dataset_generation_utilities import generate_one_game_trace_lin, validate_one_game_trace_lin

g = generate_one_game_trace_lin(n2=8, m=3, seed=0, validate=False)
validate_one_game_trace_lin(8, 3, **g)
```

#### `generate_lin_dataset(n2: int, m: int, num_games: int, *, seed: int = 0, validate: bool = True) -> Dict[str, np.ndarray]`

**Signature:** `generate_lin_dataset(n2: int, m: int, num_games: int, *, seed: int = 0, validate: bool = True) -> Dict[str, np.ndarray]`  
**Purpose:** Generate a dataset of independent linear (depth=1) game traces as stacked arrays with a leading batch dimension `N`.  
**Arguments:** `n2` (int): Number of field/gun bits. `m` (int): Number of communication bits. `num_games` (int): Number of games to generate; must be `> 0`. `seed` (int): Base RNG seed; game `i` uses seed `seed + i`. `validate` (bool): If `True`, validate each per-game trace during generation.  
**Returns:** `Dict[str, np.ndarray]`: A dict containing keys `field_bits`, `gun_bits`, `comms_bits`, `meas_in_a_bits`, `meas_out_a_bits`, `meas_in_b_bits`, `meas_out_b_bits`, `shoot`, each with the dataset shapes documented in the module docstring.  
**Errors:** `ValueError` if `num_games <= 0`, or if validation fails for any generated game. `ValueError` may also be raised by `lin_sizes` if `n2 < 1` or `m < 1`.  
**Example:**
```python
from Q_Sea_Battle.lin_dataset_generation_utilities import generate_lin_dataset

ds = generate_lin_dataset(n2=8, m=3, num_games=10, seed=100, validate=True)
assert ds["field_bits"].shape == (10, 2, 8)
assert ds["shoot"].shape == (10, 1)
```

#### `save_npz(path: str, ds: Dict[str, np.ndarray]) -> None`

**Signature:** `save_npz(path: str, ds: Dict[str, np.ndarray]) -> None`  
**Purpose:** Save a generated dataset dict as a compressed `.npz` file.  
**Arguments:** `path` (str): Output file path. `ds` (`Dict[str, np.ndarray]`): Dataset dict mapping names to NumPy arrays (for example, the output of `generate_lin_dataset`).  
**Returns:** `None`.  
**Errors:** Not specified (NumPy I/O may raise exceptions depending on filesystem and input).  
**Example:**
```python
from Q_Sea_Battle.lin_dataset_generation_utilities import generate_lin_dataset, save_npz

ds = generate_lin_dataset(n2=8, m=3, num_games=5, seed=0)
save_npz("lin_ds.npz", ds)
```

### Constants

None.

### Types

#### `LinSizes`

**Kind:** `@dataclass(frozen=True)`  
**Purpose:** Dimension bundle for linear (depth=1) traces.  
**Fields:** `n2` (int): Number of field/gun bits. `m` (int): Number of communication bits. `depth` (int): Trace depth; fixed to `1` in this module (default `1`).  
**Example:**
```python
from Q_Sea_Battle.lin_dataset_generation_utilities import LinSizes

s = LinSizes(n2=8, m=3)
assert s.depth == 1
```

## Dependencies

- Python standard library: `dataclasses.dataclass`, `typing.Dict`
- Third-party: `numpy` (imported as `np`)

## Planned (design-spec)

Not specified.

## Deviations

Not specified.

## Notes for Contributors

- Only the functions and type listed in the Public API are intended for external use; helper functions prefixed with `_` are internal and may change without notice.  
- All bit arrays are validated as binary `{0.0, 1.0}` values; the module does not enforce dtype beyond using float32 in generation.  
- Canonical shapes are strictly enforced by `validate_one_game_trace_lin`; update both the docstring and validator together if the format changes.

## Related

- NumPy `.npz` format via `numpy.savez_compressed` (used by `save_npz`)  
- This module’s canonical shapes (documented in the module docstring) define the expected interchange format for downstream consumers.

## Changelog

- Not specified.