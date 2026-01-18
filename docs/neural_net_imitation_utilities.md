# Q_Sea_Battle.neural_net_imitation_utilities

> Role: Utilities to generate synthetic imitation-learning datasets for NeuralNetPlayers Model A (field -> comm) and Model B (comm + gun -> shoot) using a segment-wise majority teacher strategy.

Location: `Q_Sea_Battle.neural_net_imitation_utilities`

## Overview

This module provides helper functions to (1) partition a flattened game field into contiguous segments, (2) compute majority-based communication bits per segment, and (3) generate pandas DataFrames suitable for imitation training of two neural network models: Model A learns communication from fields, and Model B learns a shoot decision from communication plus a one-hot gun position. Data is synthesized by sampling IID Bernoulli fields and using a deterministic majority teacher policy based on the segment partitioning.

## Public API

### Functions

#### `make_segments(layout: GameLayout) -> List[Tuple[int, int]]`

**Purpose:** Partition the flattened field `[0, n2)` into `m = layout.comms_size` contiguous, near-even segments, returned as Python slice-style `(start, end)` index pairs.

**Arguments:**
- `layout`: `GameLayout` instance providing `field_size` and `comms_size`.

**Returns:**
- `List[Tuple[int, int]]`: List of `(start, end)` pairs (end exclusive), length `layout.comms_size`, covering `[0, n2)` without gaps or overlaps.

**Errors:**
- `ValueError`: If `field_size < 1`, or `comms_size < 1`, or `comms_size > n2`.
- `RuntimeError`: If the constructed segments do not cover the full field (internal safety check).

**Example:**
```python
from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.neural_net_imitation_utilities import make_segments

layout = GameLayout(field_size=4, comms_size=3)
segments = make_segments(layout)
# segments is a list of 3 (start, end) pairs covering indices [0, 16)
```

#### `compute_majority_comm(fields: np.ndarray, layout: GameLayout) -> np.ndarray`

**Purpose:** Compute segment-wise majority communication bits for a batch of flattened binary fields using the segmentation from `make_segments`.

**Arguments:**
- `fields`: NumPy array of shape `(N, n2)` containing flattened fields with values in `{0, 1}`.
- `layout`: `GameLayout` defining `field_size` and `comms_size`.

**Returns:**
- `np.ndarray`: Array of shape `(N, m)` with values in `{0.0, 1.0}`, dtype `float32`, where `m = layout.comms_size`.

**Errors:**
- `ValueError`: If `fields` is not 2D, or if `fields.shape[1] != layout.field_size**2`.

**Example:**
```python
import numpy as np
from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.neural_net_imitation_utilities import compute_majority_comm

layout = GameLayout(field_size=3, comms_size=3)
fields = np.array([[0,1,0, 1,1,0, 0,0,1]], dtype=np.float32)  # shape (1, 9)
comm = compute_majority_comm(fields, layout)  # shape (1, 3)
```

#### `generate_majority_dataset_model_a(layout: GameLayout, num_samples: int, p_one: float = 0.5, seed: Optional[int] = None) -> pd.DataFrame`

**Purpose:** Generate an imitation-learning dataset for Model A mapping `field -> comm`, where fields are IID Bernoulli and `comm` is the segment-wise majority teacher output.

**Arguments:**
- `layout`: `GameLayout` defining `field_size` and `comms_size`.
- `num_samples`: Number of samples to generate (must be positive).
- `p_one`: Bernoulli probability that a field cell equals `1`.
- `seed`: Optional RNG seed for reproducible sampling.

**Returns:**
- `pd.DataFrame`: DataFrame with at least the following columns (stored as per-row NumPy arrays):
- `field`: 1D `np.ndarray` of shape `(n2,)`, dtype `float32`.
- `comm`: 1D `np.ndarray` of shape `(m,)`, dtype `float32`.
- Additional columns may be added in the future (not specified).

**Errors:**
- `ValueError`: If `num_samples <= 0`.
- Any errors raised by `compute_majority_comm` due to inconsistent `layout` and sampled field shape (not explicitly rewrapped).

**Example:**
```python
from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.neural_net_imitation_utilities import generate_majority_dataset_model_a

layout = GameLayout(field_size=5, comms_size=5)
df_a = generate_majority_dataset_model_a(layout, num_samples=1000, p_one=0.4, seed=123)
x0 = df_a.loc[0, "field"]  # np.ndarray shape (25,)
y0 = df_a.loc[0, "comm"]   # np.ndarray shape (5,)
```

#### `generate_majority_dataset_model_b(layout: GameLayout, num_samples: int, p_one: float = 0.5, seed: Optional[int] = None) -> pd.DataFrame`

**Purpose:** Generate an imitation-learning dataset for Model B mapping `(comm + gun) -> shoot`, where `gun` is a one-hot cell index and `shoot` is the majority comm bit for the segment containing the gun index.

**Arguments:**
- `layout`: `GameLayout` defining `field_size` and `comms_size`.
- `num_samples`: Number of samples to generate (must be positive).
- `p_one`: Bernoulli probability that a field cell equals `1`.
- `seed`: Optional RNG seed for reproducible sampling.

**Returns:**
- `pd.DataFrame`: DataFrame with at least the following columns:
- `field`: 1D `np.ndarray` of shape `(n2,)`, dtype `float32`.
- `comm`: 1D `np.ndarray` of shape `(m,)`, dtype `float32`.
- `gun`: 1D `np.ndarray` of shape `(n2,)`, one-hot, dtype `float32`.
- `shoot`: scalar `np.float32` in `{0.0, 1.0}`.
- The docstring states this schema matches what `NeuralNetPlayers.train_model_b` expects (the referenced symbol is not defined in this module).

**Errors:**
- `ValueError`: If `num_samples <= 0`.
- Any errors raised by `make_segments` or `compute_majority_comm` for invalid layout or shape (not explicitly rewrapped).

**Example:**
```python
from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.neural_net_imitation_utilities import generate_majority_dataset_model_b

layout = GameLayout(field_size=4, comms_size=4)
df_b = generate_majority_dataset_model_b(layout, num_samples=500, p_one=0.5, seed=7)
gun0 = df_b.loc[0, "gun"]      # one-hot np.ndarray shape (16,)
shoot0 = df_b.loc[0, "shoot"]  # np.float32 scalar
```

#### `generate_majority_imitation_datasets(layout: GameLayout, num_samples_a: int, num_samples_b: int, p_one: float = 0.5, seed: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]`

**Purpose:** Convenience wrapper to generate paired datasets for Model A and Model B, using derived seeds to make draws reproducible but distinct.

**Arguments:**
- `layout`: `GameLayout` defining `field_size` and `comms_size`.
- `num_samples_a`: Number of samples for the Model A dataset.
- `num_samples_b`: Number of samples for the Model B dataset.
- `p_one`: Bernoulli probability that a field cell equals `1`.
- `seed`: Optional RNG seed; when provided, dataset A uses `seed` and dataset B uses `seed + 1`.

**Returns:**
- `Tuple[pd.DataFrame, pd.DataFrame]`: `(dataset_a, dataset_b)` where each DataFrame matches the return schema of `generate_majority_dataset_model_a` and `generate_majority_dataset_model_b`, respectively.

**Errors:**
- Propagates `ValueError` from underlying generators if `num_samples_a <= 0` or `num_samples_b <= 0`.
- Propagates any errors from segment construction or majority computation due to invalid layout.

**Example:**
```python
from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.neural_net_imitation_utilities import generate_majority_imitation_datasets

layout = GameLayout(field_size=6, comms_size=9)
df_a, df_b = generate_majority_imitation_datasets(layout, num_samples_a=2000, num_samples_b=2000, p_one=0.3, seed=42)
```

### Constants

Not specified.

### Types

Not specified.

## Dependencies

- `numpy` (imported as `np`): used for random sampling, array operations, and float32 conversions.
- `pandas` (imported as `pd`): used to construct DataFrame datasets.
- `typing`: `List`, `Tuple`, `Optional` used for type annotations.
- `Q_Sea_Battle.game_layout.GameLayout`: required for `field_size` and `comms_size`.

## Planned (design-spec)

Unknown (no design notes provided).

## Deviations

Unknown (no external spec provided to compare against).

## Notes for Contributors

- Keep `make_segments()` as the single source of truth for segment definitions; any changes must be reflected consistently in `compute_majority_comm()` and dataset generators.
- Dataset generators store NumPy arrays per row (object dtype columns in pandas); if changing storage format (e.g., expanding into multiple numeric columns), update downstream training code accordingly (not in this module).
- `compute_majority_comm()` treats ties as majority (`count >= L/2` yields `1.0`); changing this threshold will change teacher behavior and should be considered a breaking change.

## Related

- `Q_Sea_Battle.game_layout.GameLayout`
- NeuralNetPlayers Model A and Model B training routines (referenced in module docstring and comments; not defined here)

## Changelog

- 0.1: Initial version (as indicated in module docstring).