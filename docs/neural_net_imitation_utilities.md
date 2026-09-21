# neural_net_imitation_utilities

> Role: Generate imitation-learning datasets for a split NeuralNetPlayers setup using a majority-vote teacher policy.

Location: `Q_Sea_Battle.neural_net_imitation_utilities`

## Overview

This module synthesizes supervised datasets used to train two models in a split architecture: Model A predicts communication bits from a flattened binary field, and Model B predicts a shoot decision from (communication bits, gun position). The teacher policy matches the project’s “majority strategy”: the flattened field of length `n2 = field_size ** 2` is partitioned into `m = comms_size` contiguous segments, and each communication bit indicates whether its segment contains a majority of ones (ties produce 1.0). Datasets are returned as `pandas.DataFrame` objects that store NumPy arrays per row (object dtype columns) to match expected training interfaces.

## Public API

### Functions

#### `make_segments(layout: GameLayout) -> List[Tuple[int, int]]`

**Purpose**: Partition a flattened field into `comms_size` contiguous segments with lengths as even as possible.

**Arguments**:  
- `layout`: `GameLayout` providing `field_size` and `comms_size`.

**Returns**: A list of `(start, end)` index pairs (slice-style, `end` exclusive) of length `layout.comms_size` that covers `[0, n2)` without gaps or overlaps.

**Errors**:  
- `ValueError`: If `field_size < 1`, `comms_size < 1`, or `comms_size > n2`.  
- `RuntimeError`: If the constructed segments do not cover `[0, n2)`.

**Example**:
```python
from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.neural_net_imitation_utilities import make_segments

layout = GameLayout(field_size=3, comms_size=4)
segments = make_segments(layout)  # list of (start, end) covering 0..8
```

#### `compute_majority_comm(fields: np.ndarray, layout: GameLayout) -> np.ndarray`

**Purpose**: Compute teacher communication bits by per-segment majority vote over flattened binary fields.

**Arguments**:  
- `fields`: NumPy array of shape `(N, n2)` containing flattened binary fields (values expected in `{0, 1}` or float equivalents).  
- `layout`: `GameLayout` defining `field_size` and `comms_size`.

**Returns**: NumPy array of shape `(N, m)` with values in `{0.0, 1.0}` and dtype `np.float32`.

**Errors**:  
- `ValueError`: If `fields` is not 2D or its second dimension is not `field_size ** 2`.

**Example**:
```python
import numpy as np
from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.neural_net_imitation_utilities import compute_majority_comm

layout = GameLayout(field_size=2, comms_size=2)  # n2=4, m=2
fields = np.array([[1, 0, 1, 1],
                   [0, 0, 1, 0]], dtype=np.float32)
comms = compute_majority_comm(fields, layout)  # shape (2, 2), dtype float32
```

#### `generate_majority_dataset_model_a(layout: GameLayout, num_samples: int, p_one: float = 0.5, seed: Optional[int] = None) -> pd.DataFrame`

**Purpose**: Generate an imitation dataset for Model A (field → communication) using IID Bernoulli fields and majority-vote targets.

**Arguments**:  
- `layout`: `GameLayout` defining `field_size` and `comms_size`.  
- `num_samples`: Number of samples to generate.  
- `p_one`: Probability that a given field cell equals `1`.  
- `seed`: Optional RNG seed for reproducibility.

**Returns**: A `pandas.DataFrame` with object columns storing NumPy arrays per row:  
- `field`: 1D NumPy array of shape `(n2,)`, dtype `np.float32`.  
- `comm`: 1D NumPy array of shape `(m,)`, dtype `np.float32`.

**Errors**:  
- `ValueError`: If `num_samples <= 0`.

**Example**:
```python
from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.neural_net_imitation_utilities import generate_majority_dataset_model_a

layout = GameLayout(field_size=5, comms_size=5)
df_a = generate_majority_dataset_model_a(layout, num_samples=1000, p_one=0.5, seed=123)
x_field = df_a.loc[0, "field"]  # np.ndarray shape (25,)
y_comm = df_a.loc[0, "comm"]    # np.ndarray shape (5,)
```

#### `generate_majority_dataset_model_b(layout: GameLayout, num_samples: int, p_one: float = 0.5, seed: Optional[int] = None) -> pd.DataFrame`

**Purpose**: Generate an imitation dataset for Model B ((comm, gun) → shoot) where the label is the communication bit for the segment containing the sampled gun index.

**Arguments**:  
- `layout`: `GameLayout` defining `field_size` and `comms_size`.  
- `num_samples`: Number of samples to generate.  
- `p_one`: Probability that a given field cell equals `1`.  
- `seed`: Optional RNG seed for reproducibility.

**Returns**: A `pandas.DataFrame` with columns:  
- `field`: 1D NumPy array of shape `(n2,)`, dtype `np.float32`.  
- `comm`: 1D NumPy array of shape `(m,)`, dtype `np.float32`.  
- `gun`: 1D one-hot NumPy array of shape `(n2,)`, dtype `np.float32`.  
- `shoot`: scalar `np.float32` in `{0.0, 1.0}`.

**Errors**:  
- `ValueError`: If `num_samples <= 0`.

**Example**:
```python
from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.neural_net_imitation_utilities import generate_majority_dataset_model_b

layout = GameLayout(field_size=4, comms_size=4)
df_b = generate_majority_dataset_model_b(layout, num_samples=500, p_one=0.3, seed=7)
gun_vec = df_b.loc[0, "gun"]     # one-hot length 16
shoot = df_b.loc[0, "shoot"]     # np.float32 scalar
```

#### `generate_majority_imitation_datasets(layout: GameLayout, num_samples_a: int, num_samples_b: int, p_one: float = 0.5, seed: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]`

**Purpose**: Convenience wrapper to generate paired datasets for Model A and Model B; when seeded, uses `seed` for Model A and `seed + 1` for Model B to keep draws reproducible and independent.

**Arguments**:  
- `layout`: `GameLayout` defining `field_size` and `comms_size`.  
- `num_samples_a`: Number of samples for the Model A dataset.  
- `num_samples_b`: Number of samples for the Model B dataset.  
- `p_one`: Probability that a given field cell equals `1`.  
- `seed`: Optional RNG seed.

**Returns**: Tuple `(dataset_a, dataset_b)` where each element is a `pandas.DataFrame` in the format returned by the corresponding generator.

**Errors**: Not specified (errors may propagate from called functions).

**Example**:
```python
from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.neural_net_imitation_utilities import generate_majority_imitation_datasets

layout = GameLayout(field_size=6, comms_size=3)
df_a, df_b = generate_majority_imitation_datasets(layout, num_samples_a=1000, num_samples_b=1000, p_one=0.5, seed=42)
```

### Constants

Not specified.

### Types

Not specified.

## Dependencies

- `numpy` (imported as `np`)
- `pandas` (imported as `pd`)
- `typing`: `List`, `Tuple`, `Optional`
- `Q_Sea_Battle.game_layout.GameLayout`

## Planned (design-spec)

Design notes not provided.

## Deviations

Not specified.

## Notes for Contributors

- The segmentation scheme is defined by `make_segments()` and is used consistently by dataset generators; keep any changes backward-compatible with the expected `(field -> comm)` and `(comm, gun -> shoot)` training interfaces.  
- DataFrames store per-row NumPy arrays (object columns) for `field`, `comm`, and `gun`; do not silently convert these to expanded numeric columns unless downstream training code is updated accordingly.  
- Tie-breaking in `compute_majority_comm()` is intentional: `count >= L/2` yields `1.0` on ties.

## Related

- `Q_Sea_Battle.game_layout.GameLayout`  
- Majority strategy as used by the project’s `MajorityPlayer` (referenced conceptually; not defined in this module)

## Changelog

Not specified.