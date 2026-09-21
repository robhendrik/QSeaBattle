# MajorityPlayerB

> Role: Player B strategy that maps the active one-hot gun index into one of $m$ contiguous index segments and returns the corresponding communication bit.

Location: `Q_Sea_Battle.majority_player_b.MajorityPlayerB`

## Derived constraints

- Let $n2$ be the flattened grid size (gun vector length) and $m$ be the communication length (comm vector length). The class docstring states the layout is configured such that $m$ divides $n2$, so segment length is $segment\_len = n2 // m$.

## Constructor

Parameter | Type | Description
--- | --- | ---
game_layout | `GameLayout`, constraints: not specified, shape: scalar object | Game configuration for this player.

Preconditions

- `game_layout` is a `GameLayout` instance (specific validation not specified in this module).
- Base class `PlayerB` constructor accepts the provided `game_layout` (details not specified in this module).

Postconditions

- The instance is initialized via `super().__init__(game_layout)`.

Errors

- Not specified in this module.

Example

```python
from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.majority_player_b import MajorityPlayerB

layout = GameLayout(...)  # Not specified here
player_b = MajorityPlayerB(layout)
```

## Public Methods

### decide(gun, comm, supp=None)

Decide whether to shoot based on the segment-selected comm bit.

Parameter | Type | Description
--- | --- | ---
gun | `np.ndarray`, dtype: convertible to `int`, constraints: intended one-hot (not validated), shape: (n2,) after `ravel()` | Flattened one-hot gun vector of length $n2$; the active index is computed via `np.argmax`.
comm | `np.ndarray`, dtype: convertible to `int`, constraints: values not specified, shape: (m,) after `ravel()` | Communication vector from Player A of length $m$; the selected bit is returned as the decision.
supp | `Optional[Any]`, constraints: ignored, shape: scalar object | Optional supporting information; not used.

Returns

- `int`, constraints: derived from `comm[segment_index]` after conversion to `int` (typically in `{0,1}` if `comm` is in `{0,1}`), shape: scalar.

Preconditions

- `gun` and `comm` are array-like and convertible via `np.asarray(..., dtype=int)`.
- $m = comm.size$ must be non-zero to avoid division by zero when computing `segment_len = n2 // m` (not explicitly checked).
- Intended: `gun` is a valid one-hot vector (not validated); behavior follows `np.argmax` even if not one-hot.

Postconditions

- Returns the element `comm[segment_index]` where `segment_index = gun_index // (n2 // m)` with a defensive clamp to `m - 1` if `segment_index >= m`.

Errors

- `ZeroDivisionError` if `comm.size == 0` (via `n2 // m`).
- `IndexError` if `comm.size == 0` (attempting `comm[segment_index]`), or other indexing issues if `comm` cannot be indexed as 1D after `ravel()`.
- Other NumPy conversion errors are not specified.

Example

```python
import numpy as np
from Q_Sea_Battle.majority_player_b import MajorityPlayerB
from Q_Sea_Battle.game_layout import GameLayout

layout = GameLayout(...)  # Not specified here
b = MajorityPlayerB(layout)

gun = np.array([0, 0, 1, 0, 0, 0], dtype=int)   # n2 = 6, gun_index = 2
comm = np.array([1, 0, 1], dtype=int)           # m = 3, segment_len = 2
# segments: [0-1]->comm[0], [2-3]->comm[1], [4-5]->comm[2]
decision = b.decide(gun, comm)
```

## Data & State

- Inherits state from `PlayerB` (state fields not specified in this module).
- No additional instance attributes are defined in this module.

## Planned (design-spec)

- Not specified.

## Deviations

- The class docstring states the implementation assumes $m$ divides $n2$; the `decide` method includes a defensive clamp for cases where the derived `segment_index >= m`, which is an accommodation for $m$ not exactly dividing $n2$ (or other inconsistencies).

## Notes for Contributors

- `decide` coerces both `gun` and `comm` to `int` and flattens them via `ravel()`. If future changes require preserving original shapes or dtypes, update this conversion logic and corresponding documentation.
- Consider adding explicit validation for `comm.size > 0` and for one-hot correctness of `gun` if stricter behavior is desired.

## Related

- `Q_Sea_Battle.players_base.PlayerB`
- `Q_Sea_Battle.game_layout.GameLayout`

## Changelog

- Not specified.