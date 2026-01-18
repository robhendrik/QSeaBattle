# MajorityPlayerB

> Role: Player B implementation that maps a flattened gun index into a field segment and returns the corresponding communication bit as the shoot decision.
Location: `Q_Sea_Battle.majority_player_b.MajorityPlayerB`

## Derived constraints

- Let $n2$ be the length of the flattened gun vector and $m$ be the length of the communication vector; the implementation assumes $m$ divides $n2$ and uses $segment\_len = n2 / m$.

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | GameLayout, constraints: Not specified, shape: Not applicable | Game configuration for this player. |

Preconditions: `game_layout` must be a valid `GameLayout` instance accepted by `PlayerB.__init__` (validation rules not specified in this module).  
Postconditions: The instance is initialized via `PlayerB` with the provided `game_layout`.  
Errors: Not specified in this module; any exceptions may originate from `PlayerB.__init__`.  
Example:

```python
from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.majority_player_b import MajorityPlayerB

layout = GameLayout(...)  # Not specified here
player_b = MajorityPlayerB(layout)
```

## Public Methods

### decide

Decide whether to shoot based on majority information by segmenting the flattened gun index range into $m$ segments and selecting `comm[segment_index]`.

| Parameter | Type | Description |
| --- | --- | --- |
| gun | np.ndarray, dtype: any (coerced to int), constraints: intended one-hot after flattening, shape (n2,) after `ravel()` | Flattened one-hot gun vector of length $n2$. The implementation coerces to `int` and flattens. |
| comm | np.ndarray, dtype: any (coerced to int), constraints: values intended to be decision bits, shape (m,) after `ravel()` | Communication vector from Player A of length $m$. The implementation coerces to `int` and flattens. |
| supp | Optional[Any], constraints: unused, shape: Not applicable | Optional supporting information (unused). |

Returns: int, constraints: nominally in {0,1} (derived from `comm` values), shape: scalar; `1` to shoot or `0` to not shoot.

Preconditions: `gun` and `comm` must be array-like and convertible to `np.ndarray`; $m = len(comm) > 0$; $n2 = len(gun) \ge 1$; $m$ divides $n2$ (assumed by comment and required for equal-length segmentation); `gun` is assumed to be a valid one-hot vector (stated in comment).  
Postconditions: Returns `int(comm[segment_index])` where `segment_index` is computed from `argmax(gun)` and `segment_len = n2 // m`, with `segment_index` capped to `m - 1` if it overflows.  
Errors: May raise exceptions from `np.asarray`, `np.argmax`, division by zero when $m = 0$, or indexing errors if `comm` is empty; no explicit error handling is implemented.  
Example:

```python
import numpy as np
from Q_Sea_Battle.majority_player_b import MajorityPlayerB
from Q_Sea_Battle.game_layout import GameLayout

layout = GameLayout(...)  # Not specified here
p = MajorityPlayerB(layout)

gun = np.array([0, 0, 1, 0], dtype=int)     # n2 = 4, gun_index = 2
comm = np.array([0, 1], dtype=int)          # m = 2, segment_len = 2, segment_index = 1
decision = p.decide(gun=gun, comm=comm)     # returns 1
```

## Data & State

- Inherits all state from `PlayerB`; additional attributes are not defined in this class.
- Uses local variables within `decide`: `flat_gun` (np.ndarray, dtype int, shape (n2,)), `comm` (np.ndarray, dtype int, shape (m,)), `n2` (int, scalar), `m` (int, scalar), `segment_len` (int, scalar), `gun_index` (int, scalar), `segment_index` (int, scalar).

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- `decide` assumes the layout constraint that $m$ divides $n2$ but does not validate it; consider adding validation upstream (e.g., in `GameLayout` or `PlayerB`) if required.
- `decide` assumes `gun` is one-hot and uses `np.argmax`; for non-one-hot inputs, behavior is defined by `argmax` and may not match intended semantics.

## Related

- `Q_Sea_Battle.players_base.PlayerB` (base class; behavior and required interface not specified in this module)
- `Q_Sea_Battle.game_layout.GameLayout` (layout/config dependency; constraints not specified in this module)

## Changelog

- 0.1: Initial version (module docstring).