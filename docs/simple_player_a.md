# SimplePlayerA

> Role: Deterministic Player A policy that communicates the first $m$ bits of the flattened field.

Location: `Q_Sea_Battle.simple_player_a.SimplePlayerA`

## Derived constraints

- Let $m = \texttt{game_layout.comms_size}$.
- Output communication vector is derived as `np.asarray(field, dtype=int).ravel()[:m].copy()`.

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| `game_layout` | `GameLayout`, constraints: not specified; shape: N/A | Game configuration for this player; used to access `comms_size` and initialize the base `PlayerA`. |

Preconditions

- `game_layout`: Not specified.

Postconditions

- The instance is initialized via `PlayerA.__init__(game_layout)`.

Errors

- Not specified.

Example

```python
from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.simple_player_a import SimplePlayerA

gl = GameLayout(...)  # Not specified in this module
player = SimplePlayerA(game_layout=gl)
```

## Public Methods

### decide

Compute the communication vector for Player B by flattening the provided field in row-major order and returning the first $m$ entries.

Signature: `decide(self, field: np.ndarray, supp: Optional[Any] = None) -> np.ndarray`

Parameters

- `field`: `np.ndarray`, dtype: any (coerced to `int`), constraints: values intended to be in `{0,1}` (not enforced), shape: any (flattened internally).
- `supp`: `Optional[Any]`, constraints: unused, shape: N/A.

Returns

- `np.ndarray`, dtype `int`, constraints: not specified, shape `(m,)` where $m = \texttt{self.game_layout.comms_size}$ (if `flat_field.size >= m`; otherwise shape `(k,)` where $k = \texttt{flat_field.size}$).

Preconditions

- `self.game_layout.comms_size` is accessible (type/constraints not specified here).
- `field` is convertible via `np.asarray(field, dtype=int)`.

Postconditions

- Returns a copy of the slice `flat_field[:m]` (i.e., the returned array is not a view into `flat_field`).

Errors

- Any exception raised by `np.asarray(field, dtype=int)` or attribute access to `self.game_layout.comms_size` is not caught (exact types not specified).

Example

```python
import numpy as np
from Q_Sea_Battle.simple_player_a import SimplePlayerA

player = SimplePlayerA(game_layout=gl)  # gl provides comms_size = m
field = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8)

comms = player.decide(field)
# comms == np.asarray(field, dtype=int).ravel()[:gl.comms_size].copy()
```

## Data & State

- Inherits all data/state from `PlayerA` (not specified in this module).
- Reads `self.game_layout.comms_size` during `decide` (storage location and type not specified here).

## Planned (design-spec)

- No additional planned items provided.

## Deviations

- No deviations identified (no design notes provided).

## Notes for Contributors

- Keep determinism: do not introduce stochasticity, replay buffers, or shared resources unless explicitly added to the design notes.
- Preserve the flattening order: `np.asarray(...).ravel()` uses row-major order by default.

## Related

- `Q_Sea_Battle.players_base.PlayerA` (base class; behavior not specified here).
- `Q_Sea_Battle.game_layout.GameLayout` (provides `comms_size`; behavior not specified here).

## Changelog

- Not specified.