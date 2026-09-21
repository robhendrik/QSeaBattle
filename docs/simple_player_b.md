# SimplePlayerB

> Role: Deterministic Player B policy that uses Player A’s communication bits when addressed by the gun index, otherwise falls back to stochastic shooting.

Location: `Q_Sea_Battle.simple_player_b.SimplePlayerB`

## Constructor

Parameter | Type | Description
| --- | --- | --- |
| `game_layout` | `GameLayout`, constraints: not specified, shape: not applicable | Game configuration for this player.

Preconditions

- `game_layout` is a `GameLayout` instance (not otherwise validated here).

Postconditions

- The instance is initialized via `PlayerB.__init__(game_layout)`.
- `self.game_layout` is available as provided by the base class (exact storage details not specified in this module).

Errors

- Not specified.

Example

```python
from Q_Sea_Battle.simple_player_b import SimplePlayerB
from Q_Sea_Battle.game_layout import GameLayout

layout = GameLayout(...)  # Not specified in this module
player_b = SimplePlayerB(game_layout=layout)
```

## Public Methods

### decide

`decide(self, gun: np.ndarray, comm: np.ndarray, supp: Optional[Any] = None) -> int`

Decide whether to shoot based on the gun and Player A’s message.

Parameter | Type | Description
| --- | --- | --- |
| `gun` | `np.ndarray`, dtype: any (cast to `int` via `np.asarray(..., dtype=int)`), constraints: expected one-hot (not enforced), shape: any (flattened to 1D via `ravel()`), resulting shape `(n2,)` | One-hot gun vector over enemy field cells; the selected index is `argmax(flat_gun)`. |
| `comm` | `np.ndarray`, dtype: any (cast to `int` via `np.asarray(..., dtype=int)`), constraints: expected length `m` where `m = self.game_layout.comms_size` (not enforced before indexing), shape: any (flattened to 1D via `ravel()`), resulting shape `(m,)` | Communication vector from Player A; used as addressed bits when `gun_index < m`. |
| `supp` | `Optional[Any]`, constraints: unused, shape: not applicable | Optional supporting information (unused). |

Returns

- `int`, constraints: in `{0,1}`, shape: scalar; action where `1` means shoot and `0` means do not shoot.

Preconditions

- `self.game_layout.comms_size` is available and is compatible with indexing into `comm` when `gun_index < m` (exact type constraints not specified in this module).
- `self.game_layout.enemy_probability` is available and is used as a probability threshold `p` (not validated to be within $[0, 1]$ in this module).

Postconditions

- If `gun_index < m`, returns `int(comm[gun_index])`.
- Otherwise, returns `int(np.random.rand() < p)`.

Errors

- May raise `IndexError` if `gun_index < m` but `comm` is shorter than `m` (or shorter than `gun_index + 1`).
- Other exceptions may be raised by NumPy conversions or attribute access if inputs or `game_layout` are incompatible (not exhaustively specified).

Example

```python
import numpy as np
from Q_Sea_Battle.simple_player_b import SimplePlayerB
from Q_Sea_Battle.game_layout import GameLayout

layout = GameLayout(...)  # Not specified in this module
player_b = SimplePlayerB(layout)

gun = np.array([0, 1, 0, 0])     # argmax -> 1
comm = np.array([1, 0, 1])       # m should be 3 to match comm length here
action = player_b.decide(gun=gun, comm=comm)
```

## Data & State

- Inherits from `PlayerB`; base-class state is not specified in this module.
- Uses `self.game_layout.comms_size` (defines `m`, the number of communication bits) and `self.game_layout.enemy_probability` (defines `p`, the stochastic shoot probability outside the addressing range).

## Planned (design-spec)

- Not specified.

## Deviations

- No design notes provided; no deviations identified.

## Notes for Contributors

- The method assumes `gun` represents a valid one-hot selection, but it does not validate one-hotness; it uses `argmax` on the flattened array, so ties or non-binary inputs will select the first maximum index.
- The method casts `gun` and `comm` to integer arrays; negative or non-binary values in `comm` will be returned as-is (after `int(...)`) when addressed.
- If strict validation is required (e.g., enforcing `comm` length equals `m` and ensuring $p \in [0,1]$), it must be added explicitly; this module currently does not enforce these constraints.

## Related

- `Q_Sea_Battle.players_base.PlayerB`
- `Q_Sea_Battle.game_layout.GameLayout`

## Changelog

- Not specified.