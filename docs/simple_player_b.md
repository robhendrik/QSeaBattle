# SimplePlayerB

> Role: Deterministic Player B that maps early gun positions to communication bits and otherwise shoots probabilistically.

Location: `Q_Sea_Battle.simple_player_b.SimplePlayerB`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | GameLayout, constraints: Not specified, shape: N/A | Game configuration for this player; passed to the base `PlayerB` constructor. |

Preconditions

- `game_layout` must be compatible with `PlayerB.__init__` (exact requirements not specified in this module).

Postconditions

- The instance is initialized via `PlayerB` with the provided `game_layout`.

Errors

- Not specified (any exceptions raised by `PlayerB.__init__` may propagate).

Example

```python
from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.simple_player_b import SimplePlayerB

layout = GameLayout(...)  # Not specified in this module
player = SimplePlayerB(layout)
```

## Public Methods

### decide

Decide whether to shoot based on the gun position and the received communication vector.

| Parameter | Type | Description |
| --- | --- | --- |
| gun | np.ndarray, dtype int, constraints: intended one-hot, shape (n2,) after flattening | Flattened one-hot gun vector; the index of the maximum value is used as the gun cell index. |
| comm | np.ndarray, dtype int, constraints: values not specified, shape (m,) after flattening | Communication vector from Player A; if the gun index is within the first $m$ cells, the corresponding bit is returned. |
| supp | Any or None, constraints: unused, shape: N/A | Optional supporting information; ignored by this implementation. |

Returns

- int, constraints: {0,1}, shape: scalar; `1` means shoot, `0` means do not shoot.

Behavior

- Let `gun_index = argmax(gun)` after `gun` is converted to `np.ndarray` with `dtype=int` and flattened.
- Let `m = self.game_layout.comms_size`.
- If `gun_index < m`, return `int(comm[gun_index])` after `comm` is converted to `np.ndarray` with `dtype=int` and flattened.
- Otherwise, return `1` with probability `p = self.game_layout.enemy_probability` and `0` with probability `1 - p`, using `np.random.rand()`.

Preconditions

- `self.game_layout` must provide `comms_size` (used as `m`) and `enemy_probability` (used as `p`); types/constraints are not specified in this module.
- `comm` must have at least `m` elements after flattening to avoid index errors when `gun_index < m`.
- The intended input is a valid one-hot `gun`, but the code does not validate one-hotness.

Errors

- `IndexError` if `gun_index < m` and `comm` is shorter than `m` after flattening.
- Other NumPy-related errors may occur if inputs are not array-like; not specified further.

Example

```python
import numpy as np
from Q_Sea_Battle.simple_player_b import SimplePlayerB

player = SimplePlayerB(game_layout)  # game_layout must provide comms_size and enemy_probability

gun = np.zeros(25, dtype=int)
gun[3] = 1
comm = np.array([1, 0, 1, 0], dtype=int)

shoot = player.decide(gun=gun, comm=comm)
```

## Data & State

- Inherited state from `PlayerB` (not specified in this module).
- `self.game_layout`: GameLayout, constraints: must expose `comms_size` and `enemy_probability` for `decide`, shape: N/A.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- `decide` uses `np.argmax` on `gun` after flattening; if `gun` is not strictly one-hot, the behavior follows the maximum element rather than validating the encoding.
- Randomness is sourced from `np.random.rand()`; seeding and reproducibility controls are not handled in this class.

## Related

- `Q_Sea_Battle.simple_player_b.SimplePlayerB`
- `Q_Sea_Battle.players_base.PlayerB` (base class; behavior not specified here)
- `Q_Sea_Battle.game_layout.GameLayout` (provides `comms_size` and `enemy_probability`)

## Changelog

- 0.1: Initial documented version based on provided module text.