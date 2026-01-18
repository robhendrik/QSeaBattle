# SimplePlayerA

> Role: Deterministic Player A that encodes the first $m$ bits of the flattened field into the communication vector.

Location: `Q_Sea_Battle.simple_player_a.SimplePlayerA`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | GameLayout, constraints Not specified, shape Not applicable | Game configuration for this player. |

Preconditions

- `game_layout` is a `GameLayout` instance compatible with `PlayerA` and provides `comms_size` (type constraints Not specified).

Postconditions

- The instance is initialized via `PlayerA.__init__(game_layout)`.
- `self.game_layout` is available (inherited; exact storage not specified in this module).

Errors

- Not specified.

Example

```python
from Q_Sea_Battle.simple_player_a import SimplePlayerA
from Q_Sea_Battle.game_layout import GameLayout

gl = GameLayout(...)  # Not specified in this module
player = SimplePlayerA(gl)
```

## Public Methods

### decide

| Parameter | Type | Description |
| --- | --- | --- |
| field | np.ndarray, dtype convertible to int, constraints values intended {0,1}, shape (any, ...) | Field array of 0/1 values; any shape is accepted and will be flattened internally. |
| supp | Any or None, constraints Optional, shape Not applicable | Optional supporting information (unused). |

Returns

- np.ndarray, dtype int, constraints derived from `field` after `np.asarray(..., dtype=int)`, shape (m,) where $m = \text{self.game_layout.comms_size}$.

Preconditions

- `self.game_layout.comms_size` exists and is an integer $m$ with $0 \le m$ (upper bound not specified).
- `field` is array-like and convertible via `np.asarray(field, dtype=int)`.

Postconditions

- Returns a copy of the first $m$ elements of `np.asarray(field, dtype=int).ravel()`.

Errors

- May raise any exception propagated by `np.asarray(field, dtype=int)` if conversion fails (exact exception types not specified).
- May raise an exception if `self.game_layout` or `self.game_layout.comms_size` is missing (exact exception types not specified).

Example

```python
import numpy as np
from Q_Sea_Battle.simple_player_a import SimplePlayerA

player = SimplePlayerA(game_layout)  # game_layout must provide comms_size
field = np.array([[1, 0, 1], [0, 1, 0]])
comms = player.decide(field)
```

## Data & State

- Inherits state from `PlayerA` (not defined in this module).
- Reads `self.game_layout.comms_size` during `decide` to determine $m$ (type/validation not specified here).

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- Keep `decide` deterministic: it should depend only on `field` and `self.game_layout.comms_size`.
- `supp` is currently unused; if future behavior uses it, update the method contract accordingly.

## Related

- `Q_Sea_Battle.players_base.PlayerA` (base class; not included in this module text)
- `Q_Sea_Battle.game_layout.GameLayout` (configuration type; not included in this module text)

## Changelog

- 0.1: Initial deterministic implementation encoding the first $m$ flattened field cells into the communication vector.