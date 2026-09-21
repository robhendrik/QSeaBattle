# MajorityPlayerA

> Role: Deterministic Player A strategy that encodes per-segment majority bits from the field into the communication vector.

Location: `Q_Sea_Battle.majority_player_a.MajorityPlayerA`

## Derived constraints

- Let `field_size = game_layout.field_size`, `comms_size = game_layout.comms_size`, `n2 = field_size ** 2`, and `m = comms_size`. The implementation assumes $m$ divides $n2$, so `segment_len = n2 // m` is an integer and each segment has equal length.
- The communication vector has shape `(m,)` and dtype `int`, with each entry in `{0, 1}`.

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | GameLayout, constraints: not specified, shape: N/A | Game configuration for this player. |

Preconditions

- Not specified.

Postconditions

- `self.game_layout` is initialized via `PlayerA.__init__(game_layout)` (exact state details not specified in this module).

Errors

- Not specified.

Example

```python
from Q_Sea_Battle.majority_player_a import MajorityPlayerA
from Q_Sea_Battle.game_layout import GameLayout

gl = GameLayout(...)  # Not specified here
player = MajorityPlayerA(gl)
```

## Public Methods

### decide

Compute the communication vector from the field by taking a per-segment majority (ties map to 1).

Parameters

- `field`: np.ndarray, dtype: any (converted via `np.asarray(..., dtype=int)`), shape: any (flattened via `ravel()`); values are treated as integers when computing per-segment sums.
- `supp`: Optional[Any], constraints: may be None, shape: N/A; optional supporting information (not used).

Returns

- np.ndarray, dtype int, constraints: values in `{0,1}`, shape `(m,)` where `m = game_layout.comms_size`; each entry is the majority bit of the corresponding contiguous field segment with ties mapped to 1.

Errors

- Not specified.

Example

```python
import numpy as np
from Q_Sea_Battle.majority_player_a import MajorityPlayerA
from Q_Sea_Battle.game_layout import GameLayout

gl = GameLayout(...)  # Not specified here
player = MajorityPlayerA(gl)

field = np.random.randint(0, 2, size=(gl.field_size, gl.field_size))
comm = player.decide(field)
```

## Data & State

- Inherits from `PlayerA`; stored attributes are not specified in this module beyond using `self.game_layout` in `decide`.
- Uses `self.game_layout.field_size` and `self.game_layout.comms_size` during `decide`.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- The method `decide` flattens the provided `field` and only uses the first `n2 = field_size ** 2` values when slicing segments; any additional trailing values in `flat_field` beyond `n2` are ignored by the current slicing pattern (no explicit validation is performed in this module).
- The implementation relies on a comment-level assumption that `comms_size` divides `n2`; if this invariant can be violated elsewhere, consider adding explicit validation in `decide` or in `GameLayout`.

## Related

- `Q_Sea_Battle.game_layout.GameLayout`
- `Q_Sea_Battle.players_base.PlayerA`

## Changelog

- Not specified.