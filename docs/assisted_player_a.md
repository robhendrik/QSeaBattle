# AssistedPlayerA

> Role: Player A implementation that computes a single communication bit from a flattened binary field using iterative compression and shared randomness resources.

Location: `Q_Sea_Battle.Archive.assisted_player_a.AssistedPlayerA`

## Derived constraints

- Let field_size be `self.game_layout.field_size` (int, constraints Unknown), and let $n2 = field\_size^2$ (int, $n2 \ge 0$).
- Input `field` must be a 1D array of length $n2$ with values in `{0,1}` (enforced at runtime).
- At each compression level, the intermediate field length must be even (enforced at runtime); successful completion therefore requires $n2$ to be a power of 2 and $n2 \ge 1$ (not explicitly checked up-front, but otherwise a `ValueError` or `RuntimeError` will occur).

## Constructor

Parameter | Type | Description
--- | --- | ---
game_layout | `GameLayout`, constraints Unknown | Game configuration; stored via `PlayerA.__init__(game_layout)` and later used for `field_size`.
parent | `"AssistedPlayers"`, must be instance of `AssistedPlayers` | Owning factory providing access to shared randomness boxes; validated with `isinstance` and stored as `self.parent`.

Preconditions

- `parent` is an instance of `AssistedPlayers`.

Postconditions

- `self.game_layout` is initialized by `PlayerA`.
- `self.parent` is set to `parent`.

Errors

- `TypeError`: if `parent` is not an `AssistedPlayers` instance.

Example

```python
from Q_Sea_Battle.Archive.assisted_player_a import AssistedPlayerA
from Q_Sea_Battle.Archive.game_layout import GameLayout
from Q_Sea_Battle.Archive.assisted_players import AssistedPlayers

layout = GameLayout(...)  # Not specified in this module
factory = AssistedPlayers(...)  # Not specified in this module
player_a = AssistedPlayerA(game_layout=layout, parent=factory)
```

## Public Methods

### decide

Signature: `decide(self, field: np.ndarray, supp: Any | None = None) -> np.ndarray`

Compute the communication bit from the field by iteratively pairing bits, constructing a measurement array, querying shared randomness for an outcome, and collapsing until a single bit remains.

Parameters

- `field`: `np.ndarray, dtype int, values {0,1}, shape (n2,)` where $n2 = field\_size^2$; will be converted via `np.asarray(field, dtype=int)`.
- `supp`: `Any | None`, constraints Unknown; unused (explicitly deleted).

Returns

- `np.ndarray, dtype int, values {0,1}, shape (1,)`; the single communication bit as a length-1 array.

Preconditions

- `field` is 1D with `field.shape[0] == n2`.
- `field` contains only 0/1 values.
- For all iterations, `intermediate_field.size` is even (implies an even-length progression down to size 1).

Postconditions

- Returns a length-1 array containing the final collapsed bit.

Errors

- `ValueError`: if `field` is not 1D or does not have length `n2`.
- `ValueError`: if `field` contains values outside `{0,1}`.
- `ValueError`: if `intermediate_field` length is odd at any compression level.
- `RuntimeError`: if the final `intermediate_field` does not have length 1.

Example

```python
import numpy as np
from Q_Sea_Battle.Archive.assisted_player_a import AssistedPlayerA

# player_a is an AssistedPlayerA instance.
# n2 must equal player_a.game_layout.field_size ** 2.
field = np.zeros((player_a.game_layout.field_size ** 2,), dtype=int)
comm = player_a.decide(field)
assert comm.shape == (1,)
assert comm.dtype == int
```

## Data & State

- `parent`: `AssistedPlayers`, constraints Unknown; set in the constructor and used by `decide` via `self.parent.shared_randomness(level)`.
- `game_layout`: `GameLayout`, constraints Unknown; inherited from `PlayerA` and used by `decide` to compute `n2 = field_size ** 2`.

## Planned (design-spec)

- Not specified.

## Deviations

- No design notes were provided; deviations are not specified.

## Notes for Contributors

- `decide` assumes that repeated halving reaches length 1 without remainder; if this class is used with layouts where $n2$ is not a power of 2, the method may raise `ValueError` (odd intermediate length) or `RuntimeError` (final length not 1).
- The implementation relies on `self.parent.shared_randomness(level)` returning an object that supports `measurement_a(measurement)` where `measurement` is `np.ndarray, dtype int, shape (half,)` and the returned `outcome_a` is indexable with length `half`; the concrete interface is defined outside this module.

## Related

- `Q_Sea_Battle.Archive.players_base.PlayerA` (base class; behavior not specified in this module)
- `Q_Sea_Battle.Archive.game_layout.GameLayout` (layout providing `field_size`)
- `Q_Sea_Battle.Archive.assisted_players.AssistedPlayers` (provider of shared randomness; imported locally in `__init__`)

## Changelog

- 0.1: Initial version as indicated by module docstring; implements `AssistedPlayerA` with iterative compression using shared randomness.