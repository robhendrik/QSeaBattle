# PRAssistedPlayerA

> Role: Compute Player A's single communication bit by iteratively reducing a binary field using PR-assisted resources.

Location: `Q_Sea_Battle.pr_assisted_player_a.PRAssistedPlayerA`

## Derived constraints

- Let `field_size` be `game_layout.field_size` (int, constraints: not specified in this module).
- Let `n2 = field_size**2` (int).
- Input `field` must be a 1D binary vector of length `n2` and values in `{0,1}`.
- The iterative reduction requires the intermediate vector length to be even at each level; therefore, for the method to succeed without raising `ValueError`, `n2` must allow repeated halving until 1 (i.e., `n2` should be a power of 2).

## Constructor

Parameter | Type | Description
--- | --- | ---
game_layout | `GameLayout`, constraints: instance of `Q_Sea_Battle.game_layout.GameLayout`, shape: N/A | Game configuration; used to obtain `field_size` and to initialize the `PlayerA` base class.
parent | `"PRAssistedPlayers"`, constraints: must be an instance of `Q_Sea_Battle.pr_assisted_players.PRAssistedPlayers`, shape: N/A | Factory/owner providing access to PR-assisted boxes via `parent.pr_assisted(level)`.

Preconditions

- `parent` is an instance of `Q_Sea_Battle.pr_assisted_players.PRAssistedPlayers`.

Postconditions

- `self.game_layout` is initialized via `PlayerA.__init__(game_layout)` (exact base behavior: not specified in this module).
- `self.parent` is set to `parent`.

Errors

- `TypeError`: raised if `parent` is not a `PRAssistedPlayers` instance.

Example

```python
from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.pr_assisted_players import PRAssistedPlayers
from Q_Sea_Battle.pr_assisted_player_a import PRAssistedPlayerA

game_layout = GameLayout(...)  # parameters not specified in this module
parent = PRAssistedPlayers(...)  # parameters not specified in this module
player_a = PRAssistedPlayerA(game_layout=game_layout, parent=parent)
```

## Public Methods

### decide(field, supp=None)

Compute Player A's communication bit from the given field via iterative pairwise-equality reductions and PR-assisted measurements.

Parameter | Type | Description
--- | --- | ---
field | `np.ndarray, dtype int, values {0,1}, shape (n2,)` | Flattened field array; required length is `n2 = self.game_layout.field_size**2`.
supp | `Any | None`, constraints: unused, shape: N/A | Optional supporting information (ignored).

Returns

- `np.ndarray, dtype int, values {0,1}, shape (1,)`: A single-element array containing the communication bit.

Preconditions

- `field` is 1D and has length `n2`.
- `field` contains only 0/1 values.
- For successful completion, `n2` must be reducible to 1 by repeated halving (otherwise an even-length check will fail at some level).

Postconditions

- Returns a 1-element integer NumPy array containing the final reduced bit.

Errors

- `ValueError`: if `field` is not 1D or does not have length `n2`.
- `ValueError`: if `field` contains values other than 0/1.
- `ValueError`: if an intermediate reduction level produces an odd-length intermediate vector (the algorithm requires an even length at each level).
- `RuntimeError`: if the loop terminates but the final intermediate vector is not length 1 (marked as unreachable given expected inputs).

Example

```python
import numpy as np

field_size = player_a.game_layout.field_size
n2 = field_size**2
field = np.zeros((n2,), dtype=int)  # values must be in {0,1}
comm = player_a.decide(field)
assert comm.shape == (1,)
assert int(comm[0]) in (0, 1)
```

## Data & State

- `parent`: `PRAssistedPlayers`, constraints: instance of `Q_Sea_Battle.pr_assisted_players.PRAssistedPlayers`, shape: N/A; owner/factory used to access PR-assisted boxes via `self.parent.pr_assisted(level)`.
- `game_layout`: `GameLayout`, constraints: set by `PlayerA` base class initialization, shape: N/A; used for `field_size` (base storage details: not specified in this module).

## Planned (design-spec)

- Not specified.

## Deviations

- None identified between module docstrings and implementation.

## Notes for Contributors

- The reduction uses explicit Python loops to compute pairwise equalities and to interleave arrays; refactoring to vectorized NumPy operations may be possible, but must preserve exact dtype (`int`) and value constraints (`{0,1}`) and must keep per-level PR-assisted calls (`self.parent.pr_assisted(level)` followed by `measurement_a(measurement)`).
- This implementation depends on `parent.pr_assisted(level)` returning an object with a `measurement_a(measurement)` method; the exact interface and return dtype/shape are not specified in this module but must be compatible with indexing `outcome_a[k]` for `k in range(half)`.

## Related

- `Q_Sea_Battle.players_base.PlayerA` (base class; behavior not specified here).
- `Q_Sea_Battle.game_layout.GameLayout` (provides `field_size`).
- `Q_Sea_Battle.pr_assisted_players.PRAssistedPlayers` (must provide `pr_assisted(level)`).

## Changelog

- Not specified.