# PRAssistedPlayerB

> Role: Player B decision rule that reduces a one-hot target vector level-by-level and queries PR-assisted boxes to compute a parity-based shoot/no-shoot action.
Location: `Q_Sea_Battle.pr_assisted_player_b.PRAssistedPlayerB`

## Constructor

Parameter | Type | Description
--- | --- | ---
game_layout | GameLayout, constraints Not specified, shape N/A | Game configuration (field size, etc.).
parent | PRAssistedPlayers, constraints must be instance of PRAssistedPlayers, shape N/A | Owning factory providing access to PR-assisted boxes via `parent.pr_assisted(level)`.

### Preconditions

- `parent` is an instance of `PRAssistedPlayers`.

### Postconditions

- `self.parent` is set to the provided `parent`.
- Base class `PlayerB` is initialized with `game_layout`.

### Errors

- `TypeError`: If `parent` is not a `PRAssistedPlayers` instance.

### Example

```python
from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.pr_assisted_players import PRAssistedPlayers
from Q_Sea_Battle.pr_assisted_player_b import PRAssistedPlayerB

layout = GameLayout(...)  # Not specified
factory = PRAssistedPlayers(...)  # Not specified
player_b = PRAssistedPlayerB(game_layout=layout, parent=factory)
```

## Public Methods

### decide(gun, comm, supp=None)

Decide whether to shoot by iteratively halving a one-hot `gun` vector, querying one PR-assisted box per level, collecting one outcome bit per level at the unique active adjacent pair, and returning the parity of all collected bits plus the communicated bit.

Parameter | Type | Description
--- | --- | ---
gun | np.ndarray, dtype int {0,1}, shape (n2,) | One-hot gun vector where $n2 = field\_size^2$.
comm | np.ndarray, dtype int {0,1}, shape (1,) | Communication bit as a length-1 array; appended to the outcome bits before parity.
supp | Any \| None, constraints unused, shape N/A | Optional supporting information (unused).

Returns | Type | Description
--- | --- | ---
shoot | int, constraints {0,1}, shape scalar | `1` to shoot, `0` to not shoot.

#### Preconditions

- `gun` is 1D with shape `(n2,)`, where $n2 = field\_size^2$.
- `gun` contains only bits in `{0,1}` and is one-hot (`gun.sum() == 1`).
- `comm` is 1D with shape `(1,)` and contains only bits in `{0,1}`.

#### Postconditions

- Returns `shoot = (sum(results) % 2)` where `results` contains one PR-assisted outcome bit per reduction level plus `int(comm[0])`.

#### Errors

- `ValueError`: If `gun` shape is not `(n2,)`.
- `ValueError`: If `gun` contains values outside `{0,1}`.
- `ValueError`: If `gun` is not one-hot (`sum != 1`).
- `ValueError`: If `comm` shape is not `(1,)`.
- `ValueError`: If `comm` contains values outside `{0,1}`.
- `ValueError`: If an intermediate reduction step has odd length (`intermediate_gun.size % 2 != 0`).
- `ValueError`: If an intermediate reduction step violates one-hotness (`intermediate_gun.sum() != 1`).
- `ValueError`: If more than one active pair `(0,1)` or `(1,0)` is found at a level.
- `ValueError`: If no active pair `(0,1)` or `(1,0)` is found at a level.
- `ValueError`: If `measurement.sum()` is not in `{0,1}`.

#### Example

```python
import numpy as np
from Q_Sea_Battle.pr_assisted_player_b import PRAssistedPlayerB

# player_b: PRAssistedPlayerB (constructed with a compatible parent providing pr_assisted(level))
# Suppose field_size == 2 => n2 == 4
gun = np.array([0, 1, 0, 0], dtype=int)   # one-hot
comm = np.array([1], dtype=int)          # single bit

shoot = player_b.decide(gun=gun, comm=comm)
```

## Data & State

- `parent`: PRAssistedPlayers, constraints instance-checked in constructor, shape N/A; owning factory used to retrieve PR-assisted boxes via `self.parent.pr_assisted(level)`.
- Inherited state from `PlayerB`: Not specified in this module.

## Planned (design-spec)

- Not specified.

## Deviations

- No design notes provided; no deviations identified.

## Notes for Contributors

- Symbol definitions used by this page: `field_size` is `self.game_layout.field_size`; `n2 = field_size**2`.
- `supp` is explicitly unused and is deleted inside `decide`.
- The PR-assisted integration points are `self.parent.pr_assisted(level)` and the returned object's `measurement_b(measurement)`; their interfaces are not defined in this module.

## Related

- `Q_Sea_Battle.players_base.PlayerB` (base class; behavior not specified here).
- `Q_Sea_Battle.pr_assisted_players.PRAssistedPlayers` (provides `pr_assisted(level)`; imported locally to avoid cycles).
- `Q_Sea_Battle.game_layout.GameLayout` (provides `field_size`).

## Changelog

- Not specified.