# PRAssistedPlayerB

> Role: Player B implementation using PR-assisted resources.

Location: `Q_Sea_Battle.pr_assisted_player_b.PRAssistedPlayerB`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | GameLayout, constraints: instance of `Q_Sea_Battle.game_layout.GameLayout`, shape: N/A | Game configuration passed to `PlayerB`. |
| parent | PRAssistedPlayers, constraints: instance of `Q_Sea_Battle.pr_assisted_players.PRAssistedPlayers`, shape: N/A | Owning factory providing access to PR-assisted boxes via `parent.pr_assisted(level)`. |

Preconditions: `parent` is an instance of `PRAssistedPlayers`.  
Postconditions: `self.game_layout` is initialized by `PlayerB.__init__(game_layout)`; `self.parent` is set to `parent`.  
Errors: Raises `TypeError` if `parent` is not a `PRAssistedPlayers` instance.  
Example:

```python
from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.pr_assisted_players import PRAssistedPlayers
from Q_Sea_Battle.pr_assisted_player_b import PRAssistedPlayerB

game_layout = GameLayout(...)
parent = PRAssistedPlayers(...)
player_b = PRAssistedPlayerB(game_layout=game_layout, parent=parent)
```

## Public Methods

### decide(gun, comm, supp=None)

Decide whether to shoot using PR-assisted resources and communication.

| Parameter | Type | Description |
| --- | --- | --- |
| gun | np.ndarray, dtype int {0,1}, shape (n2,) | One-hot gun vector. Here $n2 = \mathrm{field\_size}^2$ where `field_size = self.game_layout.field_size`. Must contain exactly one `1`. |
| comm | np.ndarray, dtype int {0,1}, shape (1,) | Communication array with a single bit at index `0`. |
| supp | Any \| None, constraints: unused, shape: N/A | Optional supporting information; ignored. |

Returns: int, constraints: in {0,1}, shape: scalar; `1` to shoot or `0` to not shoot.

Preconditions: `gun` can be converted to `np.ndarray` with `dtype=int` and satisfies one-hot and shape constraints; `comm` can be converted to `np.ndarray` with `dtype=int` and satisfies shape/value constraints; `self.parent.pr_assisted(level)` returns an object supporting `measurement_b(measurement)` for successive `level` values.  
Postconditions: Does not mutate input arrays; computes `shoot` as the parity (mod 2) of a list consisting of PR-assisted outcomes gathered per reduction level and `comm[0]`.  
Errors: Raises `ValueError` if any of the following checks fail: `gun` is not 1D of length `n2`; `gun` contains values outside `{0,1}`; `gun.sum() != 1`; `comm` is not 1D of length `1`; `comm` contains values outside `{0,1}`; at any reduction level the intermediate gun length is odd; the intermediate gun is not one-hot; there is not exactly one active pair `(0, 1)` or `(1, 0)` per level; the constructed `measurement` has sum not in `{0,1}`.  
Example:

```python
import numpy as np

field_size = player_b.game_layout.field_size
n2 = field_size ** 2

gun = np.zeros(n2, dtype=int)
gun[3] = 1
comm = np.array([1], dtype=int)

shoot = player_b.decide(gun=gun, comm=comm)
assert shoot in (0, 1)
```

## Data & State

- `parent`: PRAssistedPlayers, constraints: instance of `Q_Sea_Battle.pr_assisted_players.PRAssistedPlayers`, shape: N/A; reference to the owning factory that provides PR-assisted boxes via `pr_assisted(level)`.
- Inherited state: Not specified in this module; see `Q_Sea_Battle.players_base.PlayerB`.

## Planned (design-spec)

Not specified.

## Deviations

Not specified.

## Notes for Contributors

- This class enforces strict input validation (shape, dtype conversion to `int`, and {0,1} constraints) before interacting with PR-assisted resources; update validation consistently if upstream interfaces change.
- The constructor uses a local import of `PRAssistedPlayers` to avoid an import cycle.

## Related

- `Q_Sea_Battle.players_base.PlayerB`
- `Q_Sea_Battle.pr_assisted_players.PRAssistedPlayers`
- `Q_Sea_Battle.game_layout.GameLayout`

## Changelog

- 0.1: Initial version in package metadata; PR-assisted naming update described in module docstring.