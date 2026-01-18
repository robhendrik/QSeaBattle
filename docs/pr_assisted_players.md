# PRAssistedPlayers

> Role: Factory that owns a hierarchy of PR-assisted resources and hands out paired `PRAssistedPlayerA` / `PRAssistedPlayerB` instances.

Location: `Q_Sea_Battle.pr_assisted_players.PRAssistedPlayers`

## Derived constraints

- Let `field_size` be `game_layout.field_size` (int, constraints not specified in this module) and `comms_size` be `game_layout.comms_size` (int, constraints not specified in this module).
- `comms_size == 1` is required.
- Let `n2 = field_size ** 2` (int). `n2 > 0` is required.
- `n2` must be a power of two (equivalently: `n2 & (n2 - 1) == 0` for `n2 > 0`), and additionally `_create_pr_assisted_array()` enforces exact power-of-two via `2**n == n2` where $n = \lfloor \log_2(n2) \rfloor$.
- The internal PR-assisted resource list has length $n$, and contains resources with lengths $2^{n-1}, 2^{n-2}, \dots, 2^0$.

## Constructor

| Parameter | Type | Description |
|---|---|---|
| `game_layout` | `GameLayout`, constraints Unknown, shape N/A | Game configuration used to derive `field_size` and `comms_size`. |
| `p_high` | `float`, constraints Unknown, shape N/A | Correlation parameter used for all PR-assisted resources. Coerced via `float(p_high)`. |

Preconditions

- `game_layout.comms_size == 1`.
- `n2 = game_layout.field_size ** 2` satisfies `n2 > 0`.
- `n2` is a power of two.

Postconditions

- `self.p_high` is set to `float(p_high)`.
- `self._pr_assisted_array` is created via `_create_pr_assisted_array()`.
- `self._playerA is None` and `self._playerB is None` (players are created lazily by `players()`).

Errors

- Raises `ValueError` if `game_layout.comms_size != 1`.
- Raises `ValueError` if `game_layout.field_size ** 2 <= 0`.
- Raises `ValueError` if `game_layout.field_size ** 2` is not a power of two.

Example

```python
from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.pr_assisted_players import PRAssistedPlayers

layout = GameLayout(field_size=4, comms_size=1)  # other args, if any, are not specified here
factory = PRAssistedPlayers(game_layout=layout, p_high=0.9)
player_a, player_b = factory.players()
```

## Public Methods

### players

- Signature: `players(self) -> Tuple[PlayerA, PlayerB]`
- Returns: `Tuple[PlayerA, PlayerB]`, constraints Unknown, shape `(2,)` as a 2-tuple `(player_a, player_b)`.
- Behavior: Creates `PRAssistedPlayerA` and `PRAssistedPlayerB` on first call (lazy initialization) and caches them; later calls return the cached instances.

Errors

- Not specified.

Example

```python
player_a, player_b = factory.players()
player_a2, player_b2 = factory.players()
assert player_a is player_a2 and player_b is player_b2
```

### reset

- Signature: `reset(self) -> None`
- Returns: `None`, constraints N/A, shape N/A.
- Behavior: Recreates `self._pr_assisted_array` via `_create_pr_assisted_array()`; does not modify cached `_playerA` / `_playerB` in this module.

Errors

- May raise `ValueError` propagated from `_create_pr_assisted_array()` if the current `game_layout.field_size ** 2` is not an exact power of two.

Example

```python
factory.reset()
```

### pr_assisted

- Signature: `pr_assisted(self, index: int) -> PRAssisted`
- Parameters:
  - `index`: `int`, constraints: must be a valid list index for `self._pr_assisted_array`, shape N/A.
- Returns: `PRAssisted`, constraints Unknown, shape N/A.
- Behavior: Returns the PR-assisted resource at `self._pr_assisted_array[index]`.

Errors

- Raises `IndexError` if `index` is out of bounds (propagated from list indexing).

Example

```python
box0 = factory.pr_assisted(0)
```

### shared_randomness

- Signature: `shared_randomness(self, index: int) -> PRAssisted`
- Parameters:
  - `index`: `int`, constraints: must be a valid list index for `self._pr_assisted_array`, shape N/A.
- Returns: `PRAssisted`, constraints Unknown, shape N/A.
- Behavior: Prints a deprecation warning to stdout and delegates to `pr_assisted(index)`.

Errors

- Raises `IndexError` if `index` is out of bounds (via `pr_assisted`).

Example

```python
box0 = factory.shared_randomness(0)  # prints a deprecation warning
```

## Data & State

- `game_layout`: `GameLayout`, constraints Unknown, shape N/A; inherited from `Players` (assignment performed by `Players.__init__` as invoked by `super().__init__(game_layout)`).
- `p_high`: `float`, constraints Unknown, shape N/A; correlation parameter used when constructing each `PRAssisted`.
- `_pr_assisted_array`: `list[PRAssisted]`, constraints: length `n` where $n = \log_2(n2)$ with `n2 = field_size ** 2` an exact power of two, shape N/A.
- `_playerA`: `PRAssistedPlayerA | None`, constraints Unknown, shape N/A; cached instance created by `players()`.
- `_playerB`: `PRAssistedPlayerB | None`, constraints Unknown, shape N/A; cached instance created by `players()`.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- `_create_pr_assisted_array()` uses `np.log2(n2)` and casts to `int`, then verifies exactness via `2**n == n2`; keep this check if refactoring to avoid float rounding issues.
- `shared_randomness()` prints directly; if changing deprecation behavior, ensure compatibility considerations are addressed across the package.

## Related

- `Q_Sea_Battle.pr_assisted.PRAssisted`
- `Q_Sea_Battle.pr_assisted_player_a.PRAssistedPlayerA`
- `Q_Sea_Battle.pr_assisted_player_b.PRAssistedPlayerB`
- `Q_Sea_Battle.players_base.Players`
- `Q_Sea_Battle.game_layout.GameLayout`

## Changelog

- Version 0.1: Introduced `PRAssistedPlayers` with `pr_assisted()` and deprecated alias `shared_randomness()`.