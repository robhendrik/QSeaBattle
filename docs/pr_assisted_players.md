# PRAssistedPlayers

> Role: Factory for PR-assisted players that owns a per-level hierarchy of shared PR-assisted resources and vends a cached paired `(PlayerA, PlayerB)`.

Location: `Q_Sea_Battle.pr_assisted_players.PRAssistedPlayers`

## Derived constraints

- `comms_size`: `int`, must equal `1`.
- `field_size`: `int`, must be positive.
- `n2`: `int`, defined as $n2 = field\_size^2$, must be a positive power of two.
- `p_rule`: `float`, no additional constraints specified.
- PR-assisted hierarchy size: if $n2 = 2^n$ for integer $n$, then the factory owns exactly $n$ `PRAssisted` resources with lengths $2^{n-1}, 2^{n-2}, \dots, 2^0$.

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| `game_layout` | `GameLayout`, constraints: `game_layout.comms_size == 1`, `game_layout.field_size > 0`, and `n2 = game_layout.field_size ** 2` is a power of two, shape: N/A | Game configuration and board dimensions. |
| `p_rule` | `float`, constraints: not specified, shape: scalar | Correlation parameter used for all owned PR-assisted resources. |

Preconditions

- `game_layout.comms_size == 1`.
- `game_layout.field_size > 0`.
- `n2 = game_layout.field_size ** 2` is a power of two.

Postconditions

- `self.p_rule` is set to `float(p_rule)`.
- `self._pr_assisted_array` is created via `_create_pr_assisted_array()` and contains per-level `PRAssisted` resources.
- `self._playerA` and `self._playerB` are initialized to `None` (lazy creation on first `players()` call).

Errors

- Raises `ValueError` if `comms_size != 1`.
- Raises `ValueError` if `field_size` is not positive (i.e., `field_size ** 2 <= 0`).
- Raises `ValueError` if `field_size ** 2` is not a power of two.

Example

```python
from Q_Sea_Battle.pr_assisted_players import PRAssistedPlayers
from Q_Sea_Battle.game_layout import GameLayout

layout = GameLayout(field_size=4, comms_size=1)  # n2 = 16
factory = PRAssistedPlayers(game_layout=layout, p_rule=0.9)
player_a, player_b = factory.players()
```

## Public Methods

### `players(self) -> Tuple[PlayerA, PlayerB]`

Return the paired players, creating them on first use; the created players keep a reference to this factory as their parent and use it to access the per-level PR-assisted resources.

Parameters

- None.

Returns

- `tuple[PlayerA, PlayerB]`, constraints: length exactly `2`, shape: `(2,)` tuple; returns `(player_a, player_b)`.

Errors

- Not specified.

### `reset(self) -> None`

Reset the owned PR-assisted resources by recreating the internal PR-assisted hierarchy; cached player objects are not recreated but will observe the new resources through the parent.

Parameters

- None.

Returns

- `None`, shape: N/A.

Errors

- Not specified.

### `pr_assisted(self, index: int) -> PRAssisted`

Return the PR-assisted resource at the given level.

Parameters

- `index`: `int`, constraints: must be a valid index into the internal PR-assisted resource list, shape: scalar.

Returns

- `PRAssisted`, constraints: resource exists at `index`, shape: N/A.

Errors

- Raises `IndexError` if `index` is out of bounds.

### `shared_randomness(self, index: int) -> PRAssisted`

Deprecated alias for `pr_assisted`; prints a warning and returns the same resource as `pr_assisted(index)`.

Parameters

- `index`: `int`, constraints: must be a valid index into the internal PR-assisted resource list, shape: scalar.

Returns

- `PRAssisted`, constraints: resource exists at `index`, shape: N/A.

Errors

- Raises `IndexError` if `index` is out of bounds (via `pr_assisted`).

!!! warning "Deprecation"
    `shared_randomness()` is deprecated; use `pr_assisted()` instead. This method emits a warning via `print(...)` in the current implementation.

### `set_replay_round(self, replay_specs: list[dict]) -> None`

Enable replay mode for all owned PR-assisted resources by applying `PRAssisted.set_replay_round(**spec)` to each resource.

Parameters

- `replay_specs`: `list[dict]`, constraints: length must equal `len(self._pr_assisted_array)` and every element must be a `dict`, shape: `(m,)` where `m = len(self._pr_assisted_array)`.

Returns

- `None`, shape: N/A.

Errors

- Raises `ValueError` if `len(replay_specs) != len(self._pr_assisted_array)`.
- Raises `ValueError` if any `replay_specs[i]` is not a `dict`.

### `clear_replay_round(self) -> None`

Disable replay mode for all owned PR-assisted resources by applying `PRAssisted.clear_replay_round()` to each resource.

Parameters

- None.

Returns

- `None`, shape: N/A.

Errors

- Not specified.

## Data & State

- `game_layout`: `GameLayout`, constraints: inherited from `Players` and additionally requires `comms_size == 1` and `n2 = field_size ** 2` is a power of two, shape: N/A.
- `p_rule`: `float`, constraints: not specified, shape: scalar.
- `_pr_assisted_array`: `list[PRAssisted]`, constraints: non-empty when `n2 >= 2`, ordered from largest resource to smallest, shape: `(m,)` where $m = \log_2(n2)$.
- `_playerA`: `PRAssistedPlayerA | None`, constraints: `None` until first `players()` call, shape: N/A.
- `_playerB`: `PRAssistedPlayerB | None`, constraints: `None` until first `players()` call, shape: N/A.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- `_create_pr_assisted_array()` relies on `np.log2(n2)` and integer casting; although constructor checks for power-of-two via bit test, `_create_pr_assisted_array()` also validates exact power-of-two via `2**n != n2` and raises `ValueError` on mismatch.
- `shared_randomness()` uses `print(...)` for deprecation warning; a TODO indicates an intended migration to `warnings.warn` when API policy allows.

## Related

- `Q_Sea_Battle.pr_assisted.PRAssisted`
- `Q_Sea_Battle.pr_assisted_player_a.PRAssistedPlayerA`
- `Q_Sea_Battle.pr_assisted_player_b.PRAssistedPlayerB`
- `Q_Sea_Battle.players_base.Players`
- `Q_Sea_Battle.game_layout.GameLayout`

## Changelog

- Not specified.