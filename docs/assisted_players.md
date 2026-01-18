# AssistedPlayers

> Role: Factory for assisted players with shared randomness.
Location: `Q_Sea_Battle.Archive.assisted_players.AssistedPlayers`

## Derived constraints

- Let field_size be `game_layout.field_size` (int, constraints: > 0, scalar).
- Let comms_size be `game_layout.comms_size` (int, constraints: must equal 1, scalar).
- Let $n2 = field\_size^2$ (int, constraints: > 0 and power of two, scalar).
- Let $n = \log_2(n2)$ (int, constraints: $2^n = n2$, scalar).
- The internal shared randomness array length is $n$ (int, constraints: $n \ge 0$, scalar), and the per-level lengths are $[2^{n-1}, 2^{n-2}, \ldots, 2^0]$ (list[int], constraints: each > 0 when $n>0$).

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | GameLayout, constraints: `comms_size == 1` and `field_size > 0` and `field_size ** 2` is a power of two, scalar | Game configuration. |
| p_high | float, constraints: Not specified, scalar | Correlation parameter used for all shared resources. Stored as `float(p_high)`. |

Preconditions

- `game_layout.comms_size == 1`.
- `n2 = game_layout.field_size ** 2` satisfies `n2 > 0`.
- `n2` is a power of two (bit-test check in `__init__`, and re-validated as an exact power of two inside `_create_shared_randomness_array()`).

Postconditions

- `self.game_layout` is initialized via `Players.__init__(game_layout)` (behavior of base class not specified here).
- `self.p_high` is set to `float(p_high)` (float, scalar).
- `self._shared_randomness_array` is created by `_create_shared_randomness_array()` (list[SharedRandomness], shape (n,)).
- `self._playerA` and `self._playerB` are set to `None` (each is `AssistedPlayerA | None` / `AssistedPlayerB | None`, scalar).

Errors

- Raises `ValueError` if `game_layout.comms_size != 1`.
- Raises `ValueError` if `game_layout.field_size ** 2 <= 0`.
- Raises `ValueError` if `game_layout.field_size ** 2` is not a power of two (checked in `__init__`), or not an exact power of two (checked again in `_create_shared_randomness_array()`).

Example

!!! example "Construct and retrieve players"
    ```python
    from Q_Sea_Battle.Archive.assisted_players import AssistedPlayers

    assisted = AssistedPlayers(game_layout=layout, p_high=0.9)
    player_a, player_b = assisted.players()
    ```

## Public Methods

### players

- Signature: `players(self) -> Tuple[PlayerA, PlayerB]`
- Returns: `tuple[PlayerA, PlayerB], constraints: length 2, shape (2,)` containing `(player_a, player_b)`.
- Behavior: Creates `AssistedPlayerA(self.game_layout, parent=self)` and `AssistedPlayerB(self.game_layout, parent=self)` on first call and caches them; later calls return the cached instances.

### reset

- Signature: `reset(self) -> None`
- Returns: `None, constraints: N/A, scalar`.
- Behavior: Recreates fresh shared randomness boxes by assigning `self._shared_randomness_array = self._create_shared_randomness_array()`; does not modify cached player references.

### shared_randomness

- Signature: `shared_randomness(self, index: int) -> SharedRandomness`
- Parameters:
  - `index`: `int, constraints: must be a valid index into internal array, scalar`.
- Returns: `SharedRandomness, constraints: element of internal array, scalar`.
- Errors: Raises `IndexError` if `index` is out of bounds (propagated from list indexing).

## Data & State

- `p_high`: `float, constraints: Not specified, scalar`; correlation parameter used to construct `SharedRandomness` instances.
- `_shared_randomness_array`: `list[SharedRandomness], constraints: length n where $n=\log_2(field\_size^2)$, shape (n,)`; created at construction and recreated on `reset()`.
- `_playerA`: `AssistedPlayerA | None, constraints: None until first `players()` call, scalar`; cached.
- `_playerB`: `AssistedPlayerB | None, constraints: None until first `players()` call, scalar`; cached.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- `_create_shared_randomness_array()` validates the exact power-of-two condition via `n = int(np.log2(n2))` and `2 ** n == n2`; this duplicates the earlier bit-test validation in `__init__`.
- `reset()` recreates shared randomness resources but does not clear `_playerA`/`_playerB`; cached players will continue to reference the same `AssistedPlayers` parent, and will observe the updated shared resources only if they query via the parent at use time (the players’ internal behavior is not specified in this module).

## Related

- `Q_Sea_Battle.Archive.players_base.Players`
- `Q_Sea_Battle.Archive.shared_randomness.SharedRandomness`
- `Q_Sea_Battle.Archive.assisted_player_a.AssistedPlayerA`
- `Q_Sea_Battle.Archive.assisted_player_b.AssistedPlayerB`
- `Q_Sea_Battle.Archive.game_layout.GameLayout`

## Changelog

- 0.1: Initial version (per module docstring).