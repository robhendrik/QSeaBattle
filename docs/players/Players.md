# Players

> **Role**: Factory and container for a pair of QSeaBattle players sharing a single `GameLayout`.

**Location**: `Q_Sea_Battle.players_base.Players`

!!! note "Derived constraints from GameLayout"
    Let `field_size = n` and `n2 = n**2`. Let `comms_size = m`.

    - `n2` is a power of two.
    - `m | n2` (i.e. `n2 % m == 0`).

    These constraints are validated by `GameLayout` during construction.

## Constructor

| Parameter | Type | Description |
|---|---|---|
| game_layout | `GameLayout` or `None`, scalar | Shared configuration. If `None`, constructs `GameLayout()` with defaults. |

**Preconditions**

- If provided, `game_layout` is a `GameLayout`, scalar.
- If `game_layout` is `None`, `GameLayout()` construction succeeds.

**Postconditions**

- `self.game_layout` is set to a valid `GameLayout`, scalar.

**Errors**

- Propagates exceptions raised by `GameLayout()` when `game_layout` is `None`.

!!! example "Example"
    ```python
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.players_base import Players

    layout = GameLayout(field_size=8, comms_size=4)
    players = Players(game_layout=layout)
    player_a, player_b = players.players()
    ```

## Public Methods

### players

**Signature**

- `players() -> tuple[PlayerA, PlayerB]`

**Purpose**

Construct and return a `PlayerA` and `PlayerB` instance that share the same `GameLayout`.

**Arguments**

- None.

**Returns**

- `(player_a, player_b)` where:
  - `player_a`: `PlayerA`, scalar.
  - `player_b`: `PlayerB`, scalar.

**Preconditions**

- `self.game_layout` is a valid `GameLayout`, scalar.

**Postconditions**

- Returns new player instances that reference `self.game_layout`.

**Errors**

- No explicit exceptions are raised by this method.

!!! note "Expected player interfaces"
    Orchestration expects the returned players to implement:

    - `player_a.decide(field, supp=None) -> comm`
      - `field`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)`
      - `supp`: `Any` or `None`, scalar
      - `comm`: `np.ndarray`, dtype `int` {0,1}, shape `(m,)`

    - `player_b.decide(gun, comm, supp=None) -> shoot`
      - `gun`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)`, one-hot
      - `comm`: `np.ndarray`, dtype `int` {0,1}, shape `(m,)`
      - `supp`: `Any` or `None`, scalar
      - `shoot`: `int` {0,1}, scalar

### reset

**Signature**

- `reset() -> None`

**Purpose**

Reset any internal state across both players (compatibility hook for subclasses).

**Arguments**

- None.

**Returns**

- `None`.

**Preconditions**

- None.

**Postconditions**

- Base implementation performs no state changes.

**Errors**

- No exceptions are raised by the base implementation.

## Data & State

- Attributes (public):
  - `game_layout` — `GameLayout`, scalar.

- Side effects:
  - None in the base class.

- Thread-safety:
  - The base class holds only immutable `GameLayout`; thread-safety depends on subclasses and returned player instances.

## Planned (design-spec)

- None. The factory concept and method names (`players`, `reset`) are implemented as specified.

## Deviations

- None identified.

## Notes for Contributors

- Keep `Players` lightweight: it should remain a simple factory/container.
- Subclasses may override `players()` to return specialised implementations, but should preserve tuple order:
  `(player_a, player_b)`.
- If a subclass introduces shared state (e.g., shared randomness buffers), implement `reset()` to clear it.
- Avoid embedding training logic here; training belongs in dedicated training modules or wrappers.

## Related

- See also: `PlayerA` and `PlayerB` in `Q_Sea_Battle.players_base`, and `Game` in `Q_Sea_Battle.game`.

## Changelog

- 2026-01-07 — Author: Rob Hendriks — Initial specification page for `Players`.
