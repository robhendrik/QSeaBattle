# SimplePlayers

> **Role**: `Players` factory producing a matched `(SimplePlayerA, SimplePlayerB)` pair.

**Location**: `Q_Sea_Battle.simple_players.SimplePlayers`

!!! note "Derived constraints from GameLayout"
    Let `field_size = n` and `n2 = n**2`. Let `comms_size = m`.

    - `n2` is a power of two.
    - `m | n2` (i.e. `n2 % m == 0`).

    These constraints are validated by `GameLayout` during construction.

## Constructor

| Parameter | Type | Description |
|---|---|---|
| game_layout | `GameLayout` or `None`, scalar | Optional shared configuration. If `None`, constructs `GameLayout()` with defaults. |

**Preconditions**

- If provided, `game_layout` is a valid `GameLayout`, scalar.
- If `game_layout` is `None`, `GameLayout()` construction succeeds.

**Postconditions**

- `self.game_layout` is set to a valid `GameLayout`, scalar.

**Errors**

- Propagates exceptions raised by `GameLayout()` when `game_layout` is `None`.

!!! example "Example"
    ```python
    from Q_Sea_Battle.simple_players import SimplePlayers

    players = SimplePlayers()
    player_a, player_b = players.players()
    ```

## Public Methods

### players

**Signature**

- `players() -> tuple[PlayerA, PlayerB]`

**Purpose**

Construct and return a `(SimplePlayerA, SimplePlayerB)` pair that shares `self.game_layout`.

**Arguments**

- None.

**Returns**

- `(player_a, player_b)` where:
  - `player_a`: `SimplePlayerA`, scalar.
  - `player_b`: `SimplePlayerB`, scalar.

**Preconditions**

- `self.game_layout` is a valid `GameLayout`, scalar.

**Postconditions**

- Returns newly constructed player instances.

**Errors**

- No explicit exceptions are raised by this method.

!!! note "Expected I/O shapes"
    - `field`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)` (input to `SimplePlayerA.decide`).
    - `comm`: `np.ndarray`, dtype `int` {0,1}, shape `(m,)` (output of `SimplePlayerA.decide`).
    - `gun`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)`, one-hot (input to `SimplePlayerB.decide`).
    - `shoot`: `int` {0,1}, scalar (output of `SimplePlayerB.decide`).

## Data & State

- Attributes (public):
  - `game_layout` — `GameLayout`, scalar.

- Side effects:
  - `players()` allocates new player objects on each call.

- Thread-safety:
  - Factory object is thread-safe in isolation (holds immutable `GameLayout` only).
  - Returned players are not necessarily thread-safe.

## Planned (design-spec)

- None. This factory exists to provide a baseline player pair.

## Deviations

- Determinism claim:
  - Class docstrings describe these as "deterministic" players.
  - `SimplePlayerB` uses a Bernoulli(`enemy_probability`) random decision when `gun_index >= m`, introducing
    randomness in those cases.

## Notes for Contributors

- Keep this pair minimal; it serves as a baseline and a sanity check for the environment and tournament wiring.
- If you want a fully deterministic baseline, replace the `gun_index >= m` branch in `SimplePlayerB` with a fixed rule
  and document the resulting behaviour and expected win-rate.
- Preserve the shape contracts `(n2,)` and `(m,)` and avoid hidden reshapes beyond flattening.

## Related

- See also: `SimplePlayerA`, `SimplePlayerB`, and `Players` (base factory).

## Changelog

- 2026-01-07 — Author: Rob Hendriks — Initial specification page for `SimplePlayers`.
