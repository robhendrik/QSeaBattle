# PlayerB

> **Role**: Base class for Player B, producing a binary decision `shoot` from `gun` and `comm`.

**Location**: `Q_Sea_Battle.players_base.PlayerB`

!!! note "Derived constraints from GameLayout"
    Let `field_size = n` and `n2 = n**2`. Let `comms_size = m`.

    - `n2` is a power of two.
    - `m | n2` (i.e. `n2 % m == 0`).

    These constraints are validated by `GameLayout` during construction.

## Constructor

| Parameter | Type | Description |
|---|---|---|
| game_layout | `GameLayout`, scalar | Shared configuration for this player instance. |

**Preconditions**

- `game_layout` is a valid `GameLayout`, scalar.

**Postconditions**

- `self.game_layout` references the provided `GameLayout`, scalar.

**Errors**

- No explicit exceptions are raised by the constructor.

!!! example "Example"
    ```python
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.players_base import PlayerB

    player_b = PlayerB(GameLayout(field_size=4, comms_size=2))
    ```

## Public Methods

### decide

**Signature**

- `decide(gun: np.ndarray, comm: np.ndarray, supp: Any | None = None) -> int`

**Purpose**

Return a binary decision `shoot` in `{0,1}`. The base implementation ignores inputs and samples randomly.

**Arguments**

- `gun`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)`, one-hot.
- `comm`: `np.ndarray`, dtype `int` {0,1}, shape `(m,)`.
- `supp`: `Any` or `None`, scalar.
  - Optional supporting information (unused in the base class).

**Returns**

- `shoot`: `int` {0,1}, scalar.

**Preconditions**

- `self.game_layout.comms_size == m` and `m > 0`.
- Intended input shapes are `gun.shape == (n2,)` and `comm.shape == (m,)`.
  - The base implementation does not validate shapes or one-hot property.

**Postconditions**

- Returns `0` or `1` as a Python `int`.

**Errors**

- No explicit exceptions are raised.
- NumPy RNG errors may propagate (rare; typically only in misconfigured environments).

!!! example "Example"
    ```python
    import numpy as np
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.players_base import PlayerB

    layout = GameLayout(field_size=4, comms_size=2)
    player_b = PlayerB(layout)

    gun = np.zeros((layout.field_size * layout.field_size,), dtype=int)
    gun[0] = 1
    comm = np.zeros((layout.comms_size,), dtype=int)

    shoot = player_b.decide(gun, comm, supp=None)
    ```

## Data & State

- Attributes (public):
  - `game_layout` — `GameLayout`, scalar.

- Side effects:
  - Uses NumPy global RNG to sample a random decision.

- Thread-safety:
  - Not thread-safe if used concurrently (relies on NumPy global RNG; no internal locks).

## Planned (design-spec)

- None. The base class exists to define the Player B role; concrete strategies are implemented in other modules.

## Deviations

- Input validation:
  - The design treats `gun` as one-hot and `comm` as binary of shape `(m,)`.
  - The base implementation does not validate input shapes or the one-hot property of `gun`.

## Notes for Contributors

- If you implement a strategy that uses `gun` and `comm`, keep the signature stable:
  `decide(gun, comm, supp=None) -> int`.
- Avoid returning NumPy scalar types; return a Python `int` in `{0,1}` to match orchestration expectations.
- If you add optional diagnostics, document their access patterns (e.g., `get_log_prob()` methods) separately.

## Related

- See also: `Players` (factory), `PlayerA` (comm decision), `GameEnv` (evaluation), `Game` (orchestration).

## Changelog

- 2026-01-07 — Author: Rob Hendriks — Initial specification page for `PlayerB`.
