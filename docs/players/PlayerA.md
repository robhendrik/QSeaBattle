# PlayerA

> **Role**: Base class for Player A, producing a communication vector `comm` from the observed `field`.

**Location**: `Q_Sea_Battle.players_base.PlayerA`

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
    from Q_Sea_Battle.players_base import PlayerA

    player_a = PlayerA(GameLayout(field_size=4, comms_size=2))
    ```

## Public Methods

### decide

**Signature**

- `decide(field: np.ndarray, supp: Any | None = None) -> np.ndarray`

**Purpose**

Return a communication vector of length `m`. The base implementation ignores inputs and samples a random binary vector.

**Arguments**

- `field`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)`.
- `supp`: `Any` or `None`, scalar.
  - Optional supporting information (unused in the base class).

**Returns**

- `comm`: `np.ndarray`, dtype `int` {0,1}, shape `(m,)`.

**Preconditions**

- `self.game_layout.comms_size == m` and `m > 0`.
- Intended input shape for `field` is `(n2,)`.
  - The base implementation does not validate `field` contents or shape.

**Postconditions**

- Returns a newly allocated array of length `m` with entries in `{0,1}`.

**Errors**

- No explicit exceptions are raised.
- NumPy RNG errors may propagate (rare; typically only in misconfigured environments).

!!! example "Example"
    ```python
    import numpy as np
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.players_base import PlayerA

    layout = GameLayout(field_size=4, comms_size=2)
    player_a = PlayerA(layout)

    field = np.zeros((layout.field_size * layout.field_size,), dtype=int)
    comm = player_a.decide(field, supp=None)
    ```

## Data & State

- Attributes (public):
  - `game_layout` — `GameLayout`, scalar.

- Side effects:
  - Uses NumPy global RNG to sample a random vector.

- Thread-safety:
  - Not thread-safe if used concurrently (relies on NumPy global RNG; no internal locks).

## Planned (design-spec)

- None. The base class exists to define the Player A role; concrete strategies are implemented in other modules.

## Deviations

- Input validation:
  - The design treats `field` as a binary vector of shape `(n2,)`.
  - The base implementation does not validate `field` shape or value set.

## Notes for Contributors

- If you implement a deterministic strategy, keep the signature stable:
  `decide(field, supp=None) -> comm` where `comm` has shape `(m,)`.
- If you add optional diagnostics (e.g., log-probabilities), ensure they do not change the return type of `decide`.
- Prefer explicit dtype handling (`int` with values in `{0,1}`) to avoid downstream confusion.

## Related

- See also: `Players` (factory), `PlayerB` (shoot decision), `Game` (orchestration).

## Changelog

- 2026-01-07 — Author: Rob Hendriks — Initial specification page for `PlayerA`.
