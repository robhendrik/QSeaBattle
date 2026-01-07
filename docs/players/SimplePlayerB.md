# SimplePlayerB

> **Role**: Player B baseline that uses `comm[i]` when the gun points to index `i < m`, otherwise guesses randomly.

**Location**: `Q_Sea_Battle.simple_player_b.SimplePlayerB`

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
    from Q_Sea_Battle.simple_player_b import SimplePlayerB

    player_b = SimplePlayerB(GameLayout(field_size=4, comms_size=2))
    ```

## Public Methods

### decide

**Signature**

- `decide(gun: np.ndarray, comm: np.ndarray, supp: Any | None = None) -> int`

**Purpose**

Decide whether to shoot based on the gun index and the communication vector.

**Behaviour**

- Let `gun_index` be `argmax(gun_flat)`, where `gun_flat = np.asarray(gun, dtype=int).ravel()`.
- If `gun_index < m`, return `int(comm_flat[gun_index])`.
- Otherwise, return `1` with probability `enemy_probability` and `0` otherwise.

**Arguments**

- `gun`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)` (intended), one-hot.
  - Any shape is accepted; the method flattens internally.
- `comm`: `np.ndarray`, dtype `int` {0,1}, shape `(m,)` (intended).
  - Any shape is accepted; the method flattens internally.
- `supp`: `Any` or `None`, scalar.
  - Optional supporting information (unused).

**Returns**

- `shoot`: `int` {0,1}, scalar.

**Preconditions**

- `m = self.game_layout.comms_size` and `m > 0`.
- Intended input values:
  - `gun` is one-hot over `n2` positions.
  - `comm` values are in `{0,1}`.
- The method does not validate the one-hot property or shapes; it assumes the provided inputs are well-formed.

**Postconditions**

- If `gun_index < m`, the output equals the corresponding communication bit.
- If `gun_index >= m`, the output is sampled from a Bernoulli distribution with parameter `enemy_probability`.

**Errors**

- No explicit exceptions are raised.
- If `comm` has fewer than `m` entries (after flattening), indexing may raise `IndexError` when `gun_index < m`.
- If `gun` is not one-hot, `argmax` selects the first maximum; behaviour may be unintended.

!!! example "Example"
    ```python
    import numpy as np
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.simple_player_b import SimplePlayerB

    layout = GameLayout(field_size=4, comms_size=2, enemy_probability=0.5)
    player_b = SimplePlayerB(layout)

    gun = np.zeros((layout.field_size * layout.field_size,), dtype=int)
    gun[1] = 1
    comm = np.array([1, 0], dtype=int)

    shoot = player_b.decide(gun, comm)
    ```

## Data & State

- Attributes (public):
  - `game_layout` — `GameLayout`, scalar.

- Side effects:
  - Uses NumPy global RNG when `gun_index >= m`.

- Thread-safety:
  - Not thread-safe for concurrent calls (relies on NumPy global RNG; no internal locks).

## Planned (design-spec)

- None. This is a baseline Player B consistent with the "shoot from communicated cell" strategy family.

## Deviations

- Determinism:
  - The class docstring calls the player "deterministic".
  - The implementation is stochastic when `gun_index >= m` via `np.random.rand() < enemy_probability`.

- Input shape handling:
  - Design typically treats `gun` and `comm` as already flattened.
  - Implementation accepts any shape and flattens internally.

## Notes for Contributors

- If you want the baseline pair to be fully deterministic, replace the stochastic branch with a fixed rule and document it.
- Consider adding explicit validation for:
  - `gun` one-hot property,
  - `comm` length `m`,
  - binary value sets,
  but document any new errors and keep orchestration compatibility.

## Related

- See also: `SimplePlayerA` (paired baseline) and `SimplePlayers` (factory).

## Changelog

- 2026-01-07 — Author: Rob Hendriks — Initial specification page for `SimplePlayerB`.
