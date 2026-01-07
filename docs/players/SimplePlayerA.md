# SimplePlayerA

> **Role**: Deterministic Player A baseline that transmits the first `m` cells of the flattened `field`.

**Location**: `Q_Sea_Battle.simple_player_a.SimplePlayerA`

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
    from Q_Sea_Battle.simple_player_a import SimplePlayerA

    player_a = SimplePlayerA(GameLayout(field_size=4, comms_size=2))
    ```

## Public Methods

### decide

**Signature**

- `decide(field: np.ndarray, supp: Any | None = None) -> np.ndarray`

**Purpose**

Return a communication vector equal to the first `m` bits of the flattened `field`.

**Arguments**

- `field`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)` (intended).
  - Any shape is accepted; the method flattens via `np.asarray(field, dtype=int).ravel()`.
- `supp`: `Any` or `None`, scalar.
  - Optional supporting information (unused).

**Returns**

- `comm`: `np.ndarray`, dtype `int` {0,1}, shape `(m,)`.
  - `comm[j] == field_flat[j]` for all `j` in `0..m-1`, where `field_flat` is the flattened `field`.

**Preconditions**

- `m = self.game_layout.comms_size` and `m > 0`.
- Intended input values for `field` are in `{0,1}`.
  - The method does not validate that the input is binary; it only casts to `int`.

**Postconditions**

- Returns a copy (`.copy()`) of the selected slice; mutating the return value does not mutate the input `field`.

**Errors**

- No explicit exceptions are raised.
- May return fewer than `m` elements if `field` has fewer than `m` entries after flattening.

!!! example "Example"
    ```python
    import numpy as np
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.simple_player_a import SimplePlayerA

    layout = GameLayout(field_size=4, comms_size=2)
    player_a = SimplePlayerA(layout)

    field = np.zeros((layout.field_size * layout.field_size,), dtype=int)
    field[:4] = [1, 0, 1, 1]
    comm = player_a.decide(field)
    ```

## Data & State

- Attributes (public):
  - `game_layout` — `GameLayout`, scalar.

- Side effects:
  - None (pure function of inputs).

- Thread-safety:
  - Thread-safe for concurrent calls if `game_layout` is shared read-only (immutable).

## Planned (design-spec)

- None. This is a baseline strategy consistent with the "send agreed cells" family of strategies.

## Deviations

- Input shape handling:
  - Design typically treats `field` as already flattened with shape `(n2,)`.
  - Implementation accepts any shape and flattens internally.

## Notes for Contributors

- Keep the communication rule explicit: this player is used as a baseline and should remain easy to reason about.
- If you add validation (binary check, minimum length), document the new errors precisely.
- Preserve the return-copy behaviour to avoid accidental mutation bugs when callers reuse buffers.

## Related

- See also: `SimplePlayerB` (paired baseline) and `SimplePlayers` (factory).

## Changelog

- 2026-01-07 — Author: Rob Hendriks — Initial specification page for `SimplePlayerA`.
