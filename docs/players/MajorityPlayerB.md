# MajorityPlayerB

> **Role**: Player B decoding majority-segment communication by mapping the gun index to a segment index.

**Location**: `Q_Sea_Battle.majority_player_b.MajorityPlayerB` fileciteturn7file1

!!! note "Derived constraints from GameLayout"
    Let `field_size = n` and `n2 = n**2`. Let `comms_size = m`.

    - `n2` is a power of two.
    - `m | n2` (i.e. `n2 % m == 0`). fileciteturn7file1

    These constraints are validated by `GameLayout` during construction.

## Constructor

| Parameter | Type | Description |
|---|---|---|
| game_layout | `GameLayout`, scalar | Shared configuration for this player instance. fileciteturn7file1 |

**Preconditions**

- `game_layout` is a valid `GameLayout`, scalar.

**Postconditions**

- `self.game_layout` references the provided `GameLayout`, scalar.

**Errors**

- No explicit exceptions are raised by the constructor.

!!! example "Example"
    ```python
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.majority_player_b import MajorityPlayerB

    player_b = MajorityPlayerB(GameLayout(field_size=8, comms_size=4))
    ```

## Public Methods

### decide

**Signature**

- `decide(gun: np.ndarray, comm: np.ndarray, supp: Any | None = None) -> int` fileciteturn7file1

**Purpose**

Return the communication bit corresponding to the segment in which the gun index lies.

**Behaviour**

- Flatten inputs:
  - `flat_gun = np.asarray(gun, dtype=int).ravel()`.
  - `comm = np.asarray(comm, dtype=int).ravel()`.
- Derive:
  - `n2 = flat_gun.size`.
  - `m = comm.size`.
  - `segment_len = n2 // m`.
- Compute `gun_index = argmax(flat_gun)` and `segment_index = gun_index // segment_len`.
- Return `int(comm[segment_index])`, with a defensive clamp:
  - if `segment_index >= m`, set `segment_index = m - 1`. fileciteturn7file1

**Arguments**

- `gun`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)` (intended), one-hot.
  - Any shape is accepted and flattened internally.
- `comm`: `np.ndarray`, dtype `int` {0,1}, shape `(m,)` (intended).
  - Any shape is accepted and flattened internally.
- `supp`: `Any` or `None`, scalar.
  - Optional supporting information (unused). fileciteturn7file1

**Returns**

- `shoot`: `int` {0,1}, scalar.

**Preconditions**

- Intended inputs:
  - `gun` is one-hot (exactly one `1`).
  - `comm` has length `m` and values in `{0,1}`.
  - `m | n2`.
- The method does not validate these; incorrect inputs can lead to unintended behaviour.

**Postconditions**

- If `gun` is valid one-hot and `comm` is valid length `m`, returns the segment-majority bit for the gun index.

**Errors**

- No explicit exceptions are raised.
- If `comm` is empty (`m == 0`), `segment_len = n2 // m` raises `ZeroDivisionError`.
- If `comm` has fewer than `m` elements relative to `n2` segmentation assumptions, indexing may raise `IndexError`. fileciteturn7file1

!!! example "Example"
    ```python
    import numpy as np
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.majority_player_b import MajorityPlayerB

    layout = GameLayout(field_size=4, comms_size=2)
    player_b = MajorityPlayerB(layout)

    gun = np.zeros((layout.field_size * layout.field_size,), dtype=int)
    gun[7] = 1
    comm = np.array([1, 0], dtype=int)

    shoot = player_b.decide(gun, comm)
    ```

## Data & State

- Attributes (public):
  - `game_layout` — `GameLayout`, scalar.

- Side effects:
  - None (pure function of inputs).

- Thread-safety:
  - Thread-safe for concurrent calls if inputs are independent (no RNG, no mutation).

## Planned (design-spec)

- None. This implements decoding consistent with the majority segment encoding. fileciteturn7file1

## Deviations

- Source of `n2` and `m`:
  - Implementation derives `n2` from `gun.size` and `m` from `comm.size`, rather than from `game_layout`. fileciteturn7file1
  - A design might require `n2 == field_size**2` and `m == comms_size` from `game_layout` to be enforced explicitly.

- Segment index clamping:
  - Implementation clamps `segment_index` to `m - 1` if it exceeds bounds. fileciteturn7file1
  - Under valid inputs and `segment_len = n2 // m`, `segment_index` should always be in `0..m-1`.

## Notes for Contributors

- Keep segmentation consistent with `MajorityPlayerA`.
- Consider using `game_layout` for `n2` and `m` if you want stricter coupling and earlier error detection.
- If you add input validation (one-hot check, length checks), document new error behaviour explicitly.

## Related

- See also: `MajorityPlayerA` (encoding), and `MajorityPlayers` (factory).

## Changelog

- 2026-01-07 — Author: Rob Hendriks — Initial specification page for `MajorityPlayerB`.
