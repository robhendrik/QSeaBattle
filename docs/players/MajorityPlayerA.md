# MajorityPlayerA

> **Role**: Player A encoding majority information over `m` contiguous segments of the flattened `field`.

**Location**: `Q_Sea_Battle.majority_player_a.MajorityPlayerA` fileciteturn7file0

!!! note "Derived constraints from GameLayout"
    Let `field_size = n` and `n2 = n**2`. Let `comms_size = m`.

    - `n2` is a power of two.
    - `m | n2` (i.e. `n2 % m == 0`). fileciteturn7file0

    These constraints are validated by `GameLayout` during construction.

## Constructor

| Parameter | Type | Description |
|---|---|---|
| game_layout | `GameLayout`, scalar | Shared configuration for this player instance. fileciteturn7file0 |

**Preconditions**

- `game_layout` is a valid `GameLayout`, scalar.
- `m | n2` holds (enforced by `GameLayout`). fileciteturn7file0

**Postconditions**

- `self.game_layout` references the provided `GameLayout`, scalar.

**Errors**

- No explicit exceptions are raised by the constructor.

!!! example "Example"
    ```python
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.majority_player_a import MajorityPlayerA

    player_a = MajorityPlayerA(GameLayout(field_size=8, comms_size=4))
    ```

## Public Methods

### decide

**Signature**

- `decide(field: np.ndarray, supp: Any | None = None) -> np.ndarray` fileciteturn7file0

**Purpose**

Encode per-segment majority information from `field` into a communication vector `comm` of length `m`.

**Behaviour**

- Flatten the input: `flat_field = np.asarray(field, dtype=int).ravel()`.
- Compute:
  - `n2 = field_size**2` (from `game_layout`).
  - `m = comms_size` (from `game_layout`).
  - `segment_len = n2 // m`.
- For each segment index `i` in `0..m-1`:
  - Let the segment be `flat_field[i*segment_len : (i+1)*segment_len]`.
  - Set `comm[i] = 1` if `ones >= zeros`, else `0`, where `zeros = segment_len - ones`. fileciteturn7file0

**Arguments**

- `field`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)` (intended).
  - Any shape is accepted and flattened internally.
- `supp`: `Any` or `None`, scalar.
  - Optional supporting information (unused). fileciteturn7file0

**Returns**

- `comm`: `np.ndarray`, dtype `int` {0,1}, shape `(m,)`.
  - One bit per segment, encoding segment majority. fileciteturn7file0

**Preconditions**

- `m > 0` and `m | n2`.
- Intended input values for `field` are in `{0,1}`.
- Intended input length after flattening is at least `n2`.
  - The implementation slices `flat_field` without validating length; short inputs may yield undersized segments.

**Postconditions**

- Returns an array of length `m`.
- For each segment `i`, `comm[i]` is determined solely by that segment’s majority rule.

**Errors**

- No explicit exceptions are raised.
- If the flattened field has fewer than `n2` elements, `segment` slices may be shorter than `segment_len`, which can
  bias the computed `zeros = segment_len - ones`. fileciteturn7file0

!!! example "Example"
    ```python
    import numpy as np
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.majority_player_a import MajorityPlayerA

    layout = GameLayout(field_size=4, comms_size=2)
    player_a = MajorityPlayerA(layout)

    field = np.zeros((layout.field_size * layout.field_size,), dtype=int)
    field[:8] = [1, 1, 0, 0, 1, 0, 0, 0]
    comm = player_a.decide(field)
    ```

## Data & State

- Attributes (public):
  - `game_layout` — `GameLayout`, scalar.

- Side effects:
  - None (pure function of inputs).

- Thread-safety:
  - Thread-safe for concurrent calls if inputs are independent (no RNG, no mutation).

## Planned (design-spec)

- None. This implements the majority-segment encoding described in the project materials. fileciteturn7file0

## Deviations

- Tie-breaking:
  - Implementation uses `ones >= zeros` which breaks ties toward `1`. fileciteturn7file0
  - If the design spec uses a different tie-breaking rule (e.g., ties toward `0`), document it here once identified.

## Notes for Contributors

- Keep the definition of `segment_len` consistent with `MajorityPlayerB`:
  both must agree on `segment_len = n2 // m` and contiguous segmentation.
- If you introduce alternative segmentations (e.g., shuffled indices), provide a shared index map to both players and
  document it explicitly.
- Consider adding explicit input validation in debug builds, but keep the core logic simple and fast.

## Related

- See also: `MajorityPlayerB` (decoding), and `MajorityPlayers` (factory).

## Changelog

- 2026-01-07 — Author: Rob Hendriks — Initial specification page for `MajorityPlayerA`.
