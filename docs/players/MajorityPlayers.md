# MajorityPlayers

> **Role**: `Players` factory producing a matched `(MajorityPlayerA, MajorityPlayerB)` pair.

**Location**: `Q_Sea_Battle.majority_players.MajorityPlayers` fileciteturn7file2

!!! note "Derived constraints from GameLayout"
    Let `field_size = n` and `n2 = n**2`. Let `comms_size = m`.

    - `n2` is a power of two.
    - `m | n2` (i.e. `n2 % m == 0`). fileciteturn7file2

    These constraints are validated by `GameLayout` during construction.

## Constructor

| Parameter | Type | Description |
|---|---|---|
| game_layout | `GameLayout` or `None`, scalar | Optional shared configuration. If `None`, constructs `GameLayout()` with defaults. fileciteturn7file2 |

**Preconditions**

- If provided, `game_layout` is a valid `GameLayout`, scalar.
- If `game_layout` is `None`, `GameLayout()` construction succeeds.

**Postconditions**

- `self.game_layout` is set to a valid `GameLayout`, scalar. fileciteturn7file2

**Errors**

- Propagates exceptions raised by `GameLayout()` when `game_layout` is `None`.

!!! example "Example"
    ```python
    from Q_Sea_Battle.majority_players import MajorityPlayers

    players = MajorityPlayers()
    player_a, player_b = players.players()
    ```

## Public Methods

### players

**Signature**

- `players() -> tuple[PlayerA, PlayerB]` fileciteturn7file2

**Purpose**

Construct and return a `(MajorityPlayerA, MajorityPlayerB)` pair that shares `self.game_layout`.

**Arguments**

- None.

**Returns**

- `(player_a, player_b)` where:
  - `player_a`: `MajorityPlayerA`, scalar.
  - `player_b`: `MajorityPlayerB`, scalar. fileciteturn7file2

**Preconditions**

- `self.game_layout` is a valid `GameLayout`, scalar.

**Postconditions**

- Returns newly constructed player instances.

**Errors**

- No explicit exceptions are raised by this method.

!!! note "Expected I/O shapes"
    - `field`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)` (input to `MajorityPlayerA.decide`).
    - `comm`: `np.ndarray`, dtype `int` {0,1}, shape `(m,)` (output of `MajorityPlayerA.decide`).
    - `gun`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)`, one-hot (input to `MajorityPlayerB.decide`).
    - `shoot`: `int` {0,1}, scalar (output of `MajorityPlayerB.decide`).

## Data & State

- Attributes (public):
  - `game_layout` — `GameLayout`, scalar.

- Side effects:
  - `players()` allocates new player objects on each call.

- Thread-safety:
  - Factory object is thread-safe in isolation (holds immutable `GameLayout` only).
  - Returned players are thread-safe only if called serially (they are pure given fixed inputs).

## Planned (design-spec)

- None. This factory implements the "majority segment encoding" player pair. fileciteturn7file2

## Deviations

- Segment definition:
  - Implementation uses **contiguous** segments in the flattened field (`segment_len = n2 // m`). fileciteturn7file0
  - If the design spec defines a different partitioning (e.g., agreed indices or random subsets), document it here
    once identified.

## Notes for Contributors

- Preserve the segmenting rule used by both players:
  - Player A must encode per-segment majorities.
  - Player B must decode by mapping gun index to segment index.
- If you change how segments are defined, change both Player A and Player B in lockstep and update this documentation.
- Keep implementations deterministic to serve as a stable baseline.

## Related

- See also: `MajorityPlayerA`, `MajorityPlayerB`, and `Players` (base factory).

## Changelog

- 2026-01-07 — Author: Rob Hendriks — Initial specification page for `MajorityPlayers`.
