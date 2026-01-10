# Class PRAssistedPlayerA

**Module import path**: `Q_Sea_Battle.pr_assisted_player_a.PRAssistedPlayerA`

> Player A that compresses the field through a hierarchy of PR-assisted resources and outputs a 1-bit communication.

!!! note "Derived constraints from GameLayout"
    Let `field_size = n` and `n2 = n**2`. Let `comms_size = m`.

    - `m == 1` (required by `PRAssistedPlayers`).
    - `n2` is a power of two.
    - Therefore `m | n2` holds trivially.

## Overview

`PRAssistedPlayerA` repeatedly halves the field representation until one bit remains.

At each level, it:
- Builds a measurement string from adjacent bit pairs.
- Queries the level resource as the first measurement (`measurement_a`).
- Combines the outcome with selected original bits.
- Collapses again into a shorter intermediate field.

## Constructor

### Signature

- `PRAssistedPlayerA(game_layout: GameLayout, parent: PRAssistedPlayers) -> PRAssistedPlayerA`

### Arguments

- `game_layout`: `GameLayout`, scalar.
- `parent`: `PRAssistedPlayers`, scalar.
  - Factory that provides access to PR-assisted resources via `shared_randomness(level)`.

### Returns

- `PRAssistedPlayerA`, scalar.

### Preconditions

- `game_layout` is a valid `GameLayout`, scalar.
- `parent` is an instance of `PRAssistedPlayers`, scalar.

### Postconditions

- `self.game_layout` references `game_layout`.
- `self.parent` references `parent`.

### Errors

- Raises `TypeError` if `parent` is not a `PRAssistedPlayers` instance.

## Public Methods

### decide

#### Signature

- `decide(field: np.ndarray, supp: Any | None = None) -> np.ndarray`

#### Arguments

- `field`: `np.ndarray`, dtype `int` (0, 1), shape `(n2,)`.
- `supp`: `Any` or `None`, scalar.
  - Unused.

#### Returns

- `comm`: `np.ndarray`, dtype `int` (0, 1), shape `(1,)`.

#### Preconditions

- `field` is a flattened field of length `n2`.
- `field` contains only 0/1 values.
- At each internal level, the intermediate field length is even.

#### Postconditions

- Queries exactly one PR-assisted resource per level:
  - resource type: `PRAssisted`, scalar.
  - method: `measurement_a(measurement)`.
- Returns a single communication bit derived from the final intermediate field.

#### Errors

- Raises `ValueError` if `field` shape is not `(n2,)`.
- Raises `ValueError` if `field` contains values outside `(0, 1)`.
- Raises `ValueError` if an intermediate field length is not even.
- Raises `RuntimeError` if the final intermediate field is not length 1.

!!! example "Minimal usage"
    ```python
    import numpy as np
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.pr_assisted_players import PRAssistedPlayers

    layout = GameLayout(field_size=4, comms_size=1)
    factory = PRAssistedPlayers(game_layout=layout, p_high=0.8)
    player_a, _ = factory.players()

    field = np.zeros(layout.field_size**2, dtype=int)
    field[0] = 1
    comm = player_a.decide(field)
    ```

## Data & State

- `parent`: `PRAssistedPlayers`, scalar.

## Planned (design-spec)

- None identified from the provided implementation.

## Deviations

- None identified.

## Notes for Contributors

- This player assumes the factory returns a `PRAssisted` resource from `shared_randomness(level)`.
- Keep the intermediate-field halving invariant:
  - each iteration maps length `L` to `L/2`.
- Any changes to the measurement-string definition must be mirrored in `PRAssistedPlayerB`
  to keep the protocol consistent.

## Changelog

- 2026-01-10 — Author: Rob Hendriks
