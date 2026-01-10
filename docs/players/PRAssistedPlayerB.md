# Class PRAssistedPlayerB

**Module import path**: `Q_Sea_Battle.pr_assisted_player_b.PRAssistedPlayerB`

> Player B that traces a one-hot gun through the PR-assisted hierarchy, collects outcome bits, and decides by parity.

!!! note "Derived constraints from GameLayout"
    Let `field_size = n` and `n2 = n**2`. Let `comms_size = m`.

    - `m == 1` (required by `PRAssistedPlayers`).
    - `n2` is a power of two.
    - Therefore `m | n2` holds trivially.

## Overview

`PRAssistedPlayerB` mirrors the hierarchy used by `PRAssistedPlayerA`.

At each level, it:
- Identifies the active pair in the one-hot gun representation.
- Builds a measurement string and queries the level resource (`measurement_b`).
- Extracts one outcome bit for that level.
- Collapses the gun representation to half length (preserving one-hot).

Finally, it returns the parity (XOR) of all collected outcome bits and the communication bit.

## Constructor

### Signature

- `PRAssistedPlayerB(game_layout: GameLayout, parent: PRAssistedPlayers) -> PRAssistedPlayerB`

### Arguments

- `game_layout`: `GameLayout`, scalar.
- `parent`: `PRAssistedPlayers`, scalar.

### Returns

- `PRAssistedPlayerB`, scalar.

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

- `decide(gun: np.ndarray, comm: np.ndarray, supp: Any | None = None) -> int`

#### Arguments

- `gun`: `np.ndarray`, dtype `int` (0, 1), shape `(n2,)`, one-hot.
- `comm`: `np.ndarray`, dtype `int` (0, 1), shape `(1,)`.
- `supp`: `Any` or `None`, scalar.
  - Unused.

#### Returns

- `shoot`: `int` (0, 1), scalar.

#### Preconditions

- `gun` is one-hot: `gun.sum() == 1`.
- `comm` has shape `(1,)`.
- At each internal level, the intermediate gun length is even and remains one-hot.
- Exactly one active pair `(0,1)` or `(1,0)` exists per level.

#### Postconditions

- Queries exactly one PR-assisted resource per level via:
  - `measurement_b(measurement)`.
- Appends the communication bit to the collected results.
- Returns the parity (XOR) of all collected bits as `shoot`.

#### Errors

- Raises `ValueError` if `gun` shape is not `(n2,)`.
- Raises `ValueError` if `gun` contains values outside `(0, 1)`.
- Raises `ValueError` if `gun` is not one-hot.
- Raises `ValueError` if `comm` shape is not `(1,)` or contains values outside `(0, 1)`.
- Raises `ValueError` if an intermediate gun length is not even.
- Raises `ValueError` if the intermediate gun is not one-hot at a level.
- Raises `ValueError` if there is no active pair or more than one active pair at a level.
- Raises `ValueError` if the measurement-string sum is not in `(0, 1)`.

!!! example "Minimal usage"
    ```python
    import numpy as np
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.pr_assisted_players import PRAssistedPlayers

    layout = GameLayout(field_size=4, comms_size=1)
    factory = PRAssistedPlayers(game_layout=layout, p_high=0.8)
    _, player_b = factory.players()

    gun = np.zeros(layout.field_size**2, dtype=int)
    gun[3] = 1
    comm = np.array([0], dtype=int)

    shoot = player_b.decide(gun, comm)
    ```

## Data & State

- `parent`: `PRAssistedPlayers`, scalar.

## Planned (design-spec)

- None identified from the provided implementation.

## Deviations

- None identified.

## Notes for Contributors

- Keep the one-hot invariant for `intermediate_gun` at every level.
- Any changes to the measurement-string rule must remain consistent with
  `PRAssistedPlayerA` and the expected resource correlation behaviour.

## Changelog

- 2026-01-10 — Author: Rob Hendriks
