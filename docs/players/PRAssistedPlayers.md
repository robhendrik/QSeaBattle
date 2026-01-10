# Class PRAssistedPlayers

**Module import path**: `Q_Sea_Battle.pr_assisted_players.PRAssistedPlayers`

> Factory that owns a hierarchy of PR-assisted resources (`PRAssisted`) and serves a cached
> `(PRAssistedPlayerA, PRAssistedPlayerB)` pair.

!!! note "Derived constraints from GameLayout"
    Let `field_size = n` and `n2 = n**2`. Let `comms_size = m`.

    - `m == 1` (this factory enforces `comms_size == 1`).
    - `n2` is a power of two.
    - Therefore `m | n2` holds trivially.

## Overview

`PRAssistedPlayers` is a `Players`-style factory that:

- Validates that the layout is compatible with PR-assisted players.
- Creates `n = log2(n2)` PR-assisted resources with decreasing lengths.
- Hands out `PRAssistedPlayerA` and `PRAssistedPlayerB` instances that query those resources.

!!! note "Terminology"
    The factory method name is `shared_randomness(...)`, but the returned objects are
    `PRAssisted` resources (not `SharedRandomness`).

## Constructor

### Signature

- `PRAssistedPlayers(game_layout: GameLayout, p_high: float) -> PRAssistedPlayers`

### Arguments

- `game_layout`: `GameLayout`, scalar.
- `p_high`: `float`, scalar.
  - Correlation parameter forwarded to each `PRAssisted(length=L, p_high=p_high)`.

### Returns

- `PRAssistedPlayers`, scalar.

### Preconditions

- `game_layout` is a valid `GameLayout`, scalar.
- `game_layout.comms_size == 1`.
- `n2 = game_layout.field_size**2` is a power of two.

### Postconditions

- `self.game_layout` references the provided layout.
- `self.p_high` is set.
- `self._shared_randomness_array` is created with length `log2(n2)`:
  - `list[PRAssisted]`, length `n`, where `n = log2(n2)`.
- Cached players are initialised to `None`.

### Errors

- Raises `ValueError` if `game_layout.comms_size != 1`.
- Raises `ValueError` if `n2` is not a power of two.

## Public Methods

### players

#### Signature

- `players() -> tuple[PlayerA, PlayerB]`

#### Arguments

- None.

#### Returns

- `(player_a, player_b)` where:
  - `player_a`: `PRAssistedPlayerA`, scalar.
  - `player_b`: `PRAssistedPlayerB`, scalar.

#### Preconditions

- Factory instance is constructed successfully.

#### Postconditions

- Creates the cached player instances on first call and returns them.
- Subsequent calls return the cached instances.

#### Errors

- Propagates exceptions from player construction.

!!! example "Minimal usage"
    ```python
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.pr_assisted_players import PRAssistedPlayers

    layout = GameLayout(field_size=4, comms_size=1)
    factory = PRAssistedPlayers(game_layout=layout, p_high=0.8)

    player_a, player_b = factory.players()
    ```

### reset

#### Signature

- `reset() -> None`

#### Arguments

- None.

#### Returns

- `None`.

#### Preconditions

- None.

#### Postconditions

- Recreates `self._shared_randomness_array` with fresh PR-assisted resources.

#### Errors

- Propagates exceptions from `_create_shared_randomness_array()` (e.g. invalid `n2`).

### shared_randomness

#### Signature

- `shared_randomness(index: int) -> PRAssisted`

#### Arguments

- `index`: `int`, scalar.
  - Level index into the internal resource hierarchy.

#### Returns

- `resource`: `PRAssisted`, scalar.

#### Preconditions

- `0 <= index < len(self._shared_randomness_array)`.

#### Postconditions

- No mutation.

#### Errors

- Raises `IndexError` if `index` is out of bounds.

## Internal Methods

### _create_shared_randomness_array

#### Signature

- `_create_shared_randomness_array() -> list[PRAssisted]`

#### Arguments

- None.

#### Returns

- `resources`: `list[PRAssisted]`, length `n`, where `n = log2(n2)`.

#### Preconditions

- `n2 = field_size**2` is an exact power of two.

#### Postconditions

- Returns resources with lengths:
  - `2**(n-1), 2**(n-2), ..., 2**1, 2**0`.

#### Errors

- Raises `ValueError` if `n2` is not an exact power of two.

## Data & State

- `p_high`: `float`, scalar.
- `_shared_randomness_array`: `list[PRAssisted]`, length `log2(n2)`.
- `_playerA`: `PRAssistedPlayerA` or `None`, scalar.
- `_playerB`: `PRAssistedPlayerB` or `None`, scalar.

## Planned (design-spec)

- None identified from the provided implementation.

## Deviations

- Naming:
  - Some docstrings still use the phrase "shared randomness" for historical reasons,
    but the concrete resource type is `PRAssisted`.

## Notes for Contributors

- Keep the hierarchy size consistent with the field length:
  - `len(_shared_randomness_array) == log2(field_size**2)`.
- If you change the resource API (`PRAssisted.measurement_a/measurement_b/reset`),
  update both player implementations.
- Prefer leaving `shared_randomness(...)` as-is to avoid breaking external code;
  introduce a new alias method only with a clearly defined migration plan.

## Changelog

- 2026-01-10 — Author: Rob Hendriks
