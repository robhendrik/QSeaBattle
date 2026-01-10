# Class PRAssisted

**Module import path**: `Q_Sea_Battle.pr_assisted.PRAssisted`

> Stateful two-party PR-assisted resource providing biased correlations between two measurements per round.

## Overview

`PRAssisted` implements a classical PR-assisted resource shared between two parties (A and B).
Each instance supports exactly two measurements per *round*:

1. The **first measurement** (by either party) returns a uniformly random binary string.
2. The **second measurement** returns a binary string correlated with the first according
   to the measurement settings and the parameter `p_high`.

The resource is **stateful within a round** and must be reset between rounds.

!!! note "Relation to legacy naming"
    This class is functionally identical to the former `SharedRandomness` resource.
    Only the naming and documentation have been updated.

## Constructor

### Signature

- `PRAssisted(length: int, p_high: float) -> PRAssisted`

### Arguments

- `length`: `int`, scalar.
  - Number of bits per measurement and outcome.
- `p_high`: `float`, scalar.
  - Correlation parameter in the interval `[0.0, 1.0]`.

### Returns

- `PRAssisted`, scalar.

### Preconditions

- `length >= 1`.
- `0.0 <= p_high <= 1.0`.

### Postconditions

- Internal measurement state is initialised:
  - `a_measured == False`
  - `b_measured == False`
- No previous measurement or outcome is stored.

### Errors

- Raises `TypeError` if argument types are invalid.
- Raises `ValueError` if `length < 1` or `p_high` is outside `[0.0, 1.0]`.

## Public Methods

### measurement_a

#### Signature

- `measurement_a(measurement: np.ndarray) -> np.ndarray`

#### Arguments

- `measurement`: `np.ndarray`, dtype `int` {0,1}, shape `(length,)`.

#### Returns

- `outcome`: `np.ndarray`, dtype `int` {0,1}, shape `(length,)`.

#### Preconditions

- Party A has not measured yet in the current round.
- `measurement` is 1D with shape `(length,)` and values in `{0,1}`.

#### Postconditions

- Marks party A as having measured.
- If this is the first measurement:
  - Stores the measurement and outcome internally.
- If this is the second measurement:
  - Returns a correlated outcome based on `p_high`.

#### Errors

- Raises `ValueError` if A already measured.
- Raises `ValueError` if `measurement` is invalid.

### measurement_b

#### Signature

- `measurement_b(measurement: np.ndarray) -> np.ndarray`

#### Arguments

- `measurement`: `np.ndarray`, dtype `int` {0,1}, shape `(length,)`.

#### Returns

- `outcome`: `np.ndarray`, dtype `int` {0,1}, shape `(length,)`.

#### Preconditions

- Party B has not measured yet in the current round.
- `measurement` is 1D with shape `(length,)` and values in `{0,1}`.

#### Postconditions

- Marks party B as having measured.
- Behaviour mirrors `measurement_a`.

#### Errors

- Raises `ValueError` if B already measured.
- Raises `ValueError` if `measurement` is invalid.

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

- Clears all internal measurement state.
- Allows the resource to be reused for a new round.

#### Errors

- None.

## Internal Behaviour

### Correlation rule

For each index `i`:

- If `(prev_measurement[i], current_measurement[i])` is in
  `((0, 0), (0, 1), (1, 0))`:
  - Outcome equals previous outcome with probability `p_high`.
- If `(1,1)`:
  - Outcome equals previous outcome with probability `1 - p_high`.

All indices are processed independently.

## Data & State

- `length`: `int`, scalar.
- `p_high`: `float`, scalar.
- `a_measured`: `bool`, scalar.
- `b_measured`: `bool`, scalar.
- `prev_measurement`: `np.ndarray`, dtype `int` {0,1}, shape `(length,)` or `None`.
- `prev_outcome`: `np.ndarray`, dtype `int` {0,1}, shape `(length,)` or `None`.

## Planned (design-spec)

- None identified.

## Deviations

- None identified.

## Notes for Contributors

- This class is intentionally minimal and stateful.
- Any change to the correlation rule must be mirrored in tests and in
  all PR-assisted player implementations.
- Do not add additional measurements per round without revisiting the protocol.

## Changelog

- 2026-01-10 — Author: Rob Hendriks
