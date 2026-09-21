# reference_performance_utilities

> Role: Analytic reference utilities for QSeaBattle performance benchmarks; provides baseline per-shot success probabilities and information-theoretic bounds under simplified assumptions.

Location: `Q_Sea_Battle.reference_performance_utilities`

## Overview

This module provides closed-form or numerically-evaluated baseline success probabilities for several player strategies under simplified assumptions. The `expected_win_rate_*` functions return *per-shot* success probabilities (the probability that Bob's guess matches the true cell value for a uniformly random queried cell). These utilities are intended for sanity checks and plotting reference curves rather than simulating full game dynamics.

## Public API

### Functions

#### `binary_entropy(p: Number) -> float`

**Purpose:** Compute Shannon binary entropy \(H(p)\) in bits for a Bernoulli random variable with success probability `p`.

**Arguments:**
- `p`: Bernoulli success probability.

**Returns:** Binary entropy in bits; returns `0.0` for `p <= 0` or `p >= 1` (limiting value).

**Errors:** Not specified (no explicit raises).

**Example:**
```python
from Q_Sea_Battle.reference_performance_utilities import binary_entropy

h = binary_entropy(0.5)  # 1.0
```

#### `binary_entropy_reverse(H: Number, accuracy_in_digits: int = 8) -> float`

**Purpose:** Numerically invert binary entropy on the monotone branch `p ∈ [0.5, 1.0]`, solving for `p` such that `binary_entropy(p) == H` using bisection.

**Arguments:**
- `H`: Target entropy in bits; must lie in `[0.0, 1.0]`.
- `accuracy_in_digits`: Target absolute accuracy on entropy as a decimal exponent; tolerance is `10**(-accuracy_in_digits)`.

**Returns:** Approximate `p ∈ [0.5, 1.0]` satisfying `binary_entropy(p) == H`; returns `1.0` when `H == 0.0` and `0.5` when `H == 1.0`.

**Errors:**
- `ValueError`: If `H` is outside `[0.0, 1.0]`.
- `RuntimeError`: If bisection does not converge within 200 iterations.

**Example:**
```python
from Q_Sea_Battle.reference_performance_utilities import binary_entropy_reverse

p = binary_entropy_reverse(0.5, accuracy_in_digits=10)
```

#### `expected_win_rate_simple(field_size: int, comms_size: int, enemy_probability: Number = 0.5, channel_noise: Number = 0.0) -> float`

**Purpose:** Return the analytic per-shot success probability for the "Simple" strategy under an i.i.d. Bernoulli field model, partial exact communication coverage, and optional binary symmetric channel noise.

**Arguments:**
- `field_size`: Side length of the square field.
- `comms_size`: Number of cells effectively communicated/covered; must satisfy `1 <= comms_size <= field_size**2`.
- `enemy_probability`: Bernoulli parameter `p` for a cell being `1`; must lie in `[0.0, 1.0]`.
- `channel_noise`: Channel flip probability `c`; must lie in `[0.0, 1.0]`.

**Returns:** Expected success probability for a uniformly random queried cell.

**Errors:**
- `ValueError`: If `field_size < 1`, if `comms_size` is out of range, or if `enemy_probability` / `channel_noise` are outside `[0.0, 1.0]`.

**Example:**
```python
from Q_Sea_Battle.reference_performance_utilities import expected_win_rate_simple

s = expected_win_rate_simple(field_size=8, comms_size=16, enemy_probability=0.4, channel_noise=0.1)
```

#### `expected_win_rate_majority(field_size: int, comms_size: int, enemy_probability: Number = 0.5, channel_noise: Number = 0.0) -> float`

**Purpose:** Return the analytic per-shot success probability for the "Majority" strategy where the flattened field is partitioned into `comms_size` contiguous blocks, Alice sends the majority bit per block (ties resolve to `1`), and the communicated bits pass through a binary symmetric channel.

**Arguments:**
- `field_size`: Side length of the square field.
- `comms_size`: Number of communicated block-majority bits; must satisfy `1 <= comms_size <= field_size**2` and must evenly divide `field_size**2`.
- `enemy_probability`: Bernoulli parameter `p` for a cell being `1`; must lie in `[0.0, 1.0]`.
- `channel_noise`: Channel flip probability `c`; must lie in `[0.0, 1.0]`.

**Returns:** Expected success probability for a uniformly random queried cell, averaged over both random field realization and random queried index.

**Errors:**
- `ValueError`: If `field_size < 1`, if `comms_size` is out of range, if `field_size**2` is not divisible by `comms_size`, or if `enemy_probability` / `channel_noise` are outside `[0.0, 1.0]`.

**Example:**
```python
from Q_Sea_Battle.reference_performance_utilities import expected_win_rate_majority

s = expected_win_rate_majority(field_size=8, comms_size=8, enemy_probability=0.5, channel_noise=0.0)
```

#### `expected_win_rate_assisted(field_size: int, comms_size: int, enemy_probability: Number = 0.5, channel_noise: Number = 0.0, p_rule: Number = 0.9) -> float`

**Purpose:** Return the analytic per-shot success probability for classical AssistedPlayers under a one-bit communication setting with additional structural constraints.

**Arguments:**
- `field_size`: Side length of the square field.
- `comms_size`: Communication size; must be `1` (current implementation constraint).
- `enemy_probability`: Unused in the current implementation.
- `channel_noise`: Channel flip probability `c`; must lie in `[0.0, 1.0]`.
- `p_rule`: Assisted-correlation parameter in `[0, 1]`.

**Returns:** Expected success probability, clamped to `[0.0, 1.0]`.

**Errors:**
- `ValueError`: If `field_size < 1`, if `comms_size != 1`, if `field_size**2` is not a power of two, or if `channel_noise` / `p_rule` are outside `[0.0, 1.0]`.

**Example:**
```python
from Q_Sea_Battle.reference_performance_utilities import expected_win_rate_assisted

s = expected_win_rate_assisted(field_size=8, comms_size=1, channel_noise=0.05, p_rule=0.9)
```

#### `limit_from_mutual_information(field_size: int, comms_size: int, channel_noise: Number = 0.0, accuracy_in_digits: int = 8) -> float`

**Purpose:** Compute an Information-Causality success upper bound from mutual information by converting noisy channel capacity into an effective noiseless bit budget and inverting binary entropy to get a success-probability bound.

**Arguments:**
- `field_size`: Side length of the square field.
- `comms_size`: Number of communicated bits `m`; may be `0`; must satisfy `0 <= comms_size <= field_size**2`.
- `channel_noise`: Channel flip probability `c`; must lie in `[0.0, 1.0]`.
- `accuracy_in_digits`: Accuracy passed to `binary_entropy_reverse`.

**Returns:** Upper bound on per-shot success probability in `[0.5, 1.0]`; returns `0.5` for `comms_size == 0` or when effective bits are non-positive, and returns `1.0` when effective bits exceed or equal the number of cells.

**Errors:**
- `ValueError`: If `field_size < 1`, if `comms_size` is out of range, or if `channel_noise` is outside `[0.0, 1.0]`.

**Example:**
```python
from Q_Sea_Battle.reference_performance_utilities import limit_from_mutual_information

ub = limit_from_mutual_information(field_size=8, comms_size=10, channel_noise=0.1, accuracy_in_digits=8)
```

### Constants

Not specified.

### Types

#### `Number`

- Definition: `typing.Union[float, int]`

## Dependencies

- Standard library: `math`
- Typing: `typing.Union`
- Future import: `from __future__ import annotations`

## Planned (design-spec)

Not specified.

## Deviations

- `expected_win_rate_assisted` currently supports `comms_size == 1` only and requires `field_size**2` to be a power of two; `enemy_probability` is accepted but unused.

## Notes for Contributors

- Keep probability-domain inputs validated consistently (`[0.0, 1.0]` checks) and maintain explicit error messages, as these functions are intended as benchmark/reference utilities.  
- `binary_entropy_reverse` uses a fixed iteration budget (200) and a tolerance defined on entropy space; if altering convergence behavior, preserve determinism and document any changes to accuracy semantics.

## Related

- `binary_entropy` and `binary_entropy_reverse` are used to compute the mutual-information bound in `limit_from_mutual_information`.
- Strategy reference curves: `expected_win_rate_simple`, `expected_win_rate_majority`, `expected_win_rate_assisted`.

## Changelog

Not specified.