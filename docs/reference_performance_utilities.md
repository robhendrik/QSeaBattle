# Q_Sea_Battle.reference_performance_utilities

> Role: Utility functions for analytic performance benchmarks in QSeaBattle.

Location: `Q_Sea_Battle.reference_performance_utilities`

## Overview

This module provides analytic helper utilities used to benchmark expected player win rates and information-theoretic limits in the QSeaBattle package, including Shannon binary entropy, its inversion on a specified branch, closed-form (or semi-analytic) expected win-rate models for several player types, and an Information Causality success-probability bound.

## Public API

### Functions

#### `binary_entropy(p: Number) -> float`

**Signature**: `binary_entropy(p: Number) -> float`  
**Purpose**: Compute Shannon binary entropy \(H(p)\) in bits for a Bernoulli random variable with parameter `p`; returns limiting value `0.0` for out-of-domain inputs (`p <= 0` or `p >= 1`).  
**Arguments**: `p` (Number): Bernoulli parameter.  
**Returns**: `float`: Shannon binary entropy in bits.  
**Errors**: Not specified (no explicit exceptions raised; may raise standard exceptions if `p` is not convertible to `float`).  
**Example**:
```python
from Q_Sea_Battle.reference_performance_utilities import binary_entropy

print(binary_entropy(0.5))  # 1.0
print(binary_entropy(0.0))  # 0.0
```

#### `binary_entropy_reverse(H: Number, accuracy_in_digits: int = 8) -> float`

**Signature**: `binary_entropy_reverse(H: Number, accuracy_in_digits: int = 8) -> float`  
**Purpose**: Invert the binary entropy function on the branch \(p \in [0.5, 1.0]\) using a bisection search to find `p` such that `binary_entropy(p) ~= H` within a decimal tolerance determined by `accuracy_in_digits`.  
**Arguments**: `H` (Number): Target entropy value (must lie in `[0.0, 1.0]`). `accuracy_in_digits` (int): Number of decimal digits of accuracy used to set the absolute entropy tolerance (`10**(-accuracy_in_digits)`).  
**Returns**: `float`: A probability `p` in `[0.5, 1.0]` whose binary entropy matches `H` within tolerance.  
**Errors**: `ValueError`: If `H` is outside `[0.0, 1.0]`. `RuntimeError`: If the search does not converge within 200 iterations.  
**Example**:
```python
from Q_Sea_Battle.reference_performance_utilities import binary_entropy_reverse, binary_entropy

p = binary_entropy_reverse(1.0)
print(p)  # 0.5

p2 = binary_entropy_reverse(0.0)
print(p2)  # 1.0

# Consistency check (approximately)
p3 = binary_entropy_reverse(0.8, accuracy_in_digits=10)
print(binary_entropy(p3))
```

#### `expected_win_rate_simple(field_size: int, comms_size: int, enemy_probability: Number = 0.5, channel_noise: Number = 0.0) -> float`

**Signature**: `expected_win_rate_simple(field_size: int, comms_size: int, enemy_probability: Number = 0.5, channel_noise: Number = 0.0) -> float`  
**Purpose**: Compute an analytic expected win rate for `SimplePlayers` based on a coverage fraction determined by `comms_size` over the `field_size**2` cells, a correct-covered probability `(1 - channel_noise)`, and an uncovered success probability derived from `enemy_probability`.  
**Arguments**: `field_size` (int): Field side length; must be `>= 1`. `comms_size` (int): Number of communicated cells/bits (model-specific); must satisfy `1 <= comms_size <= field_size**2`. `enemy_probability` (Number): Probability a cell is an enemy; must lie in `[0.0, 1.0]`. `channel_noise` (Number): Binary symmetric channel flip probability; must lie in `[0.0, 1.0]`.  
**Returns**: `float`: Expected success probability.  
**Errors**: `ValueError`: If `field_size < 1`, if `comms_size` is not in `[1, field_size**2]`, if `enemy_probability` is not in `[0.0, 1.0]`, or if `channel_noise` is not in `[0.0, 1.0]`.  
**Example**:
```python
from Q_Sea_Battle.reference_performance_utilities import expected_win_rate_simple

# 4x4 field, communicate 4 cells, unbiased enemies, noiseless channel
print(expected_win_rate_simple(field_size=4, comms_size=4, enemy_probability=0.5, channel_noise=0.0))
```

#### `expected_win_rate_majority(field_size: int, comms_size: int, enemy_probability: Number = 0.5, channel_noise: Number = 0.0) -> float`

**Signature**: `expected_win_rate_majority(field_size: int, comms_size: int, enemy_probability: Number = 0.5, channel_noise: Number = 0.0) -> float`  
**Purpose**: Compute an analytic expected win rate for `MajorityPlayers` under a block-majority communication model with a binary symmetric channel, averaging over random fields and a random queried cell index.  
**Arguments**: `field_size` (int): Field side length; must be `>= 1`. `comms_size` (int): Number of blocks `m`; must satisfy `1 <= comms_size <= field_size**2`, and must divide `field_size**2` (i.e., `field_size**2 % comms_size == 0`). `enemy_probability` (Number): Bernoulli parameter `p` for field cells; must lie in `[0.0, 1.0]`. `channel_noise` (Number): Channel flip probability `c`; must lie in `[0.0, 1.0]`.  
**Returns**: `float`: Average success probability.  
**Errors**: `ValueError`: If `field_size < 1`, if `comms_size` is outside `[1, field_size**2]`, if `field_size**2` is not divisible by `comms_size`, if `enemy_probability` is not in `[0.0, 1.0]`, or if `channel_noise` is not in `[0.0, 1.0]`.  
**Example**:
```python
from Q_Sea_Battle.reference_performance_utilities import expected_win_rate_majority

# 4x4 field (16 cells), 4 blocks => block length L=4
print(expected_win_rate_majority(field_size=4, comms_size=4, enemy_probability=0.5, channel_noise=0.1))
```

#### `expected_win_rate_assisted(field_size: int, comms_size: int, enemy_probability: Number = 0.5, channel_noise: Number = 0.0, p_high: Number = 0.9) -> float`

**Signature**: `expected_win_rate_assisted(field_size: int, comms_size: int, enemy_probability: Number = 0.5, channel_noise: Number = 0.0, p_high: Number = 0.9) -> float`  
**Purpose**: Compute an analytic expected win rate for classical `AssistedPlayers` under a one-bit communication model; currently restricted to `comms_size == 1` and requires `field_size**2` be a power of two.  
**Arguments**: `field_size` (int): Field side length; must be `>= 1`; additionally, `field_size**2` must be a power of two. `comms_size` (int): Communication size; must equal `1`. `enemy_probability` (Number): Present in signature but not used by the implementation. `channel_noise` (Number): Channel flip probability `c`; must lie in `[0.0, 1.0]`. `p_high` (Number): Parameter used in computing the ideal success expression; must lie in `[0.0, 1.0]`.  
**Returns**: `float`: Expected success probability, clamped to `[0.0, 1.0]`.  
**Errors**: `ValueError`: If `field_size < 1`, if `comms_size != 1`, if `field_size**2` is not a power of two, if `channel_noise` is not in `[0.0, 1.0]`, or if `p_high` is not in `[0.0, 1.0]`.  
**Example**:
```python
from Q_Sea_Battle.reference_performance_utilities import expected_win_rate_assisted

# field_size=2 => n2=4 is a power of two
print(expected_win_rate_assisted(field_size=2, comms_size=1, channel_noise=0.05, p_high=0.9))
```

#### `limit_from_mutual_information(field_size: int, comms_size: int, channel_noise: Number = 0.0, accuracy_in_digits: int = 8) -> float`

**Signature**: `limit_from_mutual_information(field_size: int, comms_size: int, channel_noise: Number = 0.0, accuracy_in_digits: int = 8) -> float`  
**Purpose**: Compute the maximum allowed success probability under Information Causality, given a field size, a communication budget, and an optional binary symmetric channel noise parameter; uses `binary_entropy` and `binary_entropy_reverse` to map an effective communication rate to a success bound.  
**Arguments**: `field_size` (int): Field side length; must be `>= 1`. `comms_size` (int): Communication size `m`; must satisfy `0 <= comms_size <= field_size**2`. `channel_noise` (Number): Channel flip probability `c`; must lie in `[0.0, 1.0]`. `accuracy_in_digits` (int): Passed through to `binary_entropy_reverse`.  
**Returns**: `float`: Success-probability bound in `[0.5, 1.0]` (as implemented, returns `0.5` in several limiting cases and `1.0` when effective communication exceeds the field size).  
**Errors**: `ValueError`: If `field_size < 1`, if `comms_size` is not in `[0, field_size**2]`, or if `channel_noise` is not in `[0.0, 1.0]`. `ValueError`/`RuntimeError`: May propagate from `binary_entropy_reverse` depending on internal target and convergence.  
**Example**:
```python
from Q_Sea_Battle.reference_performance_utilities import limit_from_mutual_information

print(limit_from_mutual_information(field_size=4, comms_size=0))         # 0.5
print(limit_from_mutual_information(field_size=4, comms_size=16))        # 1.0
print(limit_from_mutual_information(field_size=4, comms_size=4, channel_noise=0.1))
```

### Constants

Not specified.

### Types

#### `Number`

**Definition**: `Number = Union[float, int]`  
**Purpose**: Convenience type alias for numeric parameters accepted by the module’s public functions.

## Dependencies

- Standard library: `math`, `typing.Union`, `__future__.annotations`

## Planned (design-spec)

Not specified.

## Deviations

- `expected_win_rate_assisted` includes an `enemy_probability` parameter but does not use it in the current implementation.
- `expected_win_rate_assisted` explicitly restricts `comms_size` to `1` (no support for other values).
- `expected_win_rate_majority` imports `math` inside the function even though `math` is already imported at module scope.

## Notes for Contributors

- Keep numeric stability in mind: `expected_win_rate_majority` uses `lgamma` and log-space computation for the binomial PMF to reduce overflow/underflow risks.
- If extending `expected_win_rate_assisted` beyond `comms_size == 1`, update input validation and document the new model assumptions.
- When changing tolerances or iteration counts in `binary_entropy_reverse`, ensure callers relying on `accuracy_in_digits` remain consistent.

## Related

- Not specified.

## Changelog

- Version 0.1: Initial implementation (per module docstring).