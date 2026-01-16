# Module reference_performance_utilities

## Overview
Analytic benchmark utilities providing closed-form or numerically stable reference
values for success probabilities and information-theoretic limits in QSeaBattle.

## Module Import Path
`Q_Sea_Battle.reference_performance_utilities`

## Terminology
- `field_size`: linear field dimension.
- `n2 = field_size^2`.
- `comms_size = m`: number of communicated bits.
- Probabilities lie in `[0.0, 1.0]`.

## Public Interface
Functions:
- `binary_entropy`
- `binary_entropy_reverse`
- `expected_win_rate_simple`
- `expected_win_rate_majority`
- `expected_win_rate_assisted`
- `limit_from_mutual_information`

## Preconditions
- `field_size >= 1`.
- `0 <= comms_size <= n2`.
- All probabilities are in `[0.0, 1.0]`.

## Postconditions
- All returned values lie in `[0.0, 1.0]`.

## Errors
- `ValueError` on invalid domains.
- `RuntimeError` if numeric inversion fails.

## Examples
```python
from Q_Sea_Battle.reference_performance_utilities import expected_win_rate_simple
p = expected_win_rate_simple(field_size=4, comms_size=4)
```

## Testing Hooks
- `binary_entropy(0.5) == 1.0`
- `limit_from_mutual_information(field_size=4, comms_size=0) == 0.5`

## Changelog
- 2026-01-16 (Rob Hendriks): Initial utilities module page.
