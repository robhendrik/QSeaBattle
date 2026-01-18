# Q_Sea_Battle.logit_utilities

> Role: Utilities for stable conversions between logits, probabilities and log-probabilities.

Location: `Q_Sea_Battle.logit_utilities`

## Overview

This module provides numerically stable utilities for converting Bernoulli logits to probabilities and computing Bernoulli log-probabilities without explicitly forming probabilities that may underflow or overflow.

## Public API

### Functions

#### `logit_to_prob(logits: ArrayLike) -> ArrayLike`

**Signature:** `logit_to_prob(logits: ArrayLike) -> ArrayLike`  
**Purpose:** Convert Bernoulli logits to probabilities in a numerically stable way.  
**Arguments:**  
- `logits`: Scalar or array-like of pre-sigmoid activations.  
**Returns:** Probabilities with the same shape as `logits` and values in `[0.0, 1.0]`; if the input was a scalar, returns a Python `float`.  
**Errors:** Not specified.  
**Example:**
```python
import numpy as np
from Q_Sea_Battle.logit_utilities import logit_to_prob

p0 = logit_to_prob(0.0)              # 0.5
p  = logit_to_prob(np.array([-2, 0, 2], dtype=float))  # array([..., 0.5, ...])
```

#### `logit_to_logprob(logits: ArrayLike, actions: ArrayLike) -> ArrayLike`

**Signature:** `logit_to_logprob(logits: ArrayLike, actions: ArrayLike) -> ArrayLike`  
**Purpose:** Compute log-probability `log π(a | logits)` for Bernoulli actions using a stable closed-form expression `-softplus((1 - 2a) * z)`.  
**Arguments:**  
- `logits`: Scalar or array-like of logits.  
- `actions`: Scalar or array-like of the same shape as `logits` with values in `{0, 1}` indicating the chosen Bernoulli outcome.  
**Returns:** Log-probabilities with the same shape as the broadcast of `logits` and `actions`; if both inputs were scalars, returns a Python `float`.  
**Errors:**  
- `ValueError`: If `logits` and `actions` cannot be broadcast to a common shape.  
- `ValueError`: If `actions` contains values other than `0` or `1`.  
**Example:**
```python
import numpy as np
from Q_Sea_Battle.logit_utilities import logit_to_logprob

lp = logit_to_logprob(0.0, 1)  # log(0.5)

logits = np.array([-1.0, 0.0, 1.0])
actions = np.array([0, 1, 1])
lps = logit_to_logprob(logits, actions)
```

### Constants

Not specified.

### Types

- `ArrayLike`: `typing.Union[float, int, numpy.ndarray]`

## Dependencies

- Python: `__future__.annotations`
- Standard library: `typing.Union`
- Third-party: `numpy` (imported as `np`)

## Planned (design-spec)

Not specified.

## Deviations

Not specified.

## Notes for Contributors

- This module includes internal helpers (`_to_array_and_flag`, `_from_array`, `_softplus`) that implement scalar/array handling and numerical stability; they are intentionally not part of the documented public API.
- When changing numerical formulas, preserve stability for large-magnitude logits and avoid creating intermediate probabilities when computing log-probabilities.

## Related

- NumPy broadcasting rules: used by `np.broadcast_arrays` in `logit_to_logprob`.

## Changelog

- 0.1: Initial version (module docstring).