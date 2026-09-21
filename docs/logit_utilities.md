# Q_Sea_Battle.logit_utilities

> Role: NumPy-based helpers for numerically stable conversions between Bernoulli logits and (log-)probabilities.

Location: `Q_Sea_Battle.logit_utilities`

## Overview

This module provides utilities for working with Bernoulli distributions parameterized by logits (pre-sigmoid activations), with attention to numerical stability for large-magnitude values. Public functions accept Python scalars or NumPy arrays and will return a Python `float` when all relevant inputs are scalar-like.

## Public API

### Functions

#### `logit_to_prob(logits: ArrayLike) -> ArrayLike`

**Signature:** `logit_to_prob(logits: ArrayLike) -> ArrayLike`  
**Purpose:** Convert Bernoulli logits to probabilities using a numerically stable sigmoid implementation that avoids overflow/underflow for large `|logits|`.  
**Arguments:**  
- `logits`: Scalar or array-like of logits (pre-sigmoid activations).  
**Returns:** Probabilities in `[0.0, 1.0]` with the same shape as `logits`; returns a Python `float` if the input was scalar-like.  
**Errors:** Not specified.  
**Example:**
```python
import numpy as np
from Q_Sea_Battle.logit_utilities import logit_to_prob

print(logit_to_prob(0.0))                 # 0.5
print(logit_to_prob(np.array([-10, 0, 10], dtype=float)))
```

#### `logit_to_logprob(logits: ArrayLike, actions: ArrayLike) -> ArrayLike`

**Signature:** `logit_to_logprob(logits: ArrayLike, actions: ArrayLike) -> ArrayLike`  
**Purpose:** Compute Bernoulli log-probabilities `log π(a | z)` for actions `a ∈ {0, 1}` from logits `z` without explicitly forming probabilities, using a stable softplus-based identity.  
**Arguments:**  
- `logits`: Scalar or array-like of logits.  
- `actions`: Scalar or array-like broadcastable to `logits`; values must be exactly `0` or `1` (after conversion to `float64`).  
**Returns:** Log-probabilities with the broadcasted shape of `logits` and `actions`; returns a Python `float` only when both inputs were scalar-like.  
**Errors:**  
- `ValueError`: If `logits` and `actions` are not broadcastable to a common shape.  
- `ValueError`: If `actions` contains values other than `0` or `1`.  
**Example:**
```python
import numpy as np
from Q_Sea_Battle.logit_utilities import logit_to_logprob

z = np.array([-2.0, 0.0, 2.0])
a = np.array([0, 1, 1])
lp = logit_to_logprob(z, a)
print(lp)

print(logit_to_logprob(1.5, 1))  # scalar in, scalar out
```

### Constants

Not specified.

### Types

#### `ArrayLike`

**Definition:** `ArrayLike = Union[float, int, np.ndarray]`  
**Purpose:** Public input/output type for functions accepting either Python scalars (`float`, `int`) or NumPy arrays.

## Dependencies

- `numpy` (imported as `np`)
- `typing.Union`

## Planned (design-spec)

Not specified.

## Deviations

Not specified.

## Notes for Contributors

- This module contains private helpers (`_to_array_and_flag`, `_from_array`, `_softplus`) used to ensure `float64` computations and stable transformations; they are not part of the public API and should not be relied upon by external code.
- `logit_to_logprob` enforces actions to be exactly `0` or `1` after conversion to `float64`; changing this behavior would be a breaking change for validation expectations.

## Related

- NumPy broadcasting semantics (`numpy.broadcast_arrays`)
- Logistic sigmoid and softplus functions used in stable probability computations

## Changelog

Not specified.