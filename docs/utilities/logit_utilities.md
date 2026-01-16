# Module logit_utilities

**Module import path**: `Q_Sea_Battle.logit_utilities`

Utilities for numerically stable conversions between logits, probabilities,
and log-probabilities for Bernoulli variables.

This module is foundational for communication, imitation learning, and
policy evaluation utilities across QSeaBattle.

---

## Overview

This module provides stable, NumPy-based helpers for:

- Converting logits to probabilities.
- Computing log-probabilities directly from logits and actions.
- Supporting both scalar and array-based workflows without numerical
  overflow.

All functions are deterministic and free of trainable parameters.

---

## Functions

### logit_to_prob

Convert Bernoulli logits to probabilities in a numerically stable way.

#### Signature

```
logit_to_prob(
    logits: float | int | np.ndarray,
) -> float | np.ndarray
```

#### Parameters

- **logits**  
  Scalar or array-like of Bernoulli logits.  
  Type: `float | int | np.ndarray, dtype float64, shape (...)`

#### Returns

- Probabilities with the same shape as `logits` and values in `[0.0, 1.0]`.  
  If the input was scalar, a Python `float` is returned.

#### Preconditions

- None.

#### Postconditions

- Output values lie in the closed interval `[0.0, 1.0]`.
- Shape of the output matches the shape of `logits`.

#### Errors

- None.

#### Example

```python
p = logit_to_prob(0.0)
```

---

### logit_to_logprob

Compute the Bernoulli log-probability `log π(a | logits)` in closed form.

#### Signature

```
logit_to_logprob(
    logits: float | int | np.ndarray,
    actions: float | int | np.ndarray,
) -> float | np.ndarray
```

#### Parameters

- **logits**  
  Scalar or array-like of Bernoulli logits.  
  Type: `float | int | np.ndarray, dtype float64, shape (...)`

- **actions**  
  Scalar or array-like Bernoulli outcomes.  
  Type: `float | int | np.ndarray, dtype float64 {0,1}, shape (...)`

#### Returns

- Log-probabilities with the broadcasted shape of `logits` and `actions`.  
  If both inputs were scalar, a Python `float` is returned.

#### Preconditions

- `logits` and `actions` must be broadcastable to a common shape.
- `actions` must take values in `(0, 1)`.

#### Postconditions

- Returned values are finite real numbers.
- Shape matches NumPy broadcast rules for the inputs.

#### Errors

- `ValueError` if inputs cannot be broadcast.
- `ValueError` if `actions` contains values outside `(0, 1)`.

#### Example

```python
lp = logit_to_logprob(logits=0.3, actions=1)
```

---

## Testing Hooks

Suggested invariants for testing:

- `logit_to_prob(0.0) == 0.5`.
- `logit_to_prob(z)` is monotonic increasing in `z`.
- For `a in {0,1}`, `exp(logit_to_logprob(z, a))` equals the Bernoulli
  probability implied by `z`.
- Scalar inputs return Python `float`, not NumPy arrays.

---

## Notes for Contributors

- Maintain numerical stability for large `|logits|`.
- Do not introduce TensorFlow or PyTorch dependencies in this module.
- Any new helpers must preserve scalar-vs-array roundtripping behavior.

---

## Changelog

- 2026-01-16 — Initial specification page. (Rob Hendriks)
