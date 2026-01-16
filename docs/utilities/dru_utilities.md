# Module dru_utilities

**Module import path**: `Q_Sea_Battle.dru_utilities`

Utilities implementing the Discretize / Regularize Unit (DRU) used for
training and execution of communicating agents following DIAL-style
protocols.

The DRU is a *fixed*, non-trainable transformation applied to communication
logits. During centralized training it produces continuous values suitable
for gradient-based optimization; during decentralized execution it produces
hard binary messages.

---

## Overview

This module provides two public functions:

- `dru_train`: differentiable DRU mapping for centralized training.
- `dru_execute`: hard discretization for decentralized execution.

The implementation follows the QSeaBattle design document and introduces
no trainable parameters.

---

## Functions

### dru_train

Differentiable DRU mapping used during centralized training.

#### Signature

```
dru_train(
    message_logits: float | np.ndarray | tf.Tensor,
    sigma: float = 2.0,
    clip_range: tuple[float, float] | None = (-10.0, 10.0),
) -> float | np.ndarray | tf.Tensor
```

#### Parameters

- **message_logits**  
  Logits for the communication channel.  
  Type: `float` or `np.ndarray, dtype float32, shape (..., m)`  
  or `tf.Tensor, dtype float32, shape (..., m)`

- **sigma**  
  Standard deviation of additive Gaussian noise applied to the logits
  before the logistic nonlinearity.  
  Type: `float`

- **clip_range**  
  Optional `(min, max)` range used to clip noisy logits before applying
  the logistic, preventing numerical overflow.  
  Type: `tuple[float, float]` or `None`

#### Returns

- Same type and shape as `message_logits`, with values in `(0, 1)`.  
  For TensorFlow inputs, the output is differentiable with respect to
  `message_logits`.

#### Preconditions

- `sigma >= 0.0`.

#### Postconditions

- Output values lie strictly between 0 and 1.
- Shape of the output matches the shape of `message_logits`.

#### Errors

- `ValueError` if `sigma < 0.0`.

#### Example

```python
probs = dru_train(logits, sigma=1.5)
```

---

### dru_execute

Discretizing DRU mapping used during decentralized execution.

#### Signature

```
dru_execute(
    message_logits: float | np.ndarray | tf.Tensor,
    threshold: float = 0.0,
) -> float | np.ndarray | tf.Tensor
```

#### Parameters

- **message_logits**  
  Logits for the communication channel.  
  Type: `float` or `np.ndarray, dtype float32, shape (..., m)`  
  or `tf.Tensor, dtype float32, shape (..., m)`

- **threshold**  
  Threshold in logit space. Values strictly greater than the threshold
  are mapped to 1, others to 0.  
  Type: `float`

#### Returns

- For NumPy inputs:  
  `np.ndarray, dtype int {0,1}, shape (..., m)`

- For TensorFlow inputs:  
  `tf.Tensor, dtype float32 {0.0,1.0}, shape (..., m)`

#### Preconditions

- None.

#### Postconditions

- Output values are binary.
- Shape of the output matches the shape of `message_logits`.

#### Errors

- None.

#### Example

```python
bits = dru_execute(logits)
```

---

## Testing Hooks

Suggested invariants for testing:

- With `sigma = 0.0`, `dru_train(logits)` equals `sigmoid(logits)`.
- For large positive logits, `dru_execute` returns all ones.
- For large negative logits, `dru_execute` returns all zeros.
- Output shapes always match input shapes.

---

## Notes for Contributors

- This module must remain free of trainable parameters.
- Changes must preserve compatibility with both NumPy and TensorFlow
  inputs.
- Probability mappings should remain consistent with
  `logit_utilities.logit_to_prob`.

---

## Changelog

- 2026-01-16 — Initial specification page. (Rob Hendriks)
