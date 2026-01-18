# dru_utilities
> Role: DRU (Discretize / Regularize Unit) utilities for transforming communication message logits into differentiable probabilities during training and discrete bits during execution.
Location: `Q_Sea_Battle.dru_utilities`

## Overview

This module implements helper functions for the Discretize / Regularize Unit (DRU) used in DIAL-style training of communicating agents. It provides a differentiable mapping (`dru_train`) that adds Gaussian noise and applies a logistic nonlinearity during centralized training, and a discretizing mapping (`dru_execute`) that hard-thresholds logits to bits for decentralized execution. The module is intentionally free of trainable parameters and supports both NumPy arrays and TensorFlow tensors.

## Public API

### Functions

#### `_is_tf_tensor(x: Any) -> bool`

**Signature:** `_is_tf_tensor(x: Any) -> bool`  
**Purpose:** Return `True` if `x` is a TensorFlow tensor.  
**Arguments:**  
- `x` (`Any`): Value to test.  
**Returns:**  
- `bool`: `True` if `tf.is_tensor(x)` is `True`, otherwise `False`.  
**Errors:**  
- Not specified.  
**Example:**
```python
import tensorflow as tf
from Q_Sea_Battle.dru_utilities import _is_tf_tensor

x = tf.constant([1.0, 2.0])
assert _is_tf_tensor(x) is True
```

#### `dru_train(message_logits: ArrayLike, sigma: float = 2.0, clip_range: Tuple[float, float] | None = (-10.0, 10.0)) -> ArrayLike`

**Signature:** `dru_train(message_logits: ArrayLike, sigma: float = 2.0, clip_range: Tuple[float, float] | None = (-10.0, 10.0)) -> ArrayLike`  
**Purpose:** Apply the differentiable DRU mapping used during centralized training: additive Gaussian noise on logits followed by a logistic/sigmoid transformation, optionally clipping the noisy logits for numerical stability.  
**Arguments:**  
- `message_logits` (`ArrayLike`): Logits for communication dimensions; may be a scalar, NumPy array, or TensorFlow tensor of shape `(..., m)`.  
- `sigma` (`float`, default `2.0`): Standard deviation of Gaussian noise added to logits; must be non-negative.  
- `clip_range` (`Tuple[float, float] | None`, default `(-10.0, 10.0)`): Optional `(min, max)` to clip noisy logits before applying the logistic; if `None`, no clipping is applied.  
**Returns:**  
- `ArrayLike`: Same type and shape as `message_logits`, with values in `(0, 1)`; TensorFlow outputs are differentiable with respect to `message_logits`.  
**Errors:**  
- `ValueError`: If `sigma < 0.0`.  
**Example:**
```python
import numpy as np
import tensorflow as tf
from Q_Sea_Battle.dru_utilities import dru_train

# NumPy usage
logits_np = np.array([0.0, 2.0, -2.0], dtype=np.float32)
probs_np = dru_train(logits_np, sigma=0.0)  # deterministic sigmoid
print(probs_np)

# TensorFlow usage (differentiable)
logits_tf = tf.constant([[0.0, 1.0, -1.0]], dtype=tf.float32)
with tf.GradientTape() as tape:
    tape.watch(logits_tf)
    probs_tf = dru_train(logits_tf, sigma=0.0)
grads = tape.gradient(probs_tf, logits_tf)
print(probs_tf, grads)
```

#### `dru_execute(message_logits: ArrayLike, threshold: float = 0.0) -> ArrayLike`

**Signature:** `dru_execute(message_logits: ArrayLike, threshold: float = 0.0) -> ArrayLike`  
**Purpose:** Apply the discretizing DRU mapping used during decentralized execution: element-wise hard thresholding of logits to bits.  
**Arguments:**  
- `message_logits` (`ArrayLike`): Logits for communication dimensions; may be a scalar, NumPy array, or TensorFlow tensor of shape `(..., m)`.  
- `threshold` (`float`, default `0.0`): Logit-space threshold used to produce discrete bits; `0.0` corresponds to probability threshold `0.5`.  
**Returns:**  
- `ArrayLike`: For NumPy inputs, a NumPy array of `int` with values in `{0, 1}` and the same shape as `message_logits`. For TensorFlow inputs, a `tf.Tensor` of `tf.float32` with values in `{0.0, 1.0}`.  
**Errors:**  
- Not specified.  
**Example:**
```python
import numpy as np
import tensorflow as tf
from Q_Sea_Battle.dru_utilities import dru_execute

logits_np = np.array([-0.1, 0.0, 0.2], dtype=np.float32)
bits_np = dru_execute(logits_np, threshold=0.0)
print(bits_np)  # [0 0 1]

logits_tf = tf.constant([-0.1, 0.0, 0.2], dtype=tf.float32)
bits_tf = dru_execute(logits_tf, threshold=0.0)
print(bits_tf)  # tf.Tensor([0. 0. 1.], shape=(3,), dtype=float32)
```

### Constants

- None.

### Types

- `ArrayLike = Union[float, np.ndarray, tf.Tensor]`

## Dependencies

- `numpy` (`np`): Used for NumPy-based computation and sampling Gaussian noise in the NumPy path.
- `tensorflow` (`tf`): Used for TensorFlow-based computation, noise sampling in the TensorFlow path, and differentiable sigmoid mapping.
- `typing`: Uses `Any`, `Tuple`, and `Union` for type annotations.
- `Q_Sea_Battle.logit_utilities.logit_to_prob`: Used to convert logits to probabilities in the NumPy path (and referenced in documentation for consistency).

## Planned (design-spec)

- Not specified.

## Deviations

- The module docstring states that the logistic nonlinearity is implemented via `Q_Sea_Battle.logit_utilities.logit_to_prob` for consistency, but the TensorFlow path uses `tf.nn.sigmoid` directly (documented as equivalent to `logit_to_prob` when using logits).

## Notes for Contributors

- Randomness/reproducibility: `dru_train` relies on global seeds for `np.random` and `tf.random` set elsewhere; tests should set seeds explicitly when deterministic behavior is required.
- Keep TensorFlow operations on-tensor in the TensorFlow path to preserve gradient flow through `message_logits`.
- Avoid adding trainable parameters to this module; it is intended to be a fixed transformation given inputs and noise settings.

## Related

- `Q_Sea_Battle.logit_utilities.logit_to_prob` (used for stable/log-consistent probability computations in the NumPy path).

## Changelog

- 0.1: Initial implementation of DRU utilities (`dru_train`, `dru_execute`) and helper `_is_tf_tensor`.