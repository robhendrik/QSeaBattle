# dru_utilities

> Role: Discretize / Regularize Unit (DRU) utilities for transforming agent communication logits during training (differentiable) and execution (discrete).

Location: `Q_Sea_Battle.dru_utilities`

## Overview

This module implements Discretize / Regularize Unit (DRU) transforms used in DIAL-style communication learning. It exposes a differentiable mapping for centralized training (`dru_train`) that adds Gaussian noise in logit space and applies a logistic nonlinearity, and a non-differentiable mapping for decentralized execution (`dru_execute`) that thresholds logits into hard bits. The DRU is parameter-free; behavior depends on the input logits and the provided noise/threshold settings.

## Public API

### Functions

#### `_is_tf_tensor(x: Any) -> bool`

**Signature:** `_is_tf_tensor(x: Any) -> bool`  
**Purpose:** Return whether `x` is a TensorFlow tensor.  
**Arguments:** `x` (Any): Value to test.  
**Returns:** `bool`: `True` if `x` is a TensorFlow tensor; otherwise `False`.  
**Errors:** Not specified.  
**Example:**
```python
import tensorflow as tf
from Q_Sea_Battle.dru_utilities import _is_tf_tensor

_is_tf_tensor(tf.constant([1.0, 2.0]))  # True
_is_tf_tensor([1.0, 2.0])               # False
```

#### `dru_train(message_logits: ArrayLike, sigma: float = 2.0, clip_range: Tuple[float, float] | None = (-10.0, 10.0)) -> ArrayLike`

**Signature:** `dru_train(message_logits: ArrayLike, sigma: float = 2.0, clip_range: Tuple[float, float] | None = (-10.0, 10.0)) -> ArrayLike`  
**Purpose:** Apply the differentiable DRU mapping used during centralized training by adding Gaussian noise in logit space and applying a logistic (sigmoid) nonlinearity to produce continuous values in `(0, 1)`.  
**Arguments:** `message_logits` (ArrayLike): Message logits; may be a scalar, NumPy array, or TensorFlow tensor.  
**Arguments:** `sigma` (float): Standard deviation of additive Gaussian noise in logit space; must be non-negative; `0` disables noise.  
**Arguments:** `clip_range` (Tuple[float, float] | None): Optional `(min, max)` range to clip noisy logits before applying the logistic; if `None`, no clipping is applied.  
**Returns:** `ArrayLike`: Values in `(0, 1)` with the same shape as `message_logits`; return type matches the input family (NumPy vs TensorFlow).  
**Errors:** `ValueError`: If `sigma` is negative.  
**Example:**
```python
import numpy as np
import tensorflow as tf
from Q_Sea_Battle.dru_utilities import dru_train

# NumPy: continuous relaxation in (0, 1)
logits_np = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
probs_np = dru_train(logits_np, sigma=0.0)
print(probs_np)

# TensorFlow: differentiable path
logits_tf = tf.constant([-2.0, 0.0, 2.0], dtype=tf.float32)
probs_tf = dru_train(logits_tf, sigma=1.0, clip_range=(-10.0, 10.0))
print(probs_tf)
```

#### `dru_execute(message_logits: ArrayLike, threshold: float = 0.0) -> ArrayLike`

**Signature:** `dru_execute(message_logits: ArrayLike, threshold: float = 0.0) -> ArrayLike`  
**Purpose:** Apply the discrete DRU mapping used during decentralized execution by thresholding logits element-wise into hard binary bits.  
**Arguments:** `message_logits` (ArrayLike): Message logits; may be a scalar, NumPy array, or TensorFlow tensor.  
**Arguments:** `threshold` (float): Logit threshold used to produce discrete bits.  
**Returns:** `ArrayLike`: Discrete bits with the same shape as `message_logits`; NumPy input returns `np.ndarray` of `int` values in `{0, 1}`, TensorFlow input returns `tf.Tensor` of dtype `tf.float32` with values in `{0.0, 1.0}`.  
**Errors:** Not specified.  
**Example:**
```python
import numpy as np
import tensorflow as tf
from Q_Sea_Battle.dru_utilities import dru_execute

logits_np = np.array([-0.1, 0.0, 0.1], dtype=np.float32)
bits_np = dru_execute(logits_np, threshold=0.0)  # [0, 0, 1]
print(bits_np)

logits_tf = tf.constant([-0.1, 0.0, 0.1], dtype=tf.float32)
bits_tf = dru_execute(logits_tf, threshold=0.0)  # [0.0, 0.0, 1.0]
print(bits_tf)
```

### Constants

No public constants are defined in this module.

### Types

#### `ArrayLike`

**Definition:** `ArrayLike = Union[float, np.ndarray, tf.Tensor]`  
**Purpose:** Input/output type for DRU functions; supports scalar floats, NumPy arrays, and TensorFlow tensors.

## Dependencies

- `numpy` (`np`): Used for NumPy execution path, noise sampling, clipping, and array conversion.  
- `tensorflow` (`tf`): Used for TensorFlow execution path, noise sampling, clipping, and sigmoid.  
- `typing`: Uses `Any`, `Tuple`, and `Union` for type annotations.  
- `Q_Sea_Battle.logit_utilities.logit_to_prob`: Used to compute probabilities from logits on the NumPy path.  
- `sys`: Used to append `"./src"` to `sys.path` (module-level side effect).

## Planned (design-spec)

Not specified.

## Deviations

- The module mutates `sys.path` at import time by appending `"./src"`, which is a global side effect and may affect import resolution outside this module.  
- `dru_train` uses `tf.nn.sigmoid` for TensorFlow inputs but uses `logit_to_prob` for NumPy inputs; numerical equivalence depends on the implementation of `logit_to_prob` (not specified here).  
- `dru_execute` returns different dtypes by backend (NumPy `int` vs TensorFlow `tf.float32`), which callers may need to normalize.

## Notes for Contributors

- Reproducibility: `dru_train` draws randomness from `np.random` or `tf.random` depending on the input type; seeding must be handled externally via NumPy and TensorFlow global seeds.  
- Gradient flow: Only the TensorFlow path in `dru_train` is intended to be differentiable with respect to `message_logits`; `dru_execute` is intended for inference/execution.  
- If modifying clipping behavior, ensure the default `clip_range` remains compatible with both TensorFlow and NumPy paths and update both implementations consistently.

## Related

- `Q_Sea_Battle.logit_utilities.logit_to_prob`: Utility used to map logits to probabilities on the NumPy path.  
- DIAL-style communication training: The DRU mapping is described as following a common DIAL formulation (details not specified in this module).

## Changelog

- Unknown: Initial version information not specified in the module.