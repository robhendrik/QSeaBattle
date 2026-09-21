"""Discretize / Regularize Unit (DRU) utilities for communicating agents.

This module provides the Discretize / Regularize Unit (DRU) transforms used
in DIAL-style training for communication between agents.

Two mappings are exposed:

- ``dru_train``: A differentiable mapping for centralized training. It adds
  Gaussian noise in logit space and applies a logistic nonlinearity to obtain
  continuous values in (0, 1).
- ``dru_execute``: A non-differentiable mapping for decentralized execution.
  It thresholds logits to produce discrete bits.

The DRU is parameter-free; its behavior is fully determined by the input
logits and the provided noise/threshold settings.
"""

from __future__ import annotations

from typing import Any, Tuple, Union

import numpy as np
import tensorflow as tf
import sys
sys.path.append("./src")
from Q_Sea_Battle.logit_utilities import logit_to_prob


ArrayLike = Union[float, np.ndarray, tf.Tensor]


def _is_tf_tensor(x: Any) -> bool:
    """Return whether ``x`` is a TensorFlow tensor."""
    return tf.is_tensor(x)


def dru_train(
    message_logits: ArrayLike,
    sigma: float = 2.0,
    clip_range: Tuple[float, float] | None = (-10.0, 10.0),
) -> ArrayLike:
    """Apply the differentiable DRU mapping used during centralized training.

    The transform follows the common DIAL formulation:

        DRU(m) = logistic(m + eps),   eps ~ Normal(0, sigma)

    where ``m`` are message logits. The output is a continuous relaxation of
    binary communication bits, enabling gradient-based learning.

    Notes:
        - Randomness comes from ``np.random`` or ``tf.random`` depending on the
          input type. Reproducibility therefore depends on global seeds set
          elsewhere in the program.
        - For TensorFlow inputs, this function is suitable for use inside a
          graph; gradients flow w.r.t. ``message_logits``.

    Args:
        message_logits: Message logits. May be a scalar, NumPy array, or
            TensorFlow tensor.
        sigma: Standard deviation of the additive Gaussian noise in logit
            space. Must be non-negative. A value of 0 disables noise.
        clip_range: Optional ``(min, max)`` range used to clip the noisy logits
            before applying the logistic, primarily to avoid numerical issues
            for extreme logits. If ``None``, no clipping is applied.

    Returns:
        Values in (0, 1) with the same shape as ``message_logits``. The return
        type matches the input family (NumPy vs TensorFlow).

    Raises:
        ValueError: If ``sigma`` is negative.
    """
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative.")

    # TensorFlow path: keep everything as tensors for gradient flow.
    if _is_tf_tensor(message_logits):
        logits = tf.cast(message_logits, tf.float32)

        if sigma > 0.0:
            noise = tf.random.normal(tf.shape(logits), mean=0.0, stddev=sigma)
            logits = logits + noise

        if clip_range is not None:
            lo, hi = clip_range
            logits = tf.clip_by_value(logits, lo, hi)

        # Sigmoid is the logistic function in TensorFlow; it maps logits to
        # probabilities in a numerically stable way.
        probs = tf.nn.sigmoid(logits)
        return probs

    # NumPy path: operations are done in NumPy but follow the same formula.
    logits_np = np.asarray(message_logits, dtype=np.float32)

    if sigma > 0.0:
        noise_np = np.random.normal(loc=0.0, scale=sigma, size=logits_np.shape)
        logits_np = logits_np + noise_np

    if clip_range is not None:
        lo, hi = clip_range
        logits_np = np.clip(logits_np, lo, hi)

    # Use the shared helper to keep probability computations consistent across
    # the code base.
    probs_np = logit_to_prob(logits_np)
    return probs_np


def dru_execute(
    message_logits: ArrayLike,
    threshold: float = 0.0,
) -> ArrayLike:
    """Apply the discrete DRU mapping used during decentralized execution.

    This mapping converts logits to hard binary decisions by thresholding
    element-wise in logit space:

        bit = 1 if logit > threshold else 0

    A threshold of 0.0 corresponds to thresholding at probability 0.5 since
    ``logistic(0) = 0.5``.

    Args:
        message_logits: Message logits. May be a scalar, NumPy array, or
            TensorFlow tensor.
        threshold: Logit threshold used to produce discrete bits.

    Returns:
        Discrete bits with the same shape as ``message_logits``.

        - NumPy input: ``np.ndarray`` of ``int`` values in ``{0, 1}``.
        - TensorFlow input: ``tf.Tensor`` of dtype ``tf.float32`` with values
          in ``{0.0, 1.0}``. This path is intended for inference/execution
          rather than gradient-based optimization.
    """
    if _is_tf_tensor(message_logits):
        logits = tf.cast(message_logits, tf.float32)
        bits = tf.cast(logits > threshold, tf.float32)
        return bits

    logits_np = np.asarray(message_logits, dtype=np.float32)
    bits_np = (logits_np > threshold).astype(int)
    return bits_np