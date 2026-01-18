"""Discretize / Regularize Unit (DRU) utilities for communicating agents.

This module implements the Discretize / Regularize Unit (DRU) used in
DIAL-style training of communicating agents. It provides helper functions
to transform message logits produced by communication models into
continuous values used during centralized training, and discrete bits
used during decentralized execution.

The implementation follows the specification in the QSeaBattle design
document and is intentionally free of trainable parameters: the DRU is a
fixed, deterministic transformation given the logits and noise settings.

Author: Rob Hendriks
Package: Q_Sea_Battle
Version: 0.1
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
    """Return True if *x* is a TensorFlow tensor."""
    return tf.is_tensor(x)


def dru_train(
    message_logits: ArrayLike,
    sigma: float = 2.0,
    clip_range: Tuple[float, float] | None = (-10.0, 10.0),
) -> ArrayLike:
    """Differentiable DRU mapping used during centralized training.

    Implements the transformation described by Foerster et al.:

        DRU(m) = logistic(N(m, sigma))

    where ``m`` are message logits and ``N(m, sigma)`` denotes additive
    Gaussian noise with standard deviation ``sigma``. Reproducibility 
    depends on global seeds for np.random and tf.random set elsewhere.

    The logistic nonlinearity is implemented via
    :func:`Q_Sea_Battle.logit_utilities.logit_to_prob` so that all probability
    computations in the code base remain consistent.

    The function supports both NumPy arrays and TensorFlow tensors. When
    a TensorFlow tensor is provided, gradients are defined with respect
    to ``message_logits``, making this function suitable for use inside
    computational graphs.

    Args:
        message_logits:
            Logits for the ``m`` communication dimensions. May be a
            scalar, a NumPy array, or a TensorFlow tensor of shape
            ``(..., m)``.
        sigma:
            Standard deviation of the Gaussian noise added to the logits
            before applying the logistic. ``sigma > 0`` encourages the
            logits to move into well-separated modes during training.
        clip_range:
            Optional ``(min, max)`` range to clip the noisy logits
            ``m + epsilon`` before applying the logistic, to avoid
            numerical overflow. If ``None``, no clipping is applied.

    Returns:
        Same type and shape as ``message_logits``, with values in ``(0, 1)``.
        When a TensorFlow tensor is passed in, the returned tensor is
        differentiable with respect to ``message_logits``.
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

        # For tensors we can safely call tf.nn.sigmoid, which is equivalent
        # to logit_to_prob when using logits.
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

    # Use shared helper for numerical stability.
    probs_np = logit_to_prob(logits_np)
    return probs_np


def dru_execute(
    message_logits: ArrayLike,
    threshold: float = 0.0,
) -> ArrayLike:
    """Discretising DRU mapping used during decentralized execution.

    This function applies a hard threshold on the logits:

        DRU(m) = 1 if m > threshold else 0   (element-wise)

    A threshold of ``0.0`` corresponds to a probability threshold of 0.5,
    since ``logistic(0) = 0.5``.

    The function supports both NumPy arrays and TensorFlow tensors.

    Args:
        message_logits:
            Logits for the ``m`` communication dimensions. May be a
            scalar, a NumPy array, or a TensorFlow tensor of shape
            ``(..., m)``.
        threshold:
            Threshold in logit space used to produce discrete bits.
            ``0.0`` corresponds to thresholding at probability 0.5.

    Returns:
        For NumPy inputs, a NumPy array of ``int`` with the same shape
        as ``message_logits`` and values in ``{0, 1}``.

        For TensorFlow inputs, a ``tf.Tensor`` of type ``tf.float32``
        with values in ``{0.0, 1.0}``. Gradients are not intended to
        be used through this function.
    """
    if _is_tf_tensor(message_logits):
        logits = tf.cast(message_logits, tf.float32)
        bits = tf.cast(logits > threshold, tf.float32)
        return bits

    logits_np = np.asarray(message_logits, dtype=np.float32)
    bits_np = (logits_np > threshold).astype(int)
    return bits_np
