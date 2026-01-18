
"""Utilities for stable conversions between logits, probabilities and log-probabilities.

Author: Rob Hendriks
Package: Q_Sea_Battle
Version: 0.1
"""

from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[float, int, np.ndarray]


def _to_array_and_flag(x: ArrayLike) -> tuple[np.ndarray, bool]:
    """Convert input to a NumPy array and indicate whether it was scalar.

    Args:
        x: Scalar or array-like input.

    Returns:
        A tuple ``(arr, is_scalar)`` where ``arr`` is a NumPy array with
        ``dtype=float64`` and ``is_scalar`` indicates whether the original
        input was a scalar.
    """
    arr = np.asarray(x, dtype=np.float64)
    is_scalar = arr.shape == ()  # 0-dim array
    return arr, is_scalar


def _from_array(arr: np.ndarray, is_scalar: bool) -> ArrayLike:
    """Convert a NumPy array back to scalar if needed.

    Args:
        arr: NumPy array result.
        is_scalar: Whether the original input to the public function was scalar.

    Returns:
        A Python float if ``is_scalar`` is True, otherwise the array itself.
    """
    if is_scalar:
        return float(arr)
    return arr


def _softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus implementation.

    softplus(x) = log(1 + exp(x))

    This implementation avoids overflow for large |x| by using the identity:

        softplus(x) = max(x, 0) + log1p(exp(-|x|))

    Args:
        x: NumPy array of any shape.

    Returns:
        NumPy array of the same shape containing the softplus values.
    """
    x = np.asarray(x, dtype=np.float64)
    abs_x = np.abs(x)
    # log1p(exp(-abs_x)) is safe because -abs_x <= 0
    return np.maximum(x, 0.0) + np.log1p(np.exp(-abs_x))


def logit_to_prob(logits: ArrayLike) -> ArrayLike:
    """Convert Bernoulli logits to probabilities in a numerically stable way.

    The logit ``z`` is related to the probability ``p`` by::

        p = 1 / (1 + exp(-z))

    This function supports both scalars and NumPy arrays and uses a
    formulation that avoids overflow for large |z|.

    Args:
        logits: Scalar or array-like of pre-sigmoid activations.

    Returns:
        Probabilities with the same shape as ``logits`` and values in [0.0, 1.0].
        If the input was a scalar, a Python ``float`` is returned.
    """
    z, is_scalar = _to_array_and_flag(logits)

    # Stable sigmoid implementation:
    # For z >= 0: sigmoid(z) = 1 / (1 + exp(-z))
    # For z <  0: sigmoid(z) = exp(z) / (1 + exp(z))
    positive = z >= 0
    negative = ~positive

    out = np.empty_like(z, dtype=np.float64)

    # For positive z
    if np.any(positive):
        zp = z[positive]
        exp_neg = np.exp(-zp)
        out[positive] = 1.0 / (1.0 + exp_neg)

    # For negative z
    if np.any(negative):
        zn = z[negative]
        exp_pos = np.exp(zn)
        out[negative] = exp_pos / (1.0 + exp_pos)

    return _from_array(out, is_scalar)


def logit_to_logprob(logits: ArrayLike, actions: ArrayLike) -> ArrayLike:
    """Compute log-probability log π(a | logits) for Bernoulli actions.

    This function implements a numerically stable closed-form expression
    that avoids explicitly computing probabilities or taking ``log(0)``.
    For a Bernoulli variable with logit ``z`` and action ``a ∈ {0, 1}``, we use::

        log π(a | z) = -softplus((1 - 2a) * z)

    where ``softplus(x) = log(1 + exp(x))`` and is implemented in a
    numerically stable way.

    This formula matches the standard parameterisation used in major deep
    learning frameworks (TensorFlow Probability, PyTorch, JAX) for
    `Bernoulli(logits=...)`.

    Args:
        logits: Scalar or array-like of logits.
        actions: Scalar or array-like of the same shape as ``logits`` with
            values in {0, 1} indicating the chosen Bernoulli outcome.

    Returns:
        Log-probabilities with the same shape as the broadcast of
        ``logits`` and ``actions``. If both inputs were scalars, a Python
        ``float`` is returned.

    Raises:
        ValueError: If ``logits`` and ``actions`` cannot be broadcast to a
            common shape, or if ``actions`` contains values other than 0 or 1.
    """
    z, is_scalar_logits = _to_array_and_flag(logits)
    a, is_scalar_actions = _to_array_and_flag(actions)

    # Broadcast to a common shape; NumPy will raise a ValueError if this fails.
    try:
        z_b, a_b = np.broadcast_arrays(z, a)
    except ValueError as exc:
        raise ValueError(
            "logits and actions must be broadcastable to the same shape"
        ) from exc

    # Validate that actions are in {0, 1}
    if np.any((a_b != 0.0) & (a_b != 1.0)):
        raise ValueError("actions must be in {0, 1}")

    # t = (1 - 2a) * z  ->  t = z if a=0, t = -z if a=1
    t = (1.0 - 2.0 * a_b) * z_b

    logprob = -_softplus(t)

    # Result is scalar iff both inputs were effectively scalar
    is_scalar = is_scalar_logits and is_scalar_actions

    return _from_array(logprob, is_scalar)
