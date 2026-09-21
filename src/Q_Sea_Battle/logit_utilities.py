"""Utilities for numerically stable conversions between logits and (log-)probabilities.

This module provides small NumPy-based helpers commonly used with Bernoulli
distributions parameterized by logits (pre-sigmoid activations). All public
functions accept either Python scalars or NumPy arrays.

Notes:
    * "Logits" are real-valued parameters where probabilities are obtained via
      the sigmoid transform.
    * Where possible, computations avoid overflow/underflow for large-magnitude
      logits.
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
        Tuple of ``(arr, is_scalar)`` where ``arr`` is a NumPy array with
        ``dtype=float64`` and ``is_scalar`` is True iff the original input was
        scalar-like (represented as a 0-D array by NumPy).
    """
    arr = np.asarray(x, dtype=np.float64)
    is_scalar = arr.shape == ()  # 0-D array (NumPy's scalar representation)
    return arr, is_scalar


def _from_array(arr: np.ndarray, is_scalar: bool) -> ArrayLike:
    """Convert a NumPy array back to a Python float when the input was scalar.

    Args:
        arr: Computation result as a NumPy array.
        is_scalar: Whether the corresponding original public input was scalar.

    Returns:
        A Python float if ``is_scalar`` is True; otherwise returns ``arr``.
    """
    if is_scalar:
        return float(arr)
    return arr


def _softplus(x: np.ndarray) -> np.ndarray:
    """Compute softplus(x) = log(1 + exp(x)) in a numerically stable way.

    Uses the identity::

        softplus(x) = max(x, 0) + log1p(exp(-abs(x)))

    which avoids overflow in ``exp(x)`` for large positive x and preserves
    precision for large negative x.

    Args:
        x: NumPy array of any shape.

    Returns:
        NumPy array of the same shape with ``dtype=float64``.
    """
    x = np.asarray(x, dtype=np.float64)
    abs_x = np.abs(x)
    # exp(-abs_x) is safe because (-abs_x) <= 0 for all elements.
    return np.maximum(x, 0.0) + np.log1p(np.exp(-abs_x))


def logit_to_prob(logits: ArrayLike) -> ArrayLike:
    """Convert Bernoulli logits to probabilities using a stable sigmoid.

    For logits ``z``, the corresponding probability is::

        p = 1 / (1 + exp(-z))

    The implementation splits on the sign of ``z`` to avoid overflow in the
    exponential when ``|z|`` is large.

    Args:
        logits: Scalar or array-like of logits (pre-sigmoid activations).

    Returns:
        Probabilities in [0.0, 1.0] with the same shape as ``logits``.
        If the input was scalar, returns a Python ``float``.
    """
    z, is_scalar = _to_array_and_flag(logits)

    # Stable sigmoid:
    #   z >= 0: 1 / (1 + exp(-z))  (exp(-z) is in (0, 1])
    #   z <  0: exp(z) / (1 + exp(z))  (avoids exp(-z) overflow)
    positive = z >= 0
    negative = ~positive

    out = np.empty_like(z, dtype=np.float64)

    if np.any(positive):
        zp = z[positive]
        exp_neg = np.exp(-zp)
        out[positive] = 1.0 / (1.0 + exp_neg)

    if np.any(negative):
        zn = z[negative]
        exp_pos = np.exp(zn)
        out[negative] = exp_pos / (1.0 + exp_pos)

    return _from_array(out, is_scalar)


def logit_to_logprob(logits: ArrayLike, actions: ArrayLike) -> ArrayLike:
    """Compute Bernoulli log-probabilities for given actions from logits.

    Computes ``log π(a | z)`` for Bernoulli actions ``a ∈ {0, 1}`` and logits
    ``z`` without explicitly forming probabilities. It uses the stable identity::

        log π(a | z) = -softplus((1 - 2a) * z)

    where ``(1 - 2a) * z`` equals ``z`` when ``a = 0`` and ``-z`` when ``a = 1``.

    Args:
        logits: Scalar or array-like of logits.
        actions: Scalar or array-like broadcastable to ``logits``. Values must be
            exactly 0 or 1 (after conversion to float64).

    Returns:
        Log-probabilities with the broadcasted shape of ``logits`` and
        ``actions``. If both inputs were scalars, returns a Python ``float``.

    Raises:
        ValueError: If ``logits`` and ``actions`` are not broadcastable to a
            common shape, or if ``actions`` contains values other than 0 or 1.
    """
    z, is_scalar_logits = _to_array_and_flag(logits)
    a, is_scalar_actions = _to_array_and_flag(actions)

    # Broadcast to a common shape; NumPy raises ValueError if broadcasting fails.
    try:
        z_b, a_b = np.broadcast_arrays(z, a)
    except ValueError as exc:
        raise ValueError(
            "logits and actions must be broadcastable to the same shape"
        ) from exc

    # Enforce Bernoulli actions in {0, 1}. (Inputs are converted to float64.)
    if np.any((a_b != 0.0) & (a_b != 1.0)):
        raise ValueError("actions must be in {0, 1}")

    # Transform selects the appropriate log-probability branch:
    #   a = 0 -> t = z
    #   a = 1 -> t = -z
    t = (1.0 - 2.0 * a_b) * z_b
    logprob = -_softplus(t)

    # Scalar output only when both original inputs were scalar-like.
    is_scalar = is_scalar_logits and is_scalar_actions
    return _from_array(logprob, is_scalar)