"""Analytic reference utilities for QSeaBattle performance benchmarks.

This module provides closed-form or numerically-evaluated baseline success
probabilities for several player strategies under simplified assumptions. These
utilities are intended for quick sanity checks and plotting reference curves,
not for simulating full game dynamics.

The probabilities returned by the `expected_win_rate_*` functions are *per-shot*
success probabilities (i.e., the probability that Bob's guess matches the true
cell value for a uniformly random queried cell).

"""

from __future__ import annotations

import math
from typing import Union


Number = Union[float, int]


def binary_entropy(p: Number) -> float:
    """Compute the Shannon binary entropy ``H(p)`` in bits.

    ``H(p)`` is the entropy of a Bernoulli random variable with success
    probability ``p``.

    For ``0 < p < 1``:

        ``H(p) = -p * log2(p) - (1 - p) * log2(1 - p)``

    For ``p <= 0`` or ``p >= 1``, the limiting value ``0.0`` is returned.

    Args:
        p: Bernoulli success probability.

    Returns:
        Binary entropy in bits.
    """
    p = float(p)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def binary_entropy_reverse(H: Number, accuracy_in_digits: int = 8) -> float:
    """Invert binary entropy on the branch ``p in [0.5, 1.0]``.

    This numerically solves for ``p`` such that ``binary_entropy(p) == H``, using
    bisection on the monotone branch ``p ∈ [0.5, 1.0]`` (where entropy decreases
    from 1 to 0).

    Args:
        H: Target entropy in bits. Must lie in ``[0.0, 1.0]``.
        accuracy_in_digits: Target absolute accuracy on entropy, expressed as a
            decimal exponent (tolerance ``10**(-accuracy_in_digits)``).

    Returns:
        The value ``p ∈ [0.5, 1.0]`` that approximately satisfies
        ``binary_entropy(p) == H``.

    Raises:
        ValueError: If ``H`` is outside ``[0.0, 1.0]``.
        RuntimeError: If the bisection loop does not converge within the fixed
            iteration budget.
    """
    H = float(H)
    if H < 0.0 or H > 1.0:
        raise ValueError("H must lie in [0.0, 1.0].")

    if H == 0.0:
        return 1.0
    if H == 1.0:
        return 0.5

    target_tol = 10.0 ** (-accuracy_in_digits)

    lo, hi = 0.5, 1.0
    H_lo = binary_entropy(lo)
    H_hi = binary_entropy(hi)

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        H_mid = binary_entropy(mid)
        if abs(H_mid - H) < target_tol:
            return mid

        # On p in [0.5, 1.0], H(p) is monotone decreasing.
        if H_mid > H:
            lo, H_lo = mid, H_mid
        else:
            hi, H_hi = mid, H_mid

    raise RuntimeError("binary_entropy_reverse did not converge within 200 iterations.")


def expected_win_rate_simple(
    field_size: int,
    comms_size: int,
    enemy_probability: Number = 0.5,
    channel_noise: Number = 0.0,
) -> float:
    """Return the analytic per-shot success probability for the "Simple" strategy.

    This model assumes:
    - The field has ``N = field_size**2`` i.i.d. Bernoulli(``p``) cells.
    - Alice can communicate ``m = comms_size`` exact cell values (a covered subset).
    - For covered cells, the communicated bit is flipped with probability ``c``
      (binary symmetric channel).
    - For uncovered cells, Bob uses the optimal constant guess under symmetry,
      yielding success probability ``p^2 + (1-p)^2`` (probability that Alice and
      Bob's independent samples match).

    Args:
        field_size: Side length of the square field.
        comms_size: Number of cells effectively communicated/covered.
        enemy_probability: Bernoulli parameter ``p`` for a cell being 1.
        channel_noise: Channel flip probability ``c``.

    Returns:
        Expected success probability for a uniformly random queried cell.

    Raises:
        ValueError: If argument constraints are violated.
    """
    if field_size < 1:
        raise ValueError("field_size must be >= 1.")
    n2 = field_size * field_size
    if comms_size < 1 or comms_size > n2:
        raise ValueError("comms_size must satisfy 1 <= comms_size <= field_size**2.")

    p = float(enemy_probability)
    c = float(channel_noise)
    if not (0.0 <= p <= 1.0):
        raise ValueError("enemy_probability must lie in [0.0, 1.0].")
    if not (0.0 <= c <= 1.0):
        raise ValueError("channel_noise must lie in [0.0, 1.0].")

    m = comms_size

    p_cov = 1.0 - c
    p_uncovered = p * p + (1.0 - p) * (1.0 - p)

    frac_cov = m / float(n2)
    return frac_cov * p_cov + (1.0 - frac_cov) * p_uncovered


def expected_win_rate_majority(
    field_size: int,
    comms_size: int,
    enemy_probability: Number = 0.5,
    channel_noise: Number = 0.0,
) -> float:
    """Return the analytic per-shot success probability for the "Majority" strategy.

    Model:
    - The field has ``N = field_size**2`` i.i.d. Bernoulli(``p``) cells.
    - Alice flattens the field and partitions it into ``m = comms_size`` contiguous
      blocks of equal length ``L = N/m``.
    - For each block, Alice sends a single majority bit computed from that block
      (ties are resolved as 1).
    - The communicated bits pass through a binary symmetric channel with flip
      probability ``c``.
    - Bob is queried at a uniformly random cell index, maps that index to its
      block, and outputs the corresponding (possibly flipped) majority bit as his
      guess for the queried cell.

    The returned value averages over both the random field realization and the
    random queried index.

    Args:
        field_size: Side length of the square field.
        comms_size: Number of communicated block-majority bits. Must evenly divide
            ``field_size**2``.
        enemy_probability: Bernoulli parameter ``p`` for a cell being 1.
        channel_noise: Channel flip probability ``c``.

    Returns:
        Expected success probability for a uniformly random queried cell.

    Raises:
        ValueError: If argument constraints are violated.
    """
    import math

    if field_size < 1:
        raise ValueError("field_size must be >= 1.")
    N = field_size * field_size
    m = comms_size
    if m < 1 or m > N:
        raise ValueError("comms_size must satisfy 1 <= comms_size <= field_size**2.")
    if N % m != 0:
        raise ValueError("field_size**2 must be divisible by comms_size.")

    p = float(enemy_probability)
    c = float(channel_noise)
    if not (0.0 <= p <= 1.0):
        raise ValueError("enemy_probability must lie in [0.0, 1.0].")
    if not (0.0 <= c <= 1.0):
        raise ValueError("channel_noise must lie in [0.0, 1.0].")

    # Degenerate fields: block majority is deterministic; only channel flips can fail.
    if p == 0.0 or p == 1.0:
        return 1.0 - c

    L = N // m

    # Precompute logs to evaluate the binomial pmf in log-space (numerical stability).
    log_p = math.log(p)
    log_q = math.log(1.0 - p)

    def binom_prob(L_: int, k_: int) -> float:
        """Return Binomial(L_, p) pmf at k_, computed in log-space."""
        log_prob = (
            math.lgamma(L_ + 1)
            - math.lgamma(k_ + 1)
            - math.lgamma(L_ - k_ + 1)
            + k_ * log_p
            + (L_ - k_) * log_q
        )
        return math.exp(log_prob)

    expected_success = 0.0

    for k in range(L + 1):
        # k is the number of 1s in a block.
        majority_bit = 1 if (k * 2 >= L) else 0  # ties -> 1
        # Given k ones in a block, a uniformly random cell in that block is 1 with k/L.
        p_cell_1 = k / float(L)
        p_cell_0 = 1.0 - p_cell_1

        # Binary symmetric channel: majority_bit is flipped with probability c.
        if majority_bit == 1:
            # Bob outputs 1 with probability (1-c) and 0 with probability c.
            p_success_given_k = (1.0 - c) * p_cell_1 + c * p_cell_0
        else:
            # Bob outputs 0 with probability (1-c) and 1 with probability c.
            p_success_given_k = (1.0 - c) * p_cell_0 + c * p_cell_1

        expected_success += binom_prob(L, k) * p_success_given_k

    return float(expected_success)


def expected_win_rate_assisted(
    field_size: int,
    comms_size: int,
    enemy_probability: Number = 0.5,
    channel_noise: Number = 0.0,
    p_rule: Number = 0.9,
) -> float:
    """Return the analytic per-shot success probability for classical AssistedPlayers.

    Note:
        This function currently implements a one-bit communication setting only
        (``comms_size == 1``) and requires the number of field cells
        ``field_size**2`` to be a power of two.

    The parameter ``p_rule`` represents the high-probability behavior used in the
    underlying assisted correlation model; the implementation uses it to compute
    an idealized success rate that is then passed through a binary symmetric
    channel with flip probability ``channel_noise``.

    Args:
        field_size: Side length of the square field.
        comms_size: Communication size. Must be 1.
        enemy_probability: Unused in the current implementation.
        channel_noise: Channel flip probability ``c``.
        p_rule: Assisted-correlation parameter in ``[0, 1]``.

    Returns:
        Expected success probability, clamped to ``[0.0, 1.0]``.

    Raises:
        ValueError: If argument constraints are violated.
    """
    if field_size < 1:
        raise ValueError("field_size must be >= 1.")
    n2 = field_size * field_size
    if comms_size != 1:
        raise ValueError("expected_win_rate_assisted currently supports comms_size == 1 only.")

    if n2 & (n2 - 1) != 0:
        raise ValueError("field_size**2 must be a power of two.")

    c = float(channel_noise)
    if not (0.0 <= c <= 1.0):
        raise ValueError("channel_noise must lie in [0.0, 1.0].")

    ph = float(p_rule)
    if not (0.0 <= ph <= 1.0):
        raise ValueError("p_rule must lie in [0.0, 1.0].")

    bit_string_length = n2
    exponent = int(math.log2(bit_string_length))

    K = 4.0 * (2.0 * ph - 1.0)
    s_ideal = 0.5 * (1.0 + (K / 4.0) ** exponent)
    s_noisy = (1.0 - c) * s_ideal + c * (1.0 - s_ideal)
    return max(0.0, min(1.0, float(s_noisy)))


def limit_from_mutual_information(
    field_size: int,
    comms_size: int,
    channel_noise: Number = 0.0,
    accuracy_in_digits: int = 8,
) -> float:
    """Compute an Information-Causality success upper bound from mutual information.

    The bound is computed by:
    - Taking the capacity (in bits) of a binary symmetric channel with flip
      probability ``c``: ``capacity = 1 - H2(c)``, where ``H2`` is binary entropy.
    - Converting ``m = comms_size`` communicated bits into an effective number of
      noiseless bits: ``m_eff = m * capacity``.
    - Mapping the per-cell information rate ``r = m_eff / N`` (with
      ``N = field_size**2``) to a target conditional entropy ``H_target = 1 - r``.
    - Inverting the binary entropy on the branch ``p ∈ [0.5, 1.0]`` to obtain the
      implied maximum per-shot success probability.

    Args:
        field_size: Side length of the square field.
        comms_size: Number of communicated bits ``m``. May be 0.
        channel_noise: Channel flip probability ``c``.
        accuracy_in_digits: Accuracy passed to :func:`binary_entropy_reverse`.

    Returns:
        Upper bound on per-shot success probability in ``[0.5, 1.0]``.

    Raises:
        ValueError: If argument constraints are violated.
    """
    if field_size < 1:
        raise ValueError("field_size must be >= 1.")
    n2 = field_size * field_size

    m = comms_size
    if m < 0 or m > n2:
        raise ValueError("comms_size must satisfy 0 <= comms_size <= field_size**2.")

    c = float(channel_noise)
    if not (0.0 <= c <= 1.0):
        raise ValueError("channel_noise must lie in [0.0, 1.0].")

    if m == 0:
        return 0.5

    capacity = 1.0 - binary_entropy(c)
    m_eff = m * capacity

    if m_eff <= 0.0:
        return 0.5

    if m_eff >= n2:
        return 1.0

    r = m_eff / float(n2)
    H_target = 1.0 - r
    p = binary_entropy_reverse(H_target, accuracy_in_digits=accuracy_in_digits)
    return float(p)