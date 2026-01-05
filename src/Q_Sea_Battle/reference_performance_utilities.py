
"""Utility functions for analytic performance benchmarks in QSeaBattle.

Author: Rob Hendriks
Version: 0.1
"""

from __future__ import annotations

import math
from typing import Union


Number = Union[float, int]


def binary_entropy(p: Number) -> float:
    """Compute Shannon binary entropy H(p) in bits.

    For 0 < p < 1, returns:

        H(p) = -p * log2(p) - (1 - p) * log2(1 - p)

    For p <= 0 or p >= 1, returns 0.0 (the limiting value).
    """
    p = float(p)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def binary_entropy_reverse(H: Number, accuracy_in_digits: int = 8) -> float:
    """Invert binary entropy on the branch p in [0.5, 1.0]."""
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
    """Analytic expected win rate for SimplePlayers."""
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
    """Analytic expected win rate for MajorityPlayers.

    Model:
    - Field has N = field_size^2 i.i.d. Bernoulli(p) cells, p = enemy_probability.
    - Alice partitions the flattened field into m = comms_size equal contiguous blocks,
      each of length L = N/m, and sends the per-block majority bit (ties -> 1).
    - The communicated bits pass through a binary symmetric channel with flip prob c.
    - Bob, given a uniformly random gun index, selects the corresponding block bit
      and outputs it as his guess for that cell.

    Returns the average success probability over the random field and random gun index.
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

    # Degenerate fields: majority bit is deterministic; only channel flips can fail.
    if p == 0.0 or p == 1.0:
        return 1.0 - c

    L = N // m

    # Precompute logs for stability
    log_p = math.log(p)
    log_q = math.log(1.0 - p)

    def binom_prob(L_: int, k_: int) -> float:
        """Binomial pmf in log-space to avoid overflow."""
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
        # k = number of ones in the block
        majority_bit = 1 if (k * 2 >= L) else 0  # ties -> 1, per your MajorityPlayerA spec
        p_cell_1 = k / float(L)       # probability the queried cell in this block is 1
        p_cell_0 = 1.0 - p_cell_1

        # Channel flip: transmitted majority_bit is flipped with prob c before Bob uses it.
        if majority_bit == 1:
            # Bob outputs 1 w.p. (1-c), 0 w.p. c
            p_success_given_k = (1.0 - c) * p_cell_1 + c * p_cell_0
        else:
            # Bob outputs 0 w.p. (1-c), 1 w.p. c
            p_success_given_k = (1.0 - c) * p_cell_0 + c * p_cell_1

        expected_success += binom_prob(L, k) * p_success_given_k

    return float(expected_success)



def expected_win_rate_assisted(
    field_size: int,
    comms_size: int,
    enemy_probability: Number = 0.5,
    channel_noise: Number = 0.0,
    p_high: Number = 0.9,
) -> float:
    """Analytic expected win rate for classical AssistedPlayers (one-bit comm)."""
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

    ph = float(p_high)
    if not (0.0 <= ph <= 1.0):
        raise ValueError("p_high must lie in [0.0, 1.0].")

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
    """Maximum allowed success probability under Information Causality."""
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
