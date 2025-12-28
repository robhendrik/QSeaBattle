
"""Tests for reference_performance_utilities module."""

import sys
import math

import numpy as np

sys.path.append("./src")

import Q_Sea_Battle as qsb  # type: ignore[import]


def test_binary_entropy_basic_values() -> None:
    be = qsb.binary_entropy
    assert be(0.0) == 0.0
    assert be(1.0) == 0.0
    assert abs(be(0.5) - 1.0) < 1e-12


def test_binary_entropy_reverse_roundtrip() -> None:
    p_true = 0.8
    H = qsb.binary_entropy(p_true)
    p_est = qsb.binary_entropy_reverse(H, accuracy_in_digits=8)
    assert 0.5 <= p_est <= 1.0
    assert abs(p_est - p_true) < 1e-3


def brute_force_simple(field_size: int, comms_size: int, enemy_probability: float, channel_noise: float) -> float:
    n2 = field_size * field_size
    m = comms_size
    p = enemy_probability
    c = channel_noise

    total_prob = 0.0
    total_reward = 0.0

    for field_int in range(1 << n2):
        field = np.array([(field_int >> i) & 1 for i in range(n2)], dtype=int)
        k = int(field.sum())
        prob_field = (p ** k) * ((1.0 - p) ** (n2 - k))

        for gun_idx in range(n2):
            gun = np.zeros(n2, dtype=int)
            gun[gun_idx] = 1
            prob_gun = 1.0 / n2

            comm = field[:m].copy()

            for noise_int in range(1 << m):
                noise = np.array([(noise_int >> i) & 1 for i in range(m)], dtype=int)
                flips = int(noise.sum())
                prob_noise = (c ** flips) * ((1.0 - c) ** (m - flips))
                noisy_comm = comm ^ noise

                if gun_idx < m:
                    shoot = int(noisy_comm[gun_idx])
                    cell_value = int(field[gun_idx])
                    reward = 1.0 if shoot == cell_value else 0.0

                    prob = prob_field * prob_gun * prob_noise
                    total_prob += prob
                    total_reward += prob * reward
                else:
                    for shoot_val in (0, 1):
                        prob_shoot = p if shoot_val == 1 else (1.0 - p)
                        cell_value = int(field[gun_idx])
                        reward = 1.0 if shoot_val == cell_value else 0.0

                        prob = prob_field * prob_gun * prob_noise * prob_shoot
                        total_prob += prob
                        total_reward += prob * reward

    assert abs(total_prob - 1.0) < 1e-9
    return total_reward


def test_expected_win_rate_simple_matches_enumeration() -> None:
    field_size = 2
    comms_size = 2
    enemy_probability = 0.3
    channel_noise = 0.1

    analytic = qsb.expected_win_rate_simple(
        field_size=field_size,
        comms_size=comms_size,
        enemy_probability=enemy_probability,
        channel_noise=channel_noise,
    )
    brute = brute_force_simple(field_size, comms_size, enemy_probability, channel_noise)
    assert abs(analytic - brute) < 1e-9


def brute_force_majority(field_size: int, comms_size: int, enemy_probability: float, channel_noise: float) -> float:
    n2 = field_size * field_size
    m = comms_size
    L = n2 // m
    p = enemy_probability
    c = channel_noise

    total_prob = 0.0
    total_reward = 0.0

    for field_int in range(1 << n2):
        field = np.array([(field_int >> i) & 1 for i in range(n2)], dtype=int)
        k_field = int(field.sum())
        prob_field = (p ** k_field) * ((1.0 - p) ** (n2 - k_field))

        comm = np.zeros(m, dtype=int)
        for seg in range(m):
            start = seg * L
            end = start + L
            segment = field[start:end]
            ones = int(segment.sum())
            comm[seg] = 1 if (2 * ones >= L) else 0

        for gun_idx in range(n2):
            gun = np.zeros(n2, dtype=int)
            gun[gun_idx] = 1
            prob_gun = 1.0 / n2

            seg = gun_idx // L

            for flip_bit in (0, 1):
                prob_noise = (1.0 - c) if flip_bit == 0 else c
                noisy_bit = comm[seg] ^ flip_bit
                shoot = int(noisy_bit)
                cell_value = int(field[gun_idx])
                reward = 1.0 if shoot == cell_value else 0.0

                prob = prob_field * prob_gun * prob_noise
                total_prob += prob
                total_reward += prob * reward

    assert abs(total_prob - 1.0) < 1e-9
    return total_reward


def test_expected_win_rate_majority_matches_enumeration() -> None:
    field_size = 2
    comms_size = 1
    enemy_probability = 0.4
    channel_noise = 0.0

    analytic = qsb.expected_win_rate_majority(
        field_size=field_size,
        comms_size=comms_size,
        enemy_probability=enemy_probability,
        channel_noise=channel_noise,
    )
    brute = brute_force_majority(field_size, comms_size, enemy_probability, channel_noise)

    assert abs(analytic - brute) < 1e-9


def test_expected_win_rate_assisted_noise_symmetry() -> None:
    field_size = 4
    comms_size = 1
    p_high = 0.9

    s0 = qsb.expected_win_rate_assisted(
        field_size=field_size,
        comms_size=comms_size,
        channel_noise=0.0,
        p_high=p_high,
    )
    s_half = qsb.expected_win_rate_assisted(
        field_size=field_size,
        comms_size=comms_size,
        channel_noise=0.5,
        p_high=p_high,
    )
    s1 = qsb.expected_win_rate_assisted(
        field_size=field_size,
        comms_size=comms_size,
        channel_noise=1.0,
        p_high=p_high,
    )

    assert abs(s_half - 0.5) < 1e-12
    assert abs(s1 - (1.0 - s0)) < 1e-12


def test_limit_from_mutual_information_edge_cases() -> None:
    p0 = qsb.limit_from_mutual_information(
        field_size=2,
        comms_size=0,
        channel_noise=0.0,
    )
    assert abs(p0 - 0.5) < 1e-12

    p_full = qsb.limit_from_mutual_information(
        field_size=2,
        comms_size=4,
        channel_noise=0.0,
    )
    assert abs(p_full - 1.0) < 1e-12

    p_noise = qsb.limit_from_mutual_information(
        field_size=2,
        comms_size=4,
        channel_noise=0.5,
    )
    assert abs(p_noise - 0.5) < 1e-12


def test_limit_from_mutual_information_matches_entropy_reverse() -> None:
    field_size = 2
    n2 = field_size * field_size
    m = 2
    c = 0.0

    p_bound = qsb.limit_from_mutual_information(
        field_size=field_size,
        comms_size=m,
        channel_noise=c,
    )

    r = m / n2
    H_target = 1.0 - r
    p_expected = qsb.binary_entropy_reverse(H_target, accuracy_in_digits=8)

    assert abs(p_bound - p_expected) < 1e-3
