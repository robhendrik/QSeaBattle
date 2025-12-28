"""Tests for the SharedRandomness helper class."""

import sys

sys.path.append("./src")

import numpy as np
import pytest

from Q_Sea_Battle.shared_randomness import SharedRandomness


def test_first_measurement_shape_and_values() -> None:
    """First measurement should give a 0/1 vector of the right shape."""
    sr = SharedRandomness(length=4, p_high=0.8)
    meas = np.array([0, 1, 0, 1], dtype=int)
    out = sr.measurement_a(meas)

    assert out.shape == meas.shape
    assert out.dtype == int
    assert set(np.unique(out)).issubset({0, 1})


def test_second_measurement_shape_and_values() -> None:
    """Second measurement should also give a 0/1 vector of the right shape."""
    sr = SharedRandomness(length=4, p_high=0.8)

    meas_a = np.array([1, 0, 1, 0], dtype=int)
    _ = sr.measurement_a(meas_a)

    meas_b = np.array([0, 1, 0, 1], dtype=int)
    out_b = sr.measurement_b(meas_b)

    assert out_b.shape == meas_b.shape
    assert out_b.dtype == int
    assert set(np.unique(out_b)).issubset({0, 1})


def test_second_measurement_deterministic_p_high_1() -> None:
    """For p_high = 1.0, high and low cases become deterministic.

    Design rule:
      * High-correlation cases (keep previous outcome with prob p_high):
            (prev, curr) in {(0, 0), (0, 1), (1, 0)}
      * Low-correlation case (keep with prob 1 - p_high, flip with p_high):
            (prev, curr) = (1, 1)

    With p_high = 1.0:
      * (1, 1) -> always flip previous outcome
      * (0, 0), (0, 1), (1, 0) -> always keep previous outcome
    """
    length = 4
    sr = SharedRandomness(length=length, p_high=1.0)

    # Low-correlation case: (0, 0) everywhere -> always keep.
    sr.reset()
    meas_prev = np.zeros(length, dtype=int)
    prev_out = sr.measurement_a(meas_prev)
    meas_curr = np.zeros(length, dtype=int)
    out_b = sr.measurement_b(meas_curr)
    assert np.array_equal(out_b, prev_out)

    # High-correlation case: (1, 1) everywhere -> always flip.
    sr.reset()
    meas_prev = np.ones(length, dtype=int)
    prev_out = sr.measurement_a(meas_prev)
    meas_curr = np.ones(length, dtype=int)
    out_b = sr.measurement_b(meas_curr)
    assert np.array_equal(out_b, 1-prev_out)

    # High-correlation case: (0, 1) everywhere -> always keep.
    sr.reset()
    meas_prev = np.zeros(length, dtype=int)
    prev_out = sr.measurement_a(meas_prev)
    meas_curr = np.ones(length, dtype=int)
    out_b = sr.measurement_b(meas_curr)
    assert np.array_equal(out_b, prev_out)

    # High-correlation case: (1, 0) everywhere -> always keep.
    sr.reset()
    meas_prev = np.ones(length, dtype=int)
    prev_out = sr.measurement_a(meas_prev)
    meas_curr = np.zeros(length, dtype=int)
    out_b = sr.measurement_b(meas_curr)
    assert np.array_equal(out_b, prev_out)


def test_second_measurement_probabilistic_p_high_half() -> None:
    """With p_high = 0.5, all cases behave like fair coins.

    For any (prev, curr) pair, the probability to keep vs flip should
    be ~0.5. This checks that the implementation does not introduce
    any asymmetric bias beyond what p_high specifies.
    """
    length = 1
    sr = SharedRandomness(length=length, p_high=0.5)

    rng = np.random.default_rng(123)
    patterns = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]

    for prev_bit, curr_bit in patterns:
        keeps = 0
        trials = 2000

        for _ in range(trials):
            sr.reset()
            prev_out = sr.measurement_a(np.array([prev_bit], dtype=int))
            out_b = sr.measurement_b(np.array([curr_bit], dtype=int))
            if out_b[0] == prev_out[0]:
                keeps += 1

        keep_rate = keeps / trials
        # Should be roughly 0.5, allow some tolerance.
        assert 0.4 < keep_rate < 0.6, (prev_bit, curr_bit, keep_rate)


def test_double_measurement_same_party_raises() -> None:
    """Calling measurement_a twice without reset should raise."""
    sr = SharedRandomness(length=4, p_high=0.8)
    meas = np.array([0, 0, 1, 1], dtype=int)
    _ = sr.measurement_a(meas)
    with pytest.raises(ValueError):
        _ = sr.measurement_a(meas)
