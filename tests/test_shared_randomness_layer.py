"""Tests for SharedRandomnessLayer.

These tests focus on the core contract:

* Expected-mode matches an analytic reference for binary inputs.
* Sample-mode is reproducible given a fixed seed.
* Shapes are preserved for batched and unbatched inputs.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

sys.path.append("./src")

tf = pytest.importorskip("tensorflow")

from Q_Sea_Battle.shared_randomness_layer import SharedRandomnessLayer


def _expected_second_outcome(
    prev_m: np.ndarray,
    curr_m: np.ndarray,
    prev_o: np.ndarray,
    p_high: float,
) -> np.ndarray:
    """Analytic expected outcome for the second measurement on binary inputs."""

    prev_m = prev_m.astype(np.float32)
    curr_m = curr_m.astype(np.float32)
    prev_o = prev_o.astype(np.float32)

    both_one = prev_m * curr_m
    p_same = p_high + both_one * (1.0 - 2.0 * p_high)
    # E = p_same * prev_o + (1-p_same)*(1-prev_o)
    return (1.0 - p_same) + prev_o * (2.0 * p_same - 1.0)


def test_expected_mode_matches_reference_binary() -> None:
    tf.keras.backend.clear_session()

    length = 8
    p_high = 0.8
    layer = SharedRandomnessLayer(length=length, p_high=p_high, mode="expected")

    # Binary inputs, batch shape (B, length)
    prev_m = np.array(
        [[0, 0, 0, 1, 1, 0, 1, 1], [1, 0, 1, 0, 1, 0, 0, 1]], dtype=np.float32
    )
    curr_m = np.array(
        [[0, 1, 0, 1, 0, 0, 1, 0], [1, 1, 0, 0, 1, 0, 1, 0]], dtype=np.float32
    )
    prev_o = np.array(
        [[0, 1, 1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 1, 0, 1, 0]], dtype=np.float32
    )

    # Second measurement
    first = np.zeros((prev_m.shape[0], 1), dtype=np.float32)
    out = layer(
        {
            "current_measurement": tf.constant(curr_m),
            "previous_measurement": tf.constant(prev_m),
            "previous_outcome": tf.constant(prev_o),
            "first_measurement": tf.constant(first),
        }
    ).numpy()

    ref = _expected_second_outcome(prev_m, curr_m, prev_o, p_high=p_high)
    np.testing.assert_allclose(out, ref, rtol=0.0, atol=1e-6)

    # First measurement should be 0.5 in expected mode (regardless of other inputs).
    first = np.ones((prev_m.shape[0], 1), dtype=np.float32)
    out_first = layer(
        {
            "current_measurement": tf.constant(curr_m),
            "previous_measurement": tf.constant(prev_m),
            "previous_outcome": tf.constant(prev_o),
            "first_measurement": tf.constant(first),
        }
    ).numpy()
    np.testing.assert_allclose(out_first, 0.5 * np.ones_like(prev_m), atol=0.0)


def test_sample_mode_reproducible_with_seed() -> None:
    tf.keras.backend.clear_session()

    length = 16
    p_high = 0.9
    seed = 123
    layer = SharedRandomnessLayer(
        length=length, p_high=p_high, mode="sample", resource_index=7, seed=seed
    )

    B = 4
    curr_m = tf.zeros((B, length), dtype=tf.float32)
    prev_m = tf.zeros((B, length), dtype=tf.float32)
    prev_o = tf.zeros((B, length), dtype=tf.float32)

    # First measurement: should sample deterministically.
    first = tf.ones((B, 1), dtype=tf.float32)
    out1 = layer(
        {
            "current_measurement": curr_m,
            "previous_measurement": prev_m,
            "previous_outcome": prev_o,
            "first_measurement": first,
        }
    ).numpy()
    out2 = layer(
        {
            "current_measurement": curr_m,
            "previous_measurement": prev_m,
            "previous_outcome": prev_o,
            "first_measurement": first,
        }
    ).numpy()

    np.testing.assert_array_equal(out1, out2)
    assert set(np.unique(out1)).issubset({0.0, 1.0})


def test_shapes_batched_and_unbatched() -> None:
    tf.keras.backend.clear_session()

    length = 8
    layer = SharedRandomnessLayer(length=length, p_high=0.75, mode="expected")

    # Batched
    B = 3
    z = tf.zeros((B, length), dtype=tf.float32)
    out_b = layer(
        {
            "current_measurement": z,
            "previous_measurement": z,
            "previous_outcome": z,
            "first_measurement": tf.ones((B, 1), dtype=tf.float32),
        }
    )
    assert tuple(out_b.shape) == (B, length)

    # Unbatched (length,)
    z1 = tf.zeros((length,), dtype=tf.float32)
    out_u = layer(
        {
            "current_measurement": z1,
            "previous_measurement": z1,
            "previous_outcome": z1,
            "first_measurement": tf.constant(1.0, dtype=tf.float32),
        }
    )
    assert tuple(out_u.shape) == (length,)
