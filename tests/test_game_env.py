"""Tests for the GameEnv environment class."""

from __future__ import annotations

import numpy as np
import pytest
import sys  
sys.path.append("./src")
from Q_Sea_Battle.game_env import GameEnv
from Q_Sea_Battle.game_layout import GameLayout


def test_reset_and_provide_shapes_and_types() -> None:
    """reset() followed by provide() should produce correct shapes and types."""
    layout = GameLayout(field_size=4, enemy_probability=0.5)
    env = GameEnv(layout)

    env.reset()
    field, gun = env.provide()

    n2 = layout.field_size ** 2

    assert isinstance(field, np.ndarray)
    assert isinstance(gun, np.ndarray)
    assert field.shape == (n2,)
    assert gun.shape == (n2,)

    # Values must be 0 or 1.
    assert set(np.unique(field)).issubset({0, 1})
    assert set(np.unique(gun)).issubset({0, 1})

    # Gun must be one-hot.
    assert int(gun.sum()) == 1


def test_provide_without_reset_raises() -> None:
    """Calling provide() before reset should raise a RuntimeError."""
    env = GameEnv(GameLayout())
    with pytest.raises(RuntimeError):
        _ = env.provide()


def test_evaluate_reward_hit_and_miss() -> None:
    """Evaluate should return 1.0 for correct decision, 0.0 otherwise."""
    layout = GameLayout(field_size=2, enemy_probability=0.5)
    env = GameEnv(layout)

    # Manually set a simple field and gun.
    env.field = np.array([[0, 0], [0, 1]], dtype=int)
    env.gun = np.array([[0, 0], [0, 1]], dtype=int)

    # Gun points at a cell with value 1.
    reward_hit = env.evaluate(1)
    reward_miss = env.evaluate(0)

    assert reward_hit == pytest.approx(1.0)
    assert reward_miss == pytest.approx(0.0)

    # Now make the target cell 0 and test again.
    env.field = np.array([[0, 0], [0, 0]], dtype=int)

    reward_hit_zero = env.evaluate(0)
    reward_miss_zero = env.evaluate(1)

    assert reward_hit_zero == pytest.approx(1.0)
    assert reward_miss_zero == pytest.approx(0.0)


def test_evaluate_without_reset_raises() -> None:
    """Calling evaluate() before reset should raise a RuntimeError."""
    env = GameEnv(GameLayout())
    with pytest.raises(RuntimeError):
        _ = env.evaluate(1)


def test_apply_channel_noise_zero_prob() -> None:
    """With channel_noise = 0.0, comm should be unchanged."""
    layout = GameLayout(channel_noise=0.0)
    env = GameEnv(layout)

    comm = np.array([0, 1, 1, 0], dtype=int)
    noisy = env.apply_channel_noise(comm)

    assert np.array_equal(noisy, comm)
    # Ensure original array was not modified in-place.
    assert noisy is not comm


def test_apply_channel_noise_one_prob() -> None:
    """With channel_noise = 1.0, all bits should be flipped."""
    layout = GameLayout(channel_noise=1.0)
    env = GameEnv(layout)

    comm = np.array([0, 1, 1, 0], dtype=int)
    noisy = env.apply_channel_noise(comm)

    expected = 1 - comm
    assert np.array_equal(noisy, expected)


def test_apply_channel_noise_intermediate_prob_statistics() -> None:
    """With intermediate noise, roughly the right fraction of bits should flip."""
    layout = GameLayout(channel_noise=0.5)
    env = GameEnv(layout)

    # Use many bits and trials to get a reasonable empirical estimate.
    comm = np.zeros(1000, dtype=int)
    flips = []

    for _ in range(20):
        noisy = env.apply_channel_noise(comm)
        flips.append(noisy.mean())  # fraction of 1s == fraction of flips

    # Average fraction of flips should be in a reasonable range around 0.5.
    avg_flip_rate = float(np.mean(flips))
    assert 0.3 <= avg_flip_rate <= 0.7
