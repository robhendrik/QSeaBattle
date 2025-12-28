"""Tests for NeuralNetPlayerB (v0.3)."""

import sys

import numpy as np
import tensorflow as tf

sys.path.append("./src")

import Q_Sea_Battle as qsb  # type: ignore[import]


def build_dummy_model_b(field_size: int, comms_size: int) -> tf.keras.Model:
    """Build a tiny dummy model for Player B testing.

    The model expects compact inputs of shape (1 + m,) where the first
    dimension is the normalised gun index and the remaining m entries
    are communication bits.
    """
    in_dim = 1 + comms_size
    inputs = tf.keras.Input(shape=(in_dim,))
    x = tf.keras.layers.Dense(4, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(1, activation=None)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def test_neural_net_player_b_decide_and_logprob() -> None:
    """NeuralNetPlayerB should return a valid shoot bit and log-prob."""
    layout = qsb.GameLayout(field_size=2, comms_size=1)
    model_b = build_dummy_model_b(layout.field_size, layout.comms_size)
    player_b = qsb.NeuralNetPlayerB(layout, model_b)

    gun = np.array([0, 1, 0, 0], dtype=int)
    comm = np.array([1], dtype=int)

    # Deterministic mode.
    player_b.explore = False
    shoot = player_b.decide(gun, comm)
    assert shoot in (0, 1)

    logprob_det = player_b.get_log_prob()
    assert isinstance(logprob_det, float)

    # Stochastic mode.
    player_b.explore = True
    shoot2 = player_b.decide(gun, comm)
    assert shoot2 in (0, 1)
    logprob_stoch = player_b.get_log_prob()
    assert isinstance(logprob_stoch, float)


def test_neural_net_player_b_reset() -> None:
    """Reset should clear stored log-probability."""
    layout = qsb.GameLayout(field_size=2, comms_size=1)
    model_b = build_dummy_model_b(layout.field_size, layout.comms_size)
    player_b = qsb.NeuralNetPlayerB(layout, model_b)

    gun = np.array([1, 0, 0, 0], dtype=int)
    comm = np.array([0], dtype=int)
    player_b.decide(gun, comm)
    _ = player_b.get_log_prob()

    player_b.reset()
    try:
        _ = player_b.get_log_prob()
        assert False, "Expected RuntimeError after reset with no decision"
    except RuntimeError:
        pass
