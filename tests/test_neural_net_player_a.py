"""Tests for NeuralNetPlayerA (v0.3)."""

import sys

import numpy as np
import tensorflow as tf

sys.path.append("./src")

import Q_Sea_Battle as qsb  # type: ignore[import]


def build_dummy_model_a(field_size: int, comms_size: int) -> tf.keras.Model:
    """Build a tiny dummy model for testing purposes.

    The model expects a *scaled* field of shape (n2,) as input and produces
    logits of shape (m,).
    """
    n2 = field_size ** 2
    inputs = tf.keras.Input(shape=(n2,))
    x = tf.keras.layers.Dense(4, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(comms_size, activation=None)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def test_neural_net_player_a_decide_and_logprob() -> None:
    """NeuralNetPlayerA should produce valid comm bits and a log-prob."""
    layout = qsb.GameLayout(field_size=2, comms_size=2)
    model_a = build_dummy_model_a(layout.field_size, layout.comms_size)
    player_a = qsb.NeuralNetPlayerA(layout, model_a)

    field = np.array([0, 1, 1, 0], dtype=int)

    # Deterministic mode.
    player_a.explore = False
    comm = player_a.decide(field)
    assert comm.shape == (layout.comms_size,)
    assert np.all((comm == 0) | (comm == 1))

    logprob_det = player_a.get_log_prob()
    assert isinstance(logprob_det, float)

    # Stochastic mode.
    player_a.explore = True
    comm2 = player_a.decide(field)
    assert comm2.shape == (layout.comms_size,)
    logprob_stoch = player_a.get_log_prob()
    assert isinstance(logprob_stoch, float)


def test_neural_net_player_a_reset() -> None:
    """Reset should clear stored log-probability."""
    layout = qsb.GameLayout(field_size=2, comms_size=1)
    model_a = build_dummy_model_a(layout.field_size, layout.comms_size)
    player_a = qsb.NeuralNetPlayerA(layout, model_a)

    field = np.array([1, 0, 0, 1], dtype=int)
    player_a.decide(field)
    _ = player_a.get_log_prob()

    player_a.reset()
    try:
        _ = player_a.get_log_prob()
        assert False, "Expected RuntimeError after reset with no decision"
    except RuntimeError:
        pass
