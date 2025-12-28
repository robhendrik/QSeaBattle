"""Tests for the dru_utils module."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import sys

sys.path.append("./src")
from Q_Sea_Battle.dru_utils import dru_train, dru_execute
from Q_Sea_Battle.logit_utils import logit_to_prob


def test_dru_train_sigma_zero_numpy() -> None:
    """With sigma = 0, dru_train should reduce to logistic(logits)."""
    logits = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    out = dru_train(logits, sigma=0.0)
    expected = logit_to_prob(logits)
    assert out.shape == logits.shape
    assert np.allclose(out, expected, atol=1e-7)


def test_dru_train_range_and_shape_tf() -> None:
    """TensorFlow path should return values in (0, 1) with same shape."""
    logits = tf.constant([[0.0, 1.0], [-1.0, 3.0]], dtype=tf.float32)
    out = dru_train(logits, sigma=0.5)
    assert isinstance(out, tf.Tensor)
    assert out.shape == logits.shape
    out_np = out.numpy()
    assert np.all(out_np > 0.0)
    assert np.all(out_np < 1.0)


def test_dru_execute_numpy_thresholding() -> None:
    """dru_execute should hard-threshold NumPy inputs."""
    logits = np.array([-1.0, 0.0, 0.5, 2.0], dtype=np.float32)
    bits = dru_execute(logits, threshold=0.0)
    expected = np.array([0, 0, 1, 1], dtype=int)
    assert bits.dtype == int
    assert np.array_equal(bits, expected)


def test_dru_execute_tf_thresholding() -> None:
    """dru_execute should hard-threshold TensorFlow tensors."""
    logits = tf.constant([-1.0, 0.0, 0.5, 2.0], dtype=tf.float32)
    bits = dru_execute(logits, threshold=0.0)
    assert isinstance(bits, tf.Tensor)
    bits_np = bits.numpy()
    expected = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    assert np.array_equal(bits_np, expected)
