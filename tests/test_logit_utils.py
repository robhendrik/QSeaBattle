
"""Unit tests for the logit_utils module."""

from __future__ import annotations

import math

import numpy as np
import sys

sys.path.append("./src")
from Q_Sea_Battle.logit_utils import logit_to_prob, logit_to_logprob


def test_logit_to_prob_scalar_center():
    """logit 0 should map to probability 0.5."""
    p = logit_to_prob(0.0)
    assert isinstance(p, float)
    assert abs(p - 0.5) < 1e-12


def test_logit_to_prob_extremes():
    """Large positive/negative logits should saturate to 1.0/0.0."""
    p_pos = logit_to_prob(100.0)
    p_neg = logit_to_prob(-100.0)
    assert 1.0 - p_pos < 1e-12
    assert p_neg < 1e-12


def test_logit_to_prob_vector():
    """Vector inputs should be handled elementwise."""
    logits = np.array([-2.0, 0.0, 2.0])
    probs = logit_to_prob(logits)
    assert probs.shape == logits.shape
    # Compare against direct sigmoid
    expected = 1.0 / (1.0 + np.exp(-logits))
    assert np.allclose(probs, expected, atol=1e-12)


def test_logit_to_logprob_matches_definition():
    """logit_to_logprob should match log(sigmoid) and log(1 - sigmoid)."""
    logits = np.linspace(-5.0, 5.0, num=11)
    probs = logit_to_prob(logits)

    # Action = 1: log π(1|z) = log(sigmoid(z))
    logp_1 = logit_to_logprob(logits, np.ones_like(logits))
    expected_1 = np.log(probs)
    assert np.allclose(logp_1, expected_1, atol=1e-10)

    # Action = 0: log π(0|z) = log(1 - sigmoid(z))
    logp_0 = logit_to_logprob(logits, np.zeros_like(logits))
    expected_0 = np.log(1.0 - probs)
    assert np.allclose(logp_0, expected_0, atol=1e-10)


def test_logit_to_logprob_scalar():
    """Scalar inputs should return scalar outputs."""
    z = 1.5
    p = logit_to_prob(z)
    logp = logit_to_logprob(z, 1.0)
    assert isinstance(logp, float)
    assert abs(logp - math.log(p)) < 1e-12


def test_logit_to_logprob_invalid_actions_raises():
    """Actions outside {0,1} must raise a ValueError."""
    try:
        _ = logit_to_logprob(0.0, 0.5)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for invalid actions")
