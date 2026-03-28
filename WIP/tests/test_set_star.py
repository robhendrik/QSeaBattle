"""Tests for set_alpha, set_p_high, set_beta, and set_sr_mode on
PRAssistedReplay, PyrInternalModelA, and PyrInternalModelB.

Coverage:
    PRAssistedReplay  — set_p_high, set_beta, set_sr_mode
    PyrInternalModelA — set_alpha, set_p_high, set_beta, set_sr_mode
    PyrInternalModelB — set_alpha, set_p_high, set_beta, set_sr_mode

Note: set_alpha on PRAssistedReplay is already covered by
test_pr_assisted_replay_alpha.py and is not duplicated here.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from Q_Sea_Battle_New.pr_assisted_replay import PRAssistedReplay
from Q_Sea_Battle_New.pyr_internal_model_a import PyrInternalModelA
from Q_Sea_Battle_New.pyr_internal_model_b import PyrInternalModelB


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _Layout:
    def __init__(self, n2: int, comms_size: int = 1):
        self.n2 = n2
        self.comms_size = comms_size


def _make_layer_inputs(prev_meas, curr_meas, prev_out, *, first=0.0, replay_out=None):
    inp = {
        "current_measurement": tf.constant(curr_meas, tf.float32),
        "previous_measurement": tf.constant(prev_meas, tf.float32),
        "previous_outcome": tf.constant(prev_out, tf.float32),
        "first_measurement": tf.constant([[first]], tf.float32),
    }
    if replay_out is not None:
        inp["replay_outcome_logits"] = tf.constant(replay_out, tf.float32)
    return inp


def _sr_var(sr: PRAssistedReplay, attr: str) -> float:
    var = getattr(sr, attr, None)
    assert var is not None, f"PRAssistedReplay has no attribute {attr!r}"
    return float(var.numpy())


def _build_model_a(model: PyrInternalModelA, n2: int, batch: int = 1) -> None:
    """Materialise model_a weights and SR layers via compute_with_internal."""
    x = tf.zeros((batch, n2), dtype=tf.float32) - 0.5
    replay_out = []
    state = x
    for d in range(model.depth):
        meas = model.measure_layers[d](state, training=False)
        replay_out.append(tf.zeros_like(meas))
        state = model.combine_layers[d](state, tf.zeros_like(meas), training=False)
    model.compute_with_internal(x, replay_out_a_logits_list=replay_out, training=False)


def _build_model_b(model: PyrInternalModelB, n2: int, batch: int = 1) -> None:
    """Materialise model_b weights and SR layers via compute_with_internal."""
    depth = int(np.log2(n2))
    gun = tf.zeros((batch, n2), tf.float32)
    comm = tf.zeros((batch, 1), tf.float32)
    k = n2 // 2
    prev_meas, prev_out = [], []
    for _ in range(depth):
        prev_meas.append(tf.zeros((batch, k), tf.float32))
        prev_out.append(tf.zeros((batch, k), tf.float32))
        k //= 2
    model.compute_with_internal(gun, comm, prev_meas, prev_out, training=False)


# ===========================================================================
# PRAssistedReplay — set_p_high
# ===========================================================================

class TestPRSetPHigh:
    def test_updates_variable(self):
        layer = PRAssistedReplay(sr_mode="stochastic", p_high=0.5, seed=0)
        layer.set_p_high(0.9)
        assert abs(float(layer._p_high.numpy()) - 0.9) < 1e-6

    def test_rejects_negative(self):
        layer = PRAssistedReplay(sr_mode="stochastic", p_high=0.5, seed=0)
        with pytest.raises(ValueError):
            layer.set_p_high(-0.1)

    def test_rejects_above_one(self):
        layer = PRAssistedReplay(sr_mode="stochastic", p_high=0.5, seed=0)
        with pytest.raises(ValueError):
            layer.set_p_high(1.1)

    def test_zero_always_flips_second_measurement(self):
        """p_high=0 → second measurement always violates the PR rule."""
        # Both measurements low → pr_out_clean ≈ prev_out (no PR flip).
        # p_high=0 → follow mask always False → out = -pr_out_clean ≈ -prev_out.
        layer = PRAssistedReplay(sr_mode="stochastic", p_high=1.0, beta=10.0, alpha=5.0, seed=7)
        inp = _make_layer_inputs(
            prev_meas=[[-10.0, -10.0]],
            curr_meas=[[-10.0, -10.0]],
            prev_out=[[5.0, -5.0]],
            first=0.0,
        )
        layer.set_p_high(0.0)
        out = layer(inp, training=False).numpy()
        assert out[0, 0] < 0.0, "Expected negative logit (forced flip) with p_high=0"
        assert out[0, 1] > 0.0, "Expected positive logit (forced flip) with p_high=0"

    def test_one_never_flips_second_measurement(self):
        """p_high=1 → second measurement always follows the PR rule."""
        layer = PRAssistedReplay(sr_mode="stochastic", p_high=0.0, beta=10.0, alpha=5.0, seed=7)
        inp = _make_layer_inputs(
            prev_meas=[[-10.0, -10.0]],
            curr_meas=[[-10.0, -10.0]],
            prev_out=[[5.0, -5.0]],
            first=0.0,
        )
        layer.set_p_high(1.0)
        out = layer(inp, training=False).numpy()
        assert out[0, 0] > 0.0, "Expected positive logit (no flip) with p_high=1"
        assert out[0, 1] < 0.0, "Expected negative logit (no flip) with p_high=1"

    def test_changes_output_under_tf_function(self):
        """set_p_high must affect output inside a @tf.function (uses tf.Variable)."""
        layer = PRAssistedReplay(sr_mode="stochastic", p_high=1.0, beta=10.0, alpha=5.0, seed=3)
        inp = _make_layer_inputs(
            prev_meas=[[-10.0, -10.0]],
            curr_meas=[[-10.0, -10.0]],
            prev_out=[[5.0, -5.0]],
            first=0.0,
        )

        @tf.function
        def f(x):
            return layer(x, training=False)

        out_before = f(inp)
        layer.set_p_high(0.0)
        out_after = f(inp)

        assert not tf.reduce_all(tf.equal(out_before, out_after)), (
            "Output did not change after set_p_high under tf.function; "
            "p_high may be captured as a Python float instead of tf.Variable."
        )


# ===========================================================================
# PRAssistedReplay — set_beta
# ===========================================================================

class TestPRSetBeta:
    def test_updates_variable(self):
        layer = PRAssistedReplay(sr_mode="stochastic", beta=10.0, seed=0)
        layer.set_beta(25.0)
        assert abs(float(layer._beta.numpy()) - 25.0) < 1e-6

    def test_rejects_zero(self):
        layer = PRAssistedReplay(sr_mode="stochastic", beta=10.0, seed=0)
        with pytest.raises(ValueError):
            layer.set_beta(0.0)

    def test_rejects_negative(self):
        layer = PRAssistedReplay(sr_mode="stochastic", beta=10.0, seed=0)
        with pytest.raises(ValueError):
            layer.set_beta(-5.0)

    def test_stochastic_first_meas_magnitude_equals_beta(self):
        """In stochastic mode, first-meas logit magnitude equals beta (bits × beta)."""
        k = 4
        layer = PRAssistedReplay(sr_mode="stochastic", beta=5.0, p_high=1.0, seed=42)
        inp = _make_layer_inputs(
            prev_meas=[[0.0] * k],
            curr_meas=[[0.0] * k],
            prev_out=[[0.0] * k],
            first=1.0,
        )

        out_small = layer(inp, training=False).numpy()
        # Regardless of which bits are sampled, |logit| == beta
        assert np.allclose(np.abs(out_small), 5.0, atol=1e-5), (
            f"Expected all |logits|=5.0, got {np.abs(out_small)}"
        )

        layer.set_beta(20.0)
        out_large = layer(inp, training=False).numpy()
        assert np.allclose(np.abs(out_large), 20.0, atol=1e-5), (
            f"Expected all |logits|=20.0 after set_beta, got {np.abs(out_large)}"
        )

    def test_changes_output_under_tf_function(self):
        """set_beta must affect output inside a @tf.function (uses tf.Variable)."""
        k = 4
        layer = PRAssistedReplay(sr_mode="stochastic", beta=5.0, p_high=1.0, seed=99)
        inp = _make_layer_inputs(
            prev_meas=[[0.0] * k],
            curr_meas=[[0.0] * k],
            prev_out=[[0.0] * k],
            first=1.0,
        )

        @tf.function
        def f(x):
            return layer(x, training=False)

        f(inp)  # trace once
        # For first measurement, beta determines magnitude; record magnitude before change
        out_before = f(inp)
        mag_before = float(tf.reduce_mean(tf.abs(out_before)).numpy())

        layer.set_beta(50.0)
        out_after = f(inp)
        mag_after = float(tf.reduce_mean(tf.abs(out_after)).numpy())

        assert abs(mag_after - mag_before) > 1.0, (
            "Output magnitude did not change after set_beta under tf.function; "
            "beta may be captured as a Python float instead of tf.Variable."
        )


# ===========================================================================
# PRAssistedReplay — set_sr_mode
# ===========================================================================

class TestPRSetSrMode:
    def test_updates_code_replay(self):
        layer = PRAssistedReplay(sr_mode="stochastic", seed=0)
        layer.set_sr_mode("replay")
        assert int(layer._sr_mode_code.numpy()) == PRAssistedReplay._SR_MODE_REPLAY

    def test_updates_code_stochastic(self):
        layer = PRAssistedReplay(sr_mode="replay", seed=0)
        layer.set_sr_mode("stochastic")
        assert int(layer._sr_mode_code.numpy()) == PRAssistedReplay._SR_MODE_STOCHASTIC

    def test_rejects_invalid_mode(self):
        layer = PRAssistedReplay(sr_mode="replay", seed=0)
        with pytest.raises(ValueError):
            layer.set_sr_mode("training")

    def test_rejects_empty_string(self):
        layer = PRAssistedReplay(sr_mode="replay", seed=0)
        with pytest.raises(ValueError):
            layer.set_sr_mode("")

    def test_replay_first_meas_returns_replay_outcome(self):
        """After switching to replay, first meas == replay_outcome_logits exactly."""
        layer = PRAssistedReplay(sr_mode="stochastic", seed=1)
        replay_out = [[3.0, -3.0, 3.0, -3.0]]
        inp = _make_layer_inputs(
            prev_meas=[[0.0] * 4],
            curr_meas=[[0.0] * 4],
            prev_out=[[0.0] * 4],
            first=1.0,
            replay_out=replay_out,
        )
        layer.set_sr_mode("replay")
        out = layer(inp, training=False).numpy()
        np.testing.assert_array_equal(out, replay_out)

    def test_stochastic_first_meas_ignores_replay_outcome(self):
        """In stochastic mode, first_meas != replay_outcome_logits (value 99)."""
        layer = PRAssistedReplay(sr_mode="replay", beta=10.0, p_high=1.0, seed=2)
        # Use 99.0 as a distinctive replay value that can't be output in stochastic mode
        # (stochastic first meas returns ±beta = ±10.0, never 99.0).
        replay_out = [[99.0, 99.0, 99.0, 99.0]]
        inp = _make_layer_inputs(
            prev_meas=[[0.0] * 4],
            curr_meas=[[0.0] * 4],
            prev_out=[[0.0] * 4],
            first=1.0,
            replay_out=replay_out,
        )
        layer.set_sr_mode("stochastic")
        out = layer(inp, training=False).numpy()
        assert not np.allclose(out, replay_out), (
            "Stochastic mode must not copy replay_outcome_logits for first measurement"
        )

    def test_changes_mode_under_tf_function(self):
        """set_sr_mode must take effect inside a traced @tf.function."""
        layer = PRAssistedReplay(sr_mode="replay", beta=10.0, p_high=1.0, seed=5)
        replay_out = [[7.0, -7.0]]
        inp = _make_layer_inputs(
            prev_meas=[[0.0, 0.0]],
            curr_meas=[[0.0, 0.0]],
            prev_out=[[0.0, 0.0]],
            first=1.0,
            replay_out=replay_out,
        )

        @tf.function
        def f(x):
            return layer(x, training=False)

        out_replay = f(inp)
        np.testing.assert_array_equal(out_replay.numpy(), replay_out,
                                      err_msg="Replay mode: expected replay_outcome_logits")

        layer.set_sr_mode("stochastic")
        out_stochastic = f(inp)
        assert not np.allclose(out_stochastic.numpy(), replay_out), (
            "After set_sr_mode('stochastic') under tf.function, output should differ from replay_out. "
            "sr_mode_code may be captured as a Python int instead of tf.Variable."
        )


# ===========================================================================
# PyrInternalModelA — set_* propagation
# ===========================================================================

class TestModelASetters:
    N2 = 4  # depth = log2(4) = 2 SR layers

    def _make(self, **kwargs) -> PyrInternalModelA:
        m = PyrInternalModelA(_Layout(n2=self.N2), seed=7, **kwargs)
        _build_model_a(m, self.N2)
        return m

    def test_set_alpha_propagates_to_all_sr_layers(self):
        m = self._make(alpha=1.0)
        m.set_alpha(8.5)
        for sr in m.sr_layers:
            assert abs(_sr_var(sr, "_alpha") - 8.5) < 1e-6

    def test_set_p_high_propagates_to_all_sr_layers(self):
        m = self._make(p_high=0.3)
        m.set_p_high(0.7)
        for sr in m.sr_layers:
            assert abs(_sr_var(sr, "_p_high") - 0.7) < 1e-6

    def test_set_beta_propagates_to_all_sr_layers(self):
        m = self._make(beta=10.0)
        m.set_beta(15.0)
        for sr in m.sr_layers:
            assert abs(_sr_var(sr, "_beta") - 15.0) < 1e-6

    def test_set_sr_mode_replay_to_stochastic(self):
        m = self._make(sr_mode="replay")
        m.set_sr_mode("stochastic")
        for sr in m.sr_layers:
            assert int(sr._sr_mode_code.numpy()) == PRAssistedReplay._SR_MODE_STOCHASTIC

    def test_set_sr_mode_stochastic_to_replay(self):
        m = self._make(sr_mode="stochastic")
        m.set_sr_mode("replay")
        for sr in m.sr_layers:
            assert int(sr._sr_mode_code.numpy()) == PRAssistedReplay._SR_MODE_REPLAY

    def test_set_alpha_out_of_range_raises(self):
        m = self._make(alpha=5.0)
        with pytest.raises((ValueError, AttributeError)):
            m.set_alpha(0.0)

    def test_set_p_high_out_of_range_raises(self):
        m = self._make(p_high=0.5)
        with pytest.raises((ValueError, AttributeError)):
            m.set_p_high(1.5)

    def test_set_beta_non_positive_raises(self):
        m = self._make(beta=10.0)
        with pytest.raises((ValueError, AttributeError)):
            m.set_beta(-1.0)

    def test_set_sr_mode_invalid_raises(self):
        m = self._make(sr_mode="replay")
        with pytest.raises((ValueError, AttributeError)):
            m.set_sr_mode("invalid")

    def test_has_expected_number_of_sr_layers(self):
        m = self._make()
        import math
        assert len(m.sr_layers) == m.depth == math.floor(math.log2(self.N2))


# ===========================================================================
# PyrInternalModelB — set_* propagation
# ===========================================================================

class TestModelBSetters:
    N2 = 4  # depth = log2(4) = 2 SR layers

    def _make(self, **kwargs) -> PyrInternalModelB:
        m = PyrInternalModelB(_Layout(n2=self.N2), seed=7, **kwargs)
        _build_model_b(m, self.N2)
        return m

    def test_set_alpha_propagates_to_all_sr_layers(self):
        m = self._make(alpha=2.0)
        m.set_alpha(9.0)
        for sr in m.sr_layers:
            assert abs(_sr_var(sr, "_alpha") - 9.0) < 1e-6

    def test_set_p_high_propagates_to_all_sr_layers(self):
        m = self._make(p_high=0.4)
        m.set_p_high(0.8)
        for sr in m.sr_layers:
            assert abs(_sr_var(sr, "_p_high") - 0.8) < 1e-6

    def test_set_beta_propagates_to_all_sr_layers(self):
        m = self._make(beta=5.0)
        m.set_beta(12.0)
        for sr in m.sr_layers:
            assert abs(_sr_var(sr, "_beta") - 12.0) < 1e-6

    def test_set_sr_mode_replay_to_stochastic(self):
        m = self._make(sr_mode="replay")
        m.set_sr_mode("stochastic")
        for sr in m.sr_layers:
            assert int(sr._sr_mode_code.numpy()) == PRAssistedReplay._SR_MODE_STOCHASTIC

    def test_set_sr_mode_stochastic_to_replay(self):
        m = self._make(sr_mode="stochastic")
        m.set_sr_mode("replay")
        for sr in m.sr_layers:
            assert int(sr._sr_mode_code.numpy()) == PRAssistedReplay._SR_MODE_REPLAY

    def test_set_alpha_out_of_range_raises(self):
        m = self._make(alpha=5.0)
        with pytest.raises((ValueError, AttributeError)):
            m.set_alpha(0.0)

    def test_set_p_high_out_of_range_raises(self):
        m = self._make(p_high=0.5)
        with pytest.raises((ValueError, AttributeError)):
            m.set_p_high(-0.2)

    def test_set_beta_non_positive_raises(self):
        m = self._make(beta=10.0)
        with pytest.raises((ValueError, AttributeError)):
            m.set_beta(0.0)

    def test_set_sr_mode_invalid_raises(self):
        m = self._make(sr_mode="replay")
        with pytest.raises((ValueError, AttributeError)):
            m.set_sr_mode("train")

    def test_has_expected_number_of_sr_layers(self):
        m = self._make()
        import math
        assert len(m.sr_layers) == m.depth == math.floor(math.log2(self.N2))
