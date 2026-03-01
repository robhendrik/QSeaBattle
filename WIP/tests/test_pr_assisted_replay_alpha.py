# test_pr_assisted_replay_alpha.py
import tensorflow as tf
import pytest

from Q_Sea_Battle_New.pr_assisted_replay import PRAssistedReplay


def _inputs(prev_meas, curr_meas, prev_out, *, first=0.0):
    # PR layer expects dict inputs
    return {
        "current_measurement": tf.constant(curr_meas, tf.float32),
        "previous_measurement": tf.constant(prev_meas, tf.float32),
        "previous_outcome": tf.constant(prev_out, tf.float32),
        "first_measurement": tf.constant([[first]], tf.float32),
    }


def test_pr_set_alpha_changes_output_eager():
    # Choose values where alpha matters (not saturated, not zero)
    layer = PRAssistedReplay(sr_mode="replay", alpha=1.0, beta=10.0, seed=123)

    x = _inputs(
        prev_meas=[[0.2, 0.2]],
        curr_meas=[[0.2, 0.2]],
        prev_out=[[1.0, -1.0]],
        first=0.0,
    )

    y1 = layer(x, training=False)

    # Must exist and update runtime parameter
    assert hasattr(layer, "set_alpha"), "Expected PRAssistedReplay.set_alpha(alpha)"
    layer.set_alpha(10.0)

    y2 = layer(x, training=False)

    # Output must change when alpha changes
    assert not tf.reduce_all(tf.equal(y1, y2)), "Output did not change after alpha update"


def test_pr_set_alpha_changes_output_under_tf_function():
    layer = PRAssistedReplay(sr_mode="replay", alpha=1.0, beta=10.0, seed=123)

    x = _inputs(
        prev_meas=[[0.2, 0.2]],
        curr_meas=[[0.2, 0.2]],
        prev_out=[[1.0, -1.0]],
        first=0.0,
    )

    @tf.function
    def f(inp):
        return layer(inp, training=False)

    y1 = f(x)  # traces once

    # update alpha after tracing
    layer.set_alpha(10.0)
    y2 = f(x)  # should read new alpha if alpha is tf.Variable

    # if alpha is still a Python float, y2 == y1 and this test FAILS
    assert not tf.reduce_all(tf.equal(y1, y2)), (
        "Under tf.function, output did not change after alpha update. "
        "This usually means alpha is captured as Python float, not tf.Variable."
    )


def test_pr_set_alpha_rejects_non_positive():
    layer = PRAssistedReplay(sr_mode="replay", alpha=1.0, beta=10.0, seed=123)

    with pytest.raises(ValueError):
        layer.set_alpha(0.0)

    with pytest.raises(ValueError):
        layer.set_alpha(-1.0)
