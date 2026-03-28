import tensorflow as tf
import pytest

def _hard_logits(k, beta=10.0, seed=0):
    g = tf.random.Generator.from_seed(seed)
    u = g.uniform(shape=(1, k), minval=0.0, maxval=1.0, dtype=tf.float32)
    return tf.where(u < 0.5, -beta, beta)

def test_replay_first_measurement_identity(PRAssistedReplay):
    # Contract: sr_mode="replay", first_measurement=True => outcome_logits == replay_outcome_logits exactly.
    k = 8
    layer = PRAssistedReplay(sr_mode="replay", alpha=5.0, p_high=0.123, beta=10.0, seed=123)

    replay_out = _hard_logits(k, beta=10.0, seed=1)
    inputs = {
        "current_measurement": _hard_logits(k, beta=10.0, seed=2),
        "previous_measurement": _hard_logits(k, beta=10.0, seed=3),
        "previous_outcome": _hard_logits(k, beta=10.0, seed=4),
        "first_measurement": tf.ones((1, 1), dtype=tf.float32),
        "replay_outcome_logits": replay_out,
    }
    out = layer(inputs, training=True)
    tf.debugging.assert_equal(out, replay_out)

def test_replay_first_measurement_requires_replay_outcome(PRAssistedReplay):
    k = 4
    layer = PRAssistedReplay(sr_mode="replay", alpha=5.0, p_high=0.9, beta=10.0, seed=7)
    inputs = {
        "current_measurement": _hard_logits(k, seed=1),
        "previous_measurement": _hard_logits(k, seed=2),
        "previous_outcome": _hard_logits(k, seed=3),
        "first_measurement": tf.ones((1, 1), dtype=tf.float32),
        # missing replay_outcome_logits
    }
    with pytest.raises((ValueError, KeyError)):
        _ = layer(inputs, training=True)

def test_replay_second_measurement_ignores_replay_outcome(PRAssistedReplay):
    # Contract: for second measurement, replay_outcome_logits is ignored.
    k = 8
    layer = PRAssistedReplay(sr_mode="replay", alpha=5.0, p_high=1.0, beta=10.0, seed=1)

    prev_meas = tf.fill((1, k), -10.0)  # low
    curr_meas = tf.fill((1, k), -10.0)  # low => no flip, outcome ~= prev_outcome
    prev_out = _hard_logits(k, beta=10.0, seed=5)

    inputs_a = {
        "current_measurement": curr_meas,
        "previous_measurement": prev_meas,
        "previous_outcome": prev_out,
        "first_measurement": tf.zeros((1, 1), dtype=tf.float32),
        "replay_outcome_logits": tf.fill((1, k), 999.0),  # should be ignored
    }
    inputs_b = dict(inputs_a)
    inputs_b["replay_outcome_logits"] = tf.fill((1, k), -999.0)  # different ignored value

    out_a = layer(inputs_a, training=True)
    out_b = layer(inputs_b, training=True)
    tf.debugging.assert_near(out_a, out_b, atol=0.0)  # exact match expected if truly ignored
