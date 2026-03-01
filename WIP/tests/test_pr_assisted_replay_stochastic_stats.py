import tensorflow as tf
import pytest

@pytest.mark.slow
def test_stochastic_first_measurement_uniform_50_50(PRAssistedReplay):
    # Contract: first measurement in stochastic mode is uniform 50/50; p_high unused.
    k = 64
    N = 20000  # number of samples
    layer = PRAssistedReplay(sr_mode="stochastic", alpha=5.0, p_high=0.9, beta=10.0, seed=123)

    # Vectorize by using batch dimension N
    inputs = {
        "current_measurement": tf.zeros((N, k), tf.float32),
        "previous_measurement": tf.zeros((N, k), tf.float32),
        "previous_outcome": tf.zeros((N, k), tf.float32),
        "first_measurement": tf.ones((N, 1), tf.float32),
    }
    out = layer(inputs, training=False)
    bits = tf.cast(out >= 0.0, tf.float32)
    mean = tf.reduce_mean(bits).numpy()
    assert abs(mean - 0.5) < 0.02  # tolerance

@pytest.mark.slow
def test_stochastic_second_measurement_follow_rate_matches_p_high(PRAssistedReplay):
    # Contract: second measurement in stochastic mode follows PR rule with prob p_high.
    k = 32
    N = 20000
    p_high = 0.8
    layer = PRAssistedReplay(sr_mode="stochastic", alpha=5.0, p_high=p_high, beta=10.0, seed=7)

    # Choose hard logits such that PR rule is "flip" (high/high)
    prev_meas = tf.fill((N, k), 10.0)
    curr_meas = tf.fill((N, k), 10.0)
    prev_out = tf.where(tf.random.uniform((N, k), 0, 1) < 0.5, -10.0, 10.0)

    # Expected PR-rule outcome in logits is -prev_out for high/high
    expected = -prev_out

    inputs = {
        "current_measurement": curr_meas,
        "previous_measurement": prev_meas,
        "previous_outcome": prev_out,
        "first_measurement": tf.zeros((N, 1), tf.float32),
    }
    out = layer(inputs, training=False)

    # Follow if sign(out) == sign(expected) elementwise.
    follow = tf.cast(tf.math.sign(out) == tf.math.sign(expected), tf.float32)
    follow_rate = tf.reduce_mean(follow).numpy()
    assert abs(follow_rate - p_high) < 0.03
