import tensorflow as tf

def _hard_logits(k, val):
    return tf.fill((1, k), tf.constant(val, tf.float32))

def test_replay_second_measurement_pr_rule_approx(PRAssistedReplay):
    # With hard measurement logits and sufficiently large alpha, the soft gate should approximate:
    # - flip (negate) if both are high
    # - same otherwise
    k = 16
    beta = 10.0
    alpha = 5.0
    layer = PRAssistedReplay(sr_mode="replay", alpha=alpha, p_high=1.0, beta=beta, seed=0)

    prev_out = tf.concat([_hard_logits(k//2, beta), _hard_logits(k//2, -beta)], axis=1)

    def run(prev_meas_val, curr_meas_val):
        inputs = {
            "current_measurement": _hard_logits(k, curr_meas_val),
            "previous_measurement": _hard_logits(k, prev_meas_val),
            "previous_outcome": prev_out,
            "first_measurement": tf.zeros((1, 1), dtype=tf.float32),
        }
        return layer(inputs, training=True)

    out_hh = run(+10.0, +10.0)
    out_hl = run(+10.0, -10.0)
    out_lh = run(-10.0, +10.0)
    out_ll = run(-10.0, -10.0)

    # Expect close to -prev_out for high/high
    tf.debugging.assert_near(out_hh, -prev_out, atol=1e-3)

    # Expect close to +prev_out otherwise
    tf.debugging.assert_near(out_hl, prev_out, atol=1e-3)
    tf.debugging.assert_near(out_lh, prev_out, atol=1e-3)
    tf.debugging.assert_near(out_ll, prev_out, atol=1e-3)
