import tensorflow as tf

def test_replay_second_measurement_gradients_flow(PRAssistedReplay):
    # Core requirement: in replay mode, second measurement must be differentiable wrt measurement logits.
    k = 8
    layer = PRAssistedReplay(sr_mode="replay", alpha=2.0, p_high=0.9, beta=10.0, seed=0)

    prev_meas = tf.Variable(tf.linspace(-1.0, 1.0, k)[None, :], dtype=tf.float32)
    curr_meas = tf.Variable(tf.linspace(1.0, -1.0, k)[None, :], dtype=tf.float32)
    prev_out = tf.fill((1, k), 10.0)

    with tf.GradientTape() as tape:
        inputs = {
            "current_measurement": curr_meas,
            "previous_measurement": prev_meas,
            "previous_outcome": prev_out,
            "first_measurement": tf.zeros((1, 1), dtype=tf.float32),
        }
        out = layer(inputs, training=True)
        # Any outcome-dependent loss; mean is fine
        loss = tf.reduce_mean(out)

    g_prev, g_curr = tape.gradient(loss, [prev_meas, curr_meas])

    assert g_prev is not None
    assert g_curr is not None
    assert tf.linalg.norm(g_prev).numpy() > 1e-6
    assert tf.linalg.norm(g_curr).numpy() > 1e-6
