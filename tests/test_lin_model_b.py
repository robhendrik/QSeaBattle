import sys
sys.path.append("./src")

import numpy as np
import tensorflow as tf

from Q_Sea_Battle.lin_trainable_assisted_model_a import LinTrainableAssistedModelA
from Q_Sea_Battle.lin_trainable_assisted_model_b import LinTrainableAssistedModelB


def test_model_b_shape_and_compatibility_with_model_a_prev_lists() -> None:
    tf.random.set_seed(0)
    np.random.seed(0)

    field_size = 4
    n2 = field_size * field_size
    m = 1
    B = 32

    # Model A produces comm logits and internal tensors in expected mode.
    model_a = LinTrainableAssistedModelA(field_size=field_size, comms_size=m, sr_mode="expected", seed=123)
    fields = np.random.randint(0, 2, size=(B, n2)).astype("float32")
    comm_logits, meas_list, out_list = model_a.compute_with_internal(fields)

    assert isinstance(meas_list, list) and len(meas_list) == 1
    assert isinstance(out_list, list) and len(out_list) == 1
    assert tuple(meas_list[0].shape) == (B, n2)
    assert tuple(out_list[0].shape) == (B, n2)

    # Model B consumes A's lists.
    model_b = LinTrainableAssistedModelB(field_size=field_size, comms_size=m, sr_mode="expected", seed=999)
    gun = np.random.randint(0, 2, size=(B, n2)).astype("float32")

    shoot_logit = model_b([gun, comm_logits, meas_list, out_list])
    assert tuple(shoot_logit.shape) == (B, 1)

    # Ensure prev tensors actually influence the result: changing prev_out_list should change outputs.
    out_list_zero = [tf.zeros_like(out_list[0])]
    shoot_logit_zero_prev = model_b([gun, comm_logits, meas_list, out_list_zero])
    assert not np.allclose(shoot_logit.numpy(), shoot_logit_zero_prev.numpy(), atol=1e-6)


def test_model_b_gradient_flow_expected_mode() -> None:
    tf.random.set_seed(1)
    np.random.seed(1)

    field_size = 4
    n2 = field_size * field_size
    m = 1
    B = 16

    model_a = LinTrainableAssistedModelA(field_size=field_size, comms_size=m, sr_mode="expected", seed=321)
    model_b = LinTrainableAssistedModelB(field_size=field_size, comms_size=m, sr_mode="expected", seed=654)

    fields = np.random.randint(0, 2, size=(B, n2)).astype("float32")
    gun = np.random.randint(0, 2, size=(B, n2)).astype("float32")

    comm_logits, meas_list, out_list = model_a.compute_with_internal(fields)
    # Use A's logits directly as the comm input (shape (B,m)).
    comm = tf.stop_gradient(comm_logits)

    y = tf.random.uniform((B, 1), minval=0, maxval=2, dtype=tf.int32)
    y = tf.cast(y, tf.float32)

    with tf.GradientTape() as tape:
        shoot_logit = model_b([gun, comm, meas_list, out_list], training=True)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=shoot_logit))

    grads = tape.gradient(loss, model_b.trainable_variables)
    assert grads is not None
    nonzero = False
    for g in grads:
        if g is None:
            continue
        if tf.reduce_sum(tf.abs(g)).numpy() > 0:
            nonzero = True
            break
    assert nonzero, "Expected at least one nonzero gradient flowing into ModelB parameters."
