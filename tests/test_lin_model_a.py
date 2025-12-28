"""Tests for LinTrainableAssistedModelA (Step 4 acceptance).

Acceptance criteria:
- call returns (B,m)
- compute_with_internal returns (comm_logits, [meas], [out]) with list length 1
- reproducibility in expected mode
- gradient flows from comm loss back into A (expected mode)
"""

from __future__ import annotations

import sys

import numpy as np
import tensorflow as tf

sys.path.append("./src")

from Q_Sea_Battle.lin_trainable_assisted_model_a import LinTrainableAssistedModelA


def test_shapes_and_internal_lists() -> None:
    tf.random.set_seed(123)
    np.random.seed(123)

    field_size = 4
    m = 1
    n2 = field_size * field_size

    model = LinTrainableAssistedModelA(field_size=field_size, comms_size=m, sr_mode="expected", seed=123)

    B = 7
    x = tf.constant(np.random.randint(0, 2, size=(B, n2)).astype("float32"))

    y = model(x)
    assert tuple(y.shape) == (B, m)

    logits, meas_list, out_list = model.compute_with_internal(x)
    assert tuple(logits.shape) == (B, m)
    assert isinstance(meas_list, list) and isinstance(out_list, list)
    assert len(meas_list) == 1
    assert len(out_list) == 1
    assert tuple(meas_list[0].shape) == (B, n2)
    assert tuple(out_list[0].shape) == (B, n2)


def test_reproducibility_expected_mode() -> None:
    tf.random.set_seed(1)
    np.random.seed(1)

    field_size = 4
    m = 1
    n2 = field_size * field_size

    model = LinTrainableAssistedModelA(field_size=field_size, comms_size=m, sr_mode="expected", seed=999)

    B = 5
    x = tf.constant(np.random.randint(0, 2, size=(B, n2)).astype("float32"))

    y1 = model(x).numpy()
    y2 = model(x).numpy()
    assert np.array_equal(y1, y2)


def test_gradient_flow_expected_mode() -> None:
    tf.random.set_seed(7)
    np.random.seed(7)

    field_size = 4
    m = 1
    n2 = field_size * field_size

    model = LinTrainableAssistedModelA(field_size=field_size, comms_size=m, sr_mode="expected", seed=0)

    B = 16
    x = tf.constant(np.random.randint(0, 2, size=(B, n2)).astype("float32"))
    y_true = tf.constant(np.random.randint(0, 2, size=(B, m)).astype("float32"))

    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=logits))

    grads = tape.gradient(loss, model.trainable_variables)
    nonzero = False
    for g in grads:
        if g is None:
            continue
        if float(tf.reduce_sum(tf.abs(g)).numpy()) > 0.0:
            nonzero = True
            break
    assert nonzero
