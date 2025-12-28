import sys
sys.path.append("./src")

import numpy as np
import tensorflow as tf

from Q_Sea_Battle.lin_combine_layer_a import LinCombineLayerA
from Q_Sea_Battle.lin_combine_layer_b import LinCombineLayerB


def _parity_bits(x_np: np.ndarray) -> np.ndarray:
    """Return parity (sum mod 2) for each row of a binary matrix; shape (B,1) int32."""
    # ensure int32
    x = x_np.astype(np.int32)
    return (np.sum(x, axis=1) % 2).astype(np.int32)[:, None]


def test_shapes_and_ranges() -> None:
    tf.random.set_seed(0)
    B = 16
    n2 = 8
    m = 3

    outcomes = tf.random.uniform((B, n2), minval=0, maxval=2, dtype=tf.int32)
    outcomes = tf.cast(outcomes, tf.float32)

    layer_a = LinCombineLayerA(comms_size=m)
    comm_logits = layer_a(outcomes, training=False)
    assert tuple(comm_logits.shape) == (B, m)

    comm = tf.random.normal((B, m))
    layer_b = LinCombineLayerB(comms_size=m)
    shoot_logit = layer_b(outcomes, comm, training=False)
    assert tuple(shoot_logit.shape) == (B, 1)

    # unbatched support
    o1 = tf.cast(tf.random.uniform((n2,), minval=0, maxval=2, dtype=tf.int32), tf.float32)
    c1 = tf.random.normal((m,))
    out_a1 = layer_a(o1, training=False)
    out_b1 = layer_b(o1, c1, training=False)
    assert tuple(out_a1.shape) == (m,)
    assert tuple(out_b1.shape) == (1,)


def test_gradients_nonzero() -> None:
    tf.random.set_seed(1)
    B = 32
    n2 = 8
    m = 2

    outcomes = tf.cast(tf.random.uniform((B, n2), minval=0, maxval=2, dtype=tf.int32), tf.float32)
    y = tf.cast(tf.random.uniform((B, m), minval=0, maxval=2, dtype=tf.int32), tf.float32)

    layer = LinCombineLayerA(comms_size=m)

    with tf.GradientTape() as tape:
        logits = layer(outcomes, training=True)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
    grads = tape.gradient(loss, layer.trainable_variables)

    assert any(g is not None and tf.reduce_sum(tf.abs(g)).numpy() > 0 for g in grads), \
        "Expected at least one nonzero gradient for LinCombineLayerA."


def test_tiny_overfit_parity() -> None:
    tf.random.set_seed(2)
    np.random.seed(2)

    # small parity-style dataset
    B = 256
    n2 = 8
    m = 1

    outcomes_np = np.random.randint(0, 2, size=(B, n2)).astype(np.float32)
    outcomes = tf.constant(outcomes_np, dtype=tf.float32)

    # Targets: parity for A; parity(outcomes) XOR comm for B
    y_comm_int = _parity_bits(outcomes_np)              # (B,1) int32
    y_comm = tf.constant(y_comm_int.astype(np.float32)) # (B,1) float32 labels

    comm_int = np.random.randint(0, 2, size=(B, m)).astype(np.int32)
    comm = tf.constant(comm_int.astype(np.float32))     # model input (float)

    y_shoot_int = np.logical_xor(y_comm_int.astype(bool), comm_int.astype(bool)).astype(np.int32)
    y_shoot = tf.constant(y_shoot_int.astype(np.float32))  # (B,1)

    # A: outcomes -> comm_logit
    layer_a = LinCombineLayerA(comms_size=m, hidden_units=(64, 64))
    model_a = tf.keras.Sequential([tf.keras.Input(shape=(n2,)), layer_a])
    model_a.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    model_a.fit(outcomes, y_comm, batch_size=32, epochs=350, verbose=0)
    pred_comm = (tf.sigmoid(model_a(outcomes)).numpy() >= 0.5).astype(np.float32)
    acc_comm = (pred_comm == y_comm.numpy()).mean()
    assert acc_comm > 0.98

    # B: (outcomes, comm) -> shoot_logit
    layer_b = LinCombineLayerB(comms_size=m, hidden_units=(64, 64))
    o_in = tf.keras.Input(shape=(n2,))
    c_in = tf.keras.Input(shape=(m,))
    out = layer_b(o_in, c_in)
    model_b = tf.keras.Model([o_in, c_in], out)
    model_b.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    model_b.fit([outcomes, comm], y_shoot, batch_size=32, epochs=350, verbose=0)
    pred_shoot = (tf.sigmoid(model_b([outcomes, comm])).numpy() >= 0.5).astype(np.float32)
    acc_shoot = (pred_shoot == y_shoot.numpy()).mean()
    assert acc_shoot > 0.98
