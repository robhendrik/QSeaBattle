import sys

sys.path.append("./src")

import numpy as np
import tensorflow as tf

from Q_Sea_Battle import LinMeasurementLayerA, LinMeasurementLayerB


def _assert_in_unit_interval(x: tf.Tensor) -> None:
    xmin = float(tf.reduce_min(x).numpy())
    xmax = float(tf.reduce_max(x).numpy())
    assert xmin >= -1e-6, f"min was {xmin}"
    assert xmax <= 1.0 + 1e-6, f"max was {xmax}"


def test_lin_measurement_layers_shapes_and_range() -> None:
    tf.random.set_seed(0)
    np.random.seed(0)

    n2 = 16
    B = 4
    x_batch = tf.constant(np.random.randint(0, 2, size=(B, n2)).astype("float32"))
    x_vec = tf.constant(np.random.randint(0, 2, size=(n2,)).astype("float32"))

    layer_a = LinMeasurementLayerA(n2=n2, hidden_units=(32,))
    layer_b = LinMeasurementLayerB(n2=n2, hidden_units=(32,))

    yb_a = layer_a(x_batch, training=False)
    yv_a = layer_a(x_vec, training=False)
    yb_b = layer_b(x_batch, training=False)
    yv_b = layer_b(x_vec, training=False)

    assert tuple(yb_a.shape) == (B, n2)
    assert tuple(yv_a.shape) == (n2,)
    assert tuple(yb_b.shape) == (B, n2)
    assert tuple(yv_b.shape) == (n2,)

    _assert_in_unit_interval(yb_a)
    _assert_in_unit_interval(yv_a)
    _assert_in_unit_interval(yb_b)
    _assert_in_unit_interval(yv_b)


def test_lin_measurement_layers_have_nonzero_gradients() -> None:
    tf.random.set_seed(1)
    np.random.seed(1)

    n2 = 12
    B = 5
    x = tf.constant(np.random.randint(0, 2, size=(B, n2)).astype("float32"))
    y_true = tf.constant(np.random.randint(0, 2, size=(B, n2)).astype("float32"))

    layer = LinMeasurementLayerA(n2=n2, hidden_units=(16,))
    with tf.GradientTape() as tape:
        y_pred = layer(x, training=True)
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    grads = tape.gradient(loss, layer.trainable_variables)

    nonzero = False
    for g in grads:
        if g is None:
            continue
        if float(tf.reduce_sum(tf.abs(g)).numpy()) > 0.0:
            nonzero = True
            break
    assert nonzero, "Expected at least one non-zero gradient."


def _tiny_overfit(layer_ctor, seed: int) -> None:
    tf.random.set_seed(seed)
    np.random.seed(seed)

    n2 = 20
    B = 16

    # Tiny dataset: learn identity on binary inputs.
    x = tf.constant(np.random.randint(0, 2, size=(B, n2)).astype("float32"))
    y_true = tf.identity(x)

    layer = layer_ctor(n2=n2, hidden_units=(64,))
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)

    for _ in range(300):
        with tf.GradientTape() as tape:
            y_pred = layer(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
        grads = tape.gradient(loss, layer.trainable_variables)
        opt.apply_gradients(zip(grads, layer.trainable_variables))
        if float(loss.numpy()) < 0.02:
            break

    y_hat = tf.cast(y_pred > 0.5, tf.float32)
    acc = float(tf.reduce_mean(tf.cast(tf.equal(y_hat, y_true), tf.float32)).numpy())
    assert acc > 0.98, f"Overfit accuracy too low: {acc:.4f}"


def test_lin_measurement_layer_a_overfits_tiny_dataset() -> None:
    _tiny_overfit(LinMeasurementLayerA, seed=2)


def test_lin_measurement_layer_b_overfits_tiny_dataset() -> None:
    _tiny_overfit(LinMeasurementLayerB, seed=3)
