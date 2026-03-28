import tensorflow as tf
import sys

if "./src" not in sys.path:
    sys.path.append("./src")


def _import_layer():
    try:
        from Q_Sea_Battle.pyr_measurement_layer_a import PyrMeasurementLayerA  # type: ignore
        return PyrMeasurementLayerA
    except Exception:
        from pyr_measurement_layer_a import PyrMeasurementLayerA  # type: ignore
        return PyrMeasurementLayerA


def test_pyr_measurement_layer_a_shapes_trainable_and_range():
    PyrMeasurementLayerA = _import_layer()

    B = 4
    L = 8
    layer = PyrMeasurementLayerA(hidden_units=16)

    field_batch = tf.random.uniform((B, L), dtype=tf.float32)
    meas_a = layer(field_batch, training=True)

    assert tuple(meas_a.shape) == (B, L // 2)
    assert layer.trainable is True
    assert layer.trainable_variables, "Expected trainable_variables to be non-empty after build/call."

    # SR expects values in [0, 1]
    assert float(tf.reduce_min(meas_a).numpy()) >= 0.0
    assert float(tf.reduce_max(meas_a).numpy()) <= 1.0
