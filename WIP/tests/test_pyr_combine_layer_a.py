import tensorflow as tf
import sys

# Ensure src/ is importable when running pytest from repo root
if "./src" not in sys.path:
    sys.path.append("./src")


def _import_layer():
    try:
        from Q_Sea_Battle.pyr_combine_layer_a import PyrCombineLayerA  # type: ignore
        return PyrCombineLayerA
    except Exception:
        from pyr_combine_layer_a import PyrCombineLayerA  # type: ignore
        return PyrCombineLayerA


def test_pyr_combine_layer_a_shapes_and_trainable_weights():
    PyrCombineLayerA = _import_layer()

    B = 4
    L = 8  # must be even
    layer = PyrCombineLayerA(hidden_units=16)

    field_batch = tf.zeros((B, L), dtype=tf.float32)
    sr_outcome_batch = tf.zeros((B, L // 2), dtype=tf.float32)

    y = layer(field_batch, sr_outcome_batch, training=True)

    assert tuple(y.shape) == (B, L // 2)
    assert layer.trainable is True
    assert layer.trainable_variables, "Expected trainable_variables to be non-empty after build/call."
    assert any("kernel" in v.name for v in layer.trainable_variables), "Expected a Dense kernel variable."


def test_pyr_combine_layer_a_accepts_other_even_L():
    PyrCombineLayerA = _import_layer()

    B = 2
    L = 10  # must be even
    layer = PyrCombineLayerA(hidden_units=8)

    field_batch = tf.random.uniform((B, L), dtype=tf.float32)
    sr_outcome_batch = tf.random.uniform((B, L // 2), dtype=tf.float32)

    y = layer(field_batch, sr_outcome_batch, training=False)
    assert tuple(y.shape) == (B, L // 2)
