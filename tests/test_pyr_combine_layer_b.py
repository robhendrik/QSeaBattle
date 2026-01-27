import tensorflow as tf
import sys

if "./src" not in sys.path:
    sys.path.append("./src")


def _import_layer():
    try:
        from Q_Sea_Battle.pyr_combine_layer_b import PyrCombineLayerB  # type: ignore
        return PyrCombineLayerB
    except Exception:
        from pyr_combine_layer_b import PyrCombineLayerB  # type: ignore
        return PyrCombineLayerB


def test_pyr_combine_layer_b_shapes_and_trainable_weights():
    PyrCombineLayerB = _import_layer()

    B = 4
    L = 8
    layer = PyrCombineLayerB(hidden_units=16)

    gun_batch = tf.zeros((B, L), tf.float32)
    sr_outcome_batch = tf.zeros((B, L // 2), tf.float32)
    comm_batch = tf.zeros((B, 1), tf.float32)

    next_gun, next_comm = layer(gun_batch, sr_outcome_batch, comm_batch, training=True)

    assert tuple(next_gun.shape) == (B, L // 2)
    assert tuple(next_comm.shape) == (B, 1)
    assert layer.trainable
    assert layer.trainable_variables
    assert any("kernel" in v.name for v in layer.trainable_variables)


def test_pyr_combine_layer_b_accepts_other_even_L():
    PyrCombineLayerB = _import_layer()

    B = 2
    L = 10
    layer = PyrCombineLayerB(hidden_units=8)

    gun_batch = tf.random.uniform((B, L))
    sr_outcome_batch = tf.random.uniform((B, L // 2))
    comm_batch = tf.random.uniform((B, 1))

    next_gun, next_comm = layer(gun_batch, sr_outcome_batch, comm_batch)
    assert tuple(next_gun.shape) == (B, L // 2)
    assert tuple(next_comm.shape) == (B, 1)
