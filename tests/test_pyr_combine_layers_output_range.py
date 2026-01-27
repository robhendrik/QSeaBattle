import tensorflow as tf
import sys
if "./src" not in sys.path:
    sys.path.append("./src")

def test_pyr_combine_layers_outputs_in_unit_interval():
    from Q_Sea_Battle.pyr_combine_layer_a import PyrCombineLayerA
    from Q_Sea_Battle.pyr_combine_layer_b import PyrCombineLayerB

    B = 4
    L = 8  # must be even

    # ---- Combine A ----
    layer_a = PyrCombineLayerA(hidden_units=16)
    field = tf.random.uniform((B, L), minval=0.0, maxval=1.0, dtype=tf.float32)
    sr = tf.random.uniform((B, L // 2), minval=0.0, maxval=1.0, dtype=tf.float32)

    y = layer_a(field, sr, training=True)
    assert tuple(y.shape) == (B, L // 2)

    y_min = float(tf.reduce_min(y))
    y_max = float(tf.reduce_max(y))
    assert y_min >= -1e-6
    assert y_max <= 1.0 + 1e-6

    assert len(layer_a.trainable_variables) > 0

    # ---- Combine B ----
    layer_b = PyrCombineLayerB(hidden_units=16)
    gun = tf.random.uniform((B, L), minval=0.0, maxval=1.0, dtype=tf.float32)
    sr2 = tf.random.uniform((B, L // 2), minval=0.0, maxval=1.0, dtype=tf.float32)
    comm = tf.random.uniform((B, 1), minval=0.0, maxval=1.0, dtype=tf.float32)

    next_gun, next_comm = layer_b(gun, sr2, comm, training=True)
    assert tuple(next_gun.shape) == (B, L // 2)
    assert tuple(next_comm.shape) == (B, 1)

    ng_min = float(tf.reduce_min(next_gun))
    ng_max = float(tf.reduce_max(next_gun))
    nc_min = float(tf.reduce_min(next_comm))
    nc_max = float(tf.reduce_max(next_comm))

    assert ng_min >= -1e-6
    assert ng_max <= 1.0 + 1e-6
    assert nc_min >= -1e-6
    assert nc_max <= 1.0 + 1e-6

    assert len(layer_b.trainable_variables) > 0
