"""Tests for pyramid Step-1 measurement and combine layers."""

import numpy as np
import tensorflow as tf

from Q_Sea_Battle.pyr_measurement_layer_a import PyrMeasurementLayerA
from Q_Sea_Battle.pyr_combine_layer_a import PyrCombineLayerA
from Q_Sea_Battle.pyr_measurement_layer_b import PyrMeasurementLayerB
from Q_Sea_Battle.pyr_combine_layer_b import PyrCombineLayerB


def _np_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) % 2


def test_pyr_measurement_layer_a_pairwise_xor() -> None:
    layer = PyrMeasurementLayerA()
    field = np.array([[0, 0, 0, 1, 1, 1, 1, 0]], dtype=np.float32)  # L=8
    out = layer(field).numpy()
    expected = np.array([[0 ^ 0, 0 ^ 1, 1 ^ 1, 1 ^ 0]], dtype=np.float32)
    assert out.shape == (1, 4)
    np.testing.assert_array_equal(out, expected)


def test_pyr_combine_layer_a_even_xor_sr() -> None:
    layer = PyrCombineLayerA()
    field = np.array([[1, 0, 0, 1, 1, 1, 0, 0]], dtype=np.float32)  # L=8
    sr = np.array([[0, 1, 0, 1]], dtype=np.float32)               # L/2=4
    out = layer(field, sr).numpy()
    even = field[:, ::2]
    expected = _np_xor(even, sr)
    assert out.shape == (1, 4)
    np.testing.assert_array_equal(out, expected)


def test_pyr_measurement_layer_b_not_even_and_odd() -> None:
    layer = PyrMeasurementLayerB()
    gun = np.array([[0, 1, 1, 0, 0, 0, 1, 1]], dtype=np.float32)  # L=8
    out = layer(gun).numpy()
    even = gun[:, ::2]
    odd = gun[:, 1::2]
    expected = (1 - even) * odd
    assert out.shape == (1, 4)
    np.testing.assert_array_equal(out, expected)


def test_pyr_combine_layer_b_wiring_only() -> None:
    """
    Wiring-only test for PyrCombineLayerB (trainable / untrained):
    - accepts batched inputs (gun, sr, comm)
    - returns two outputs with correct batch dimension and shapes
    - outputs are finite and float32
    - outputs depend on inputs (not constant w.r.t. gun/sr/comm)
    - gradients flow to trainable variables
    """
    import numpy as np
    import tensorflow as tf

    layer = PyrCombineLayerB()

    B = 3
    L = 8
    H = L // 2

    # gun: one-hot (but we do NOT test the downsampling rule here)
    gun = np.zeros((B, L), dtype=np.float32)
    gun[np.arange(B), np.array([1, 4, 7])] = 1.0

    # sr: arbitrary 0/1 mask
    sr = np.array([[1, 0, 1, 1],
                   [0, 1, 0, 1],
                   [1, 1, 0, 0]], dtype=np.float32)

    # comm: arbitrary 0/1
    comm = np.array([[1.0], [0.0], [1.0]], dtype=np.float32)

    gun_t = tf.convert_to_tensor(gun)
    sr_t = tf.convert_to_tensor(sr)
    comm_t = tf.convert_to_tensor(comm)

    # Forward pass
    out = layer(gun_t, sr_t, comm_t)
    assert isinstance(out, (tuple, list)) and len(out) == 2, "Layer must return (next_gun, next_comm)"
    next_gun_t, next_comm_t = out

    # --- shape checks ---
    assert tuple(next_gun_t.shape) == (B, H), f"next_gun shape {next_gun_t.shape} != {(B, H)}"
    assert tuple(next_comm_t.shape) == (B, 1), f"next_comm shape {next_comm_t.shape} != {(B, 1)}"

    # --- dtype + finiteness checks ---
    assert next_gun_t.dtype == tf.float32
    assert next_comm_t.dtype == tf.float32
    next_gun = next_gun_t.numpy()
    next_comm = next_comm_t.numpy()
    assert np.isfinite(next_gun).all(), "next_gun contains non-finite values"
    assert np.isfinite(next_comm).all(), "next_comm contains non-finite values"

    # --- determinism for identical inputs (no SR sampling inside this layer) ---
    next_gun2, next_comm2 = layer(gun_t, sr_t, comm_t)
    np.testing.assert_allclose(next_gun2.numpy(), next_gun, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(next_comm2.numpy(), next_comm, atol=0.0, rtol=0.0)

    # --- outputs should depend on inputs (wiring sanity) ---
    # Change gun -> next_gun should typically change
    gun_alt = np.zeros((B, L), dtype=np.float32)
    gun_alt[np.arange(B), np.array([0, 2, 6])] = 1.0
    next_gun_alt, next_comm_alt = layer(
        tf.convert_to_tensor(gun_alt), sr_t, comm_t
    )
    assert not np.allclose(next_gun_alt.numpy(), next_gun), "next_gun did not change when gun changed (suspicious wiring)"

    # Change sr -> next_comm should typically change
    sr_alt = 1.0 - sr
    next_gun_alt2, next_comm_alt2 = layer(
        gun_t, tf.convert_to_tensor(sr_alt.astype(np.float32)), comm_t
    )
    assert not np.allclose(next_comm_alt2.numpy(), next_comm), "next_comm did not change when sr changed (suspicious wiring)"

    # Change comm -> next_comm should typically change
    comm_alt = 1.0 - comm
    next_gun_alt3, next_comm_alt3 = layer(
        gun_t, sr_t, tf.convert_to_tensor(comm_alt)
    )
    assert not np.allclose(next_comm_alt3.numpy(), next_comm), "next_comm did not change when comm changed (suspicious wiring)"

    # --- gradients should flow to trainable vars ---
    with tf.GradientTape() as tape:
        y_g, y_c = layer(gun_t, sr_t, comm_t)
        loss = tf.reduce_sum(y_g) + tf.reduce_sum(y_c)

    grads = tape.gradient(loss, layer.trainable_variables)
    # If layer has trainable variables, at least one gradient should be non-None
    if layer.trainable_variables:
        assert any(g is not None for g in grads), "No gradients flowed to trainable variables (wiring issue)"



def test_layers_are_tf_function_compatible() -> None:
    """Smoke-test tracing: ensures no Python-side shape assumptions break tracing."""
    a_meas = PyrMeasurementLayerA()
    a_comb = PyrCombineLayerA()
    b_meas = PyrMeasurementLayerB()
    b_comb = PyrCombineLayerB()

    @tf.function
    def run(field, gun, sr, comm):
        m_a = a_meas(field)
        f_next = a_comb(field, sr)
        m_b = b_meas(gun)
        g_next, c_next = b_comb(gun, sr, comm)
        return m_a, f_next, m_b, g_next, c_next

    field = tf.constant([[0.0, 1.0, 1.0, 0.0]], dtype=tf.float32)  # L=4
    gun = tf.constant([[1.0, 0.0, 0.0, 1.0]], dtype=tf.float32)    # L=4
    sr = tf.constant([[1.0, 0.0]], dtype=tf.float32)              # L/2=2
    comm = tf.constant([[0.0]], dtype=tf.float32)
    outs = run(field, gun, sr, comm)
    assert len(outs) == 5
