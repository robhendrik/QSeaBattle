import numpy as np
import pytest

pytest.importorskip("tensorflow")
import tensorflow as tf  # noqa: E402

from Q_Sea_Battle.pyr_trainable_assisted_model_a import PyrTrainableAssistedModelA
from Q_Sea_Battle.pyr_trainable_assisted_model_b import PyrTrainableAssistedModelB
from Q_Sea_Battle.pyr_trainable_assisted_imitation_utils import (
    transfer_pyr_model_a_layer_weights,
    transfer_pyr_model_b_layer_weights,
)


class Layout:
    def __init__(self, n2: int, comms_size: int = 1):
        self.n2 = n2
        self.comms_size = comms_size


def _shared_layer_model_a(layout: Layout, p_high=0.9, sr_mode="sample") -> PyrTrainableAssistedModelA:
    """Construct a model where all levels share the same layer instances (old behavior)."""
    depth = int(round(np.log2(layout.n2)))
    meas = tf.keras.layers.Lambda(lambda x: x)  # placeholder; we will replace after init
    comb = tf.keras.layers.Lambda(lambda x, y: y)  # placeholder signature mismatch; not used directly
    # Use the default model to get correct layer types, then overwrite lists to share instances.
    m = PyrTrainableAssistedModelA(layout, p_high=p_high, sr_mode=sr_mode)
    shared_meas = m.measure_layers[0]
    shared_comb = m.combine_layers[0]
    m.measure_layers = [shared_meas for _ in range(depth)]
    m.combine_layers = [shared_comb for _ in range(depth)]
    m.measure_layer = shared_meas
    m.combine_layer = shared_comb
    return m


def test_old_behavior_preserved_matches_shared_instance():
    """Regression: per-level lists should preserve outputs of the prior shared-layer approach."""

    layout = Layout(n2=16, comms_size=1)
    batch = tf.constant(np.random.randint(0, 2, size=(8, 16)).astype(np.float32))

    # New (per-level default primitives)
    model_new = PyrTrainableAssistedModelA(layout, p_high=0.9, sr_mode="expected")
    logits_new, meas_new, out_new = model_new.compute_with_internal(batch)

    # Old behavior (shared instance across levels)
    model_old = _shared_layer_model_a(layout, p_high=0.9, sr_mode="expected")
    logits_old, meas_old, out_old = model_old.compute_with_internal(batch)

    # Same shapes and numerically identical (deterministic primitives; SR controlled by same seed + resource_index).
    assert logits_new.shape == logits_old.shape == (8, 1)
    assert len(meas_new) == len(meas_old) == 4
    assert len(out_new) == len(out_old) == 4

    for a, b in zip(meas_new, meas_old):
        np.testing.assert_allclose(a.numpy(), b.numpy(), atol=0, rtol=0)
    for a, b in zip(out_new, out_old):
        np.testing.assert_allclose(a.numpy(), b.numpy(), atol=0, rtol=0)
    np.testing.assert_allclose(logits_new.numpy(), logits_old.numpy(), atol=0, rtol=0)


class MeasTrainable(tf.keras.layers.Layer):
    """Trainable measurement-like layer with fixed (L -> L/2) Dense map."""
    def __init__(self, L: int):
        super().__init__()
        self.L = int(L)
        self.dense = tf.keras.layers.Dense(self.L // 2, use_bias=True)

    def call(self, x):
        x = tf.cast(x, tf.float32)
        return self.dense(x)


class CombATrainable(tf.keras.layers.Layer):
    """Trainable combine-like layer for A: concat(state, out) -> Dense(L/2)."""
    def __init__(self, L: int):
        super().__init__()
        self.L = int(L)
        self.dense = tf.keras.layers.Dense(self.L // 2, use_bias=True)

    def call(self, state, out):
        state = tf.cast(state, tf.float32)
        out = tf.cast(out, tf.float32)
        z = tf.concat([state, out], axis=-1)
        return self.dense(z)


class CombBTrainable(tf.keras.layers.Layer):
    """Trainable combine-like layer for B: concat(state, out, comm) -> (Dense(L/2), Dense(1))."""
    def __init__(self, L: int):
        super().__init__()
        self.L = int(L)
        self.dense_g = tf.keras.layers.Dense(self.L // 2, use_bias=True)
        self.dense_c = tf.keras.layers.Dense(1, use_bias=True)

    def call(self, state, out, comm):
        state = tf.cast(state, tf.float32)
        out = tf.cast(out, tf.float32)
        comm = tf.cast(comm, tf.float32)
        z = tf.concat([state, out, comm], axis=-1)
        return self.dense_g(z), tf.sigmoid(self.dense_c(z))


def _init_layer_weights(layer: tf.keras.layers.Layer, input_shapes):
    """Build a layer by calling it with zeros of the given shapes."""
    if isinstance(input_shapes, tuple) and isinstance(input_shapes[0], (tuple, list)):
        # multiple inputs
        zeros = [tf.zeros((1, *sh), dtype=tf.float32) for sh in input_shapes]
        _ = layer(*zeros)
    elif isinstance(input_shapes, (tuple, list)) and isinstance(input_shapes[0], int):
        _ = layer(tf.zeros((1, *input_shapes), dtype=tf.float32))
    else:
        raise ValueError("Unsupported input_shapes format.")


def test_transfer_helpers_succeed_with_real_models_and_trainable_layers():
    """Regression: real pyramid models expose measure_layers/combine_layers and transfer copies weights."""
    layout = Layout(n2=16, comms_size=1)
    dims = [16, 8, 4, 2]

    # Source (trained) layers with known weights
    src_meas_a = [MeasTrainable(L) for L in dims]
    src_comb_a = [CombATrainable(L) for L in dims]
    src_meas_b = [MeasTrainable(L) for L in dims]
    src_comb_b = [CombBTrainable(L) for L in dims]

    # Build layers (so they have weights)
    for L, lyr in zip(dims, src_meas_a):
        _ = lyr(tf.zeros((1, L), dtype=tf.float32))
    for L, lyr in zip(dims, src_comb_a):
        _ = lyr(tf.zeros((1, L), dtype=tf.float32), tf.zeros((1, L//2), dtype=tf.float32))
    for L, lyr in zip(dims, src_meas_b):
        _ = lyr(tf.zeros((1, L), dtype=tf.float32))
    for L, lyr in zip(dims, src_comb_b):
        _ = lyr(tf.zeros((1, L), dtype=tf.float32), tf.zeros((1, L//2), dtype=tf.float32), tf.zeros((1, 1), dtype=tf.float32))

    # Set deterministic weights
    def set_known(layer):
        ws = layer.get_weights()
        new = []
        for w in ws:
            new.append(np.arange(w.size, dtype=np.float32).reshape(w.shape) / 100.0)
        layer.set_weights(new)

    for lst in [src_meas_a, src_comb_a, src_meas_b, src_comb_b]:
        for lyr in lst:
            set_known(lyr)

    # Destination models with same layer types but zeroed weights
    dst_meas_a = [MeasTrainable(L) for L in dims]
    dst_comb_a = [CombATrainable(L) for L in dims]
    dst_meas_b = [MeasTrainable(L) for L in dims]
    dst_comb_b = [CombBTrainable(L) for L in dims]

    for L, lyr in zip(dims, dst_meas_a):
        _ = lyr(tf.zeros((1, L), dtype=tf.float32))
    for L, lyr in zip(dims, dst_comb_a):
        _ = lyr(tf.zeros((1, L), dtype=tf.float32), tf.zeros((1, L//2), dtype=tf.float32))
    for L, lyr in zip(dims, dst_meas_b):
        _ = lyr(tf.zeros((1, L), dtype=tf.float32))
    for L, lyr in zip(dims, dst_comb_b):
        _ = lyr(tf.zeros((1, L), dtype=tf.float32), tf.zeros((1, L//2), dtype=tf.float32), tf.zeros((1, 1), dtype=tf.float32))

    # zero weights
    for lst in [dst_meas_a, dst_comb_a, dst_meas_b, dst_comb_b]:
        for lyr in lst:
            lyr.set_weights([np.zeros_like(w) for w in lyr.get_weights()])

    model_a = PyrTrainableAssistedModelA(layout, p_high=0.9, sr_mode="sample", measure_layers=dst_meas_a, combine_layers=dst_comb_a)
    model_b = PyrTrainableAssistedModelB(layout, p_high=0.9, sr_mode="sample", measure_layers=dst_meas_b, combine_layers=dst_comb_b)

    # Transfer into the models
    transfer_pyr_model_a_layer_weights(model_a, src_meas_a, src_comb_a)
    transfer_pyr_model_b_layer_weights(model_b, src_meas_b, src_comb_b)

    # Verify exact equality of weights at each level
    for i in range(len(dims)):
        for w_src, w_dst in zip(src_meas_a[i].get_weights(), model_a.measure_layers[i].get_weights()):
            np.testing.assert_allclose(w_src, w_dst, atol=0, rtol=0)
        for w_src, w_dst in zip(src_comb_a[i].get_weights(), model_a.combine_layers[i].get_weights()):
            np.testing.assert_allclose(w_src, w_dst, atol=0, rtol=0)

        for w_src, w_dst in zip(src_meas_b[i].get_weights(), model_b.measure_layers[i].get_weights()):
            np.testing.assert_allclose(w_src, w_dst, atol=0, rtol=0)
        for w_src, w_dst in zip(src_comb_b[i].get_weights(), model_b.combine_layers[i].get_weights()):
            np.testing.assert_allclose(w_src, w_dst, atol=0, rtol=0)
