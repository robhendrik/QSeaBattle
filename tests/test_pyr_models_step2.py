import math
from dataclasses import dataclass

import tensorflow as tf

from Q_Sea_Battle.pyr_trainable_assisted_model_a import PyrTrainableAssistedModelA
from Q_Sea_Battle.pyr_trainable_assisted_model_b import PyrTrainableAssistedModelB


@dataclass
class DummyLayout:
    field_size: int
    comms_size: int = 1

    @property
    def n2(self) -> int:
        return self.field_size * self.field_size


def _k_from_n2(n2: int) -> int:
    return int(round(math.log2(n2)))


def test_step2_list_lengths_and_shapes():
    layout = DummyLayout(field_size=4, comms_size=1)  # n2=16 => K=4
    k = _k_from_n2(layout.n2)

    model_a = PyrTrainableAssistedModelA(layout, p_high=0.9, sr_mode="sample")
    model_b = PyrTrainableAssistedModelB(layout, p_high=0.9, sr_mode="sample")

    batch_size = 3
    field = tf.zeros((batch_size, layout.n2), dtype=tf.float32)
    gun = tf.one_hot([0, 5, 15], depth=layout.n2, dtype=tf.float32)

    comm_logits, meas_list, out_list = model_a.compute_with_internal(field)

    assert comm_logits.shape == (batch_size, 1)
    assert isinstance(meas_list, list) and isinstance(out_list, list)
    assert len(meas_list) == k
    assert len(out_list) == k

    # Per-level shapes: (B, n2/2^(ell+1))
    for ell in range(k):
        exp_len = layout.n2 // (2 ** (ell + 1))
        assert meas_list[ell].shape == (batch_size, exp_len)
        assert out_list[ell].shape == (batch_size, exp_len)

    # B call consumes the same structure
    comm_bits = tf.cast(tf.sigmoid(comm_logits) >= 0.5, tf.float32)
    shoot_logits = model_b([gun, comm_bits, meas_list, out_list])
    assert shoot_logits.shape == (batch_size, 1)


def test_step2_reproducibility_sample_mode():
    layout = DummyLayout(field_size=4, comms_size=1)
    model_a = PyrTrainableAssistedModelA(layout, p_high=0.9, sr_mode="sample")

    field = tf.constant([[0.0] * layout.n2], dtype=tf.float32)

    tf.random.set_seed(1234)
    c1, m1, o1 = model_a.compute_with_internal(field)

    tf.random.set_seed(1234)
    c2, m2, o2 = model_a.compute_with_internal(field)

    tf.debugging.assert_equal(c1, c2)
    for a, b in zip(m1, m2):
        tf.debugging.assert_equal(a, b)
    for a, b in zip(o1, o2):
        tf.debugging.assert_equal(a, b)


def test_step2_integration_and_alignment_errors():
    layout = DummyLayout(field_size=4, comms_size=1)
    k = _k_from_n2(layout.n2)

    model_a = PyrTrainableAssistedModelA(layout, p_high=0.9, sr_mode="sample")
    model_b = PyrTrainableAssistedModelB(layout, p_high=0.9, sr_mode="sample")

    field = tf.zeros((1, layout.n2), dtype=tf.float32)
    gun = tf.one_hot([3], depth=layout.n2, dtype=tf.float32)

    tf.random.set_seed(999)
    comm_logits, meas_list, out_list = model_a.compute_with_internal(field)
    comm_bits = tf.cast(tf.sigmoid(comm_logits) >= 0.5, tf.float32)

    # Sanity: does not raise
    shoot_logits_1 = model_b([gun, comm_bits, meas_list, out_list])
    shoot_logits_2 = model_b([gun, comm_bits, meas_list, out_list])
    tf.debugging.assert_equal(shoot_logits_1, shoot_logits_2)

    # Contract enforcement: wrong list length should raise
    try:
        _ = model_b([gun, comm_bits, meas_list[:-1], out_list])
        assert False, "Expected ValueError due to list length mismatch."
    except ValueError:
        pass

    # Contract enforcement: wrong per-level length should raise
    bad_out = list(out_list)
    bad_out[0] = tf.zeros((1, bad_out[0].shape[1] + 1), dtype=tf.float32)
    try:
        _ = model_b([gun, comm_bits, meas_list, bad_out])
        assert False, "Expected ValueError due to per-level length mismatch."
    except ValueError:
        pass
