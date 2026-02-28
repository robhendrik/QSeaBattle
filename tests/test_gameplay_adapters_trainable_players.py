"""
Tests for gameplay_adapters.py (pure-logit gameplay adapters).

These tests are self-contained: they only depend on TensorFlow + pytest and use
small fake internal models.

What we verify
--------------
1) GameplayModelAAdapter:
   - Accepts binary field bits (B,n2)
   - Converts to hard logits internally (we indirectly test via the fake model)
   - Returns comm_bits + meas_list_bits + out_list_bits as float32 bits {0,1}
2) GameplayModelBAdapter:
   - Accepts binary gun/comm and prev_* lists as bits
   - Converts all to hard logits internally
   - Returns shoot_bit as float32 bit {0,1}
3) Non-binary inputs trigger TensorFlow assertion errors when asserts are enabled.
"""

from __future__ import annotations

import pytest
import tensorflow as tf
import sys
sys.path.append("./src")
# Import the module under test.
# If your project uses a package path, adjust the import accordingly.
from Q_Sea_Battle.gameplay_adapters import (
    GameplayModelAAdapter,
    GameplayModelBAdapter
)


def _is_binary_tf(x: tf.Tensor) -> bool:
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    return bool(tf.reduce_all(tf.logical_or(tf.equal(x, 0.0), tf.equal(x, 1.0))).numpy())


class FakeInternalModelA:
    """
    Minimal internal Model A:
    - expects field_logits in {-beta_field, +beta_field}
    - returns comm_logits and per-level meas/out logits
    """

    def __init__(self, depth: int = 2):
        self.depth = depth
        self.harden_between_levels = False  # adapter may set this

    def compute_with_internal(self, field_logits: tf.Tensor, harden_between_levels=False, beta_for_hardening=None):
        field_logits = tf.convert_to_tensor(field_logits, tf.float32)  # (B,n2)
        # Record kwargs for interface tests
        self.seen = getattr(self, 'seen', {})
        self.seen['harden_between_levels'] = harden_between_levels
        self.seen['beta_for_hardening'] = beta_for_hardening

        # comm_logits is sum over bits => sign varies by number of ones
        comm_logits = tf.reduce_sum(field_logits, axis=-1, keepdims=True)  # (B,1)

        B = tf.shape(field_logits)[0]
        # Make logits that include both signs so binarization is meaningful.
        # Level 0: [-1, +1], Level 1: [+1, -1] (broadcasted to batch)
        meas0 = tf.tile(tf.constant([[-1.0, +1.0]], tf.float32), [B, 1])
        out0  = tf.tile(tf.constant([[+2.0, -2.0]], tf.float32), [B, 1])

        meas_list = [meas0]
        out_list = [out0]

        if self.depth >= 2:
            meas1 = tf.tile(tf.constant([[+3.0, -3.0]], tf.float32), [B, 1])
            out1  = tf.tile(tf.constant([[-4.0, +4.0]], tf.float32), [B, 1])
            meas_list.append(meas1)
            out_list.append(out1)

        return comm_logits, meas_list, out_list


class FakeInternalModelB:
    """
    Minimal internal Model B:
    - expects all inputs as hard logits in {-beta, +beta}
    - returns a shoot_logit based on comm sign (positive => shoot)

    The adapter calls:
        shoot_logit, *_ = internal_model_b.compute_with_internal(gun_logits, comm_logits, prev_meas_logits, prev_out_logits,
                                                                 harden_between_levels=..., beta_for_hardening=...)
    """
    def __init__(self):
        self.harden_between_levels = False  # kept for older code; adapter does not mutate this
        self.seen = {}

    def compute_with_internal(self, gun_logits, comm_logits, prev_meas_list, prev_out_list,
                              harden_between_levels=False, beta_for_hardening=None):
        gun_logits = tf.convert_to_tensor(gun_logits, tf.float32)
        comm_logits = tf.convert_to_tensor(comm_logits, tf.float32)
        prev_meas_list = [tf.convert_to_tensor(t, tf.float32) for t in prev_meas_list]
        prev_out_list = [tf.convert_to_tensor(t, tf.float32) for t in prev_out_list]

        # Record for assertions in the test
        self.seen["gun_logits"] = gun_logits
        self.seen["comm_logits"] = comm_logits
        self.seen["prev_meas_logits"] = prev_meas_list
        self.seen["prev_out_logits"] = prev_out_list
        self.seen["harden_between_levels"] = harden_between_levels
        self.seen["beta_for_hardening"] = beta_for_hardening

        # shoot_logit: use comm_logits directly; reduce to (B,1)
        shoot_logit = tf.reduce_sum(comm_logits, axis=-1, keepdims=True)
        return (shoot_logit,)


def test_gameplay_model_a_adapter_binarizes_outputs_and_sets_harden_flag():
    internal_a = FakeInternalModelA(depth=2)
    a = GameplayModelAAdapter(internal_a)

    # Adapter does not mutate internal_a; it passes harden_between_levels/beta_for_hardening into compute_with_internal.
    # field bits: two ones and two zeros => field_logits sums to 0 => comm_logits==0 => comm_bits==1 (>=0)
    field_bits = tf.constant([[1.0, 0.0, 1.0, 0.0]], tf.float32)  # (1,4)
    comm_bits, meas_list_bits, out_list_bits = a.compute_with_internal(field_bits)

    assert internal_a.seen['harden_between_levels'] == a.harden_between_levels
    assert internal_a.seen['beta_for_hardening'] == a.beta

    assert comm_bits.shape == (1, 1)
    assert comm_bits.dtype == tf.float32
    assert _is_binary_tf(comm_bits)
    assert float(comm_bits.numpy()[0, 0]) == 1.0

    assert isinstance(meas_list_bits, list) and isinstance(out_list_bits, list)
    assert len(meas_list_bits) == 2
    assert len(out_list_bits) == 2

    for t in meas_list_bits + out_list_bits:
        assert t.shape.rank == 2
        assert t.dtype == tf.float32
        assert _is_binary_tf(t)

    # Check exact binarization behavior for our fake logits
    # meas0 = [-1,+1] -> [0,1]
    assert meas_list_bits[0].numpy().tolist() == [[0.0, 1.0]]
    # out0 = [+2,-2] -> [1,0]
    assert out_list_bits[0].numpy().tolist() == [[1.0, 0.0]]
    # meas1 = [+3,-3] -> [1,0]
    assert meas_list_bits[1].numpy().tolist() == [[1.0, 0.0]]
    # out1 = [-4,+4] -> [0,1]
    assert out_list_bits[1].numpy().tolist() == [[0.0, 1.0]]


def test_gameplay_model_b_adapter_converts_inputs_to_hard_logits_and_binarizes_shoot():
    internal_b = FakeInternalModelB()
    b = GameplayModelBAdapter(internal_b)

    # Adapter does not mutate internal_b; it passes harden_between_levels/beta_for_hardening into compute_with_internal.
    beta = b.beta

    gun_bits = tf.constant([[0.0, 0.0, 1.0, 0.0]], tf.float32)   # (1,4)
    comm_bits = tf.constant([[1.0]], tf.float32)                 # (1,1) => +beta
    prev_meas_bits = [tf.constant([[0.0, 1.0]], tf.float32)]     # one level
    prev_out_bits  = [tf.constant([[1.0, 0.0]], tf.float32)]     # one level

    shoot_bit = b([gun_bits, comm_bits, prev_meas_bits, prev_out_bits])

    assert internal_b.seen['harden_between_levels'] == b.harden_between_levels
    assert internal_b.seen['beta_for_hardening'] == b.beta

    assert shoot_bit.shape == (1, 1)
    assert shoot_bit.dtype == tf.float32
    assert _is_binary_tf(shoot_bit)
    assert float(shoot_bit.numpy()[0, 0]) == 1.0  # comm is positive => shoot

    # Verify internal model saw hard logits exactly ±beta
    gun_logits_seen = internal_b.seen["gun_logits"].numpy()
    comm_logits_seen = internal_b.seen["comm_logits"].numpy()
    prev_meas_seen = internal_b.seen["prev_meas_logits"][0].numpy()
    prev_out_seen = internal_b.seen["prev_out_logits"][0].numpy()

    # gun one-hot: one entry +beta, rest -beta
    assert set(gun_logits_seen.reshape(-1).tolist()) <= {+beta, -beta}
    assert gun_logits_seen[0, 2] == +beta

    # comm: +beta
    assert comm_logits_seen.shape == (1, 1)
    assert comm_logits_seen[0, 0] == +beta

    # prev bits mapped to logits
    assert prev_meas_seen.tolist() == [[-beta, +beta]]
    assert prev_out_seen.tolist() == [[+beta, -beta]]


def test_adapters_reject_non_binary_inputs_when_asserts_enabled():
    internal_a = FakeInternalModelA(depth=1)
    a = GameplayModelAAdapter(internal_a)

    # Non-binary field value should trigger tf.debugging assertion
    bad_field = tf.constant([[0.2, 0.0, 1.0, 0.0]], tf.float32)
    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = a.compute_with_internal(bad_field)

    internal_b = FakeInternalModelB()
    b = GameplayModelBAdapter(internal_b)

    gun_bits = tf.constant([[0.0, 0.0, 1.0, 0.0]], tf.float32)
    bad_comm = tf.constant([[0.3]], tf.float32)
    prev_meas_bits = [tf.constant([[0.0, 1.0]], tf.float32)]
    prev_out_bits = [tf.constant([[1.0, 0.0]], tf.float32)]

    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = b([gun_bits, bad_comm, prev_meas_bits, prev_out_bits])
