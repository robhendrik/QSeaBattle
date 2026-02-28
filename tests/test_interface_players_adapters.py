# test_interface_players_adapters.py
import numpy as np
import tensorflow as tf
import pytest

from Q_Sea_Battle.gameplay_adapters import GameplayModelAAdapter, GameplayModelBAdapter
from Q_Sea_Battle.trainable_assisted_player_a import TrainableAssistedPlayerA
from Q_Sea_Battle.trainable_assisted_player_b import TrainableAssistedPlayerB


# -------------------------
# Dummy internal models
# -------------------------

class DummyInternalA:
    """Returns LOGITS; adapter thresholds to bits."""
    def compute_with_internal(self, field_logits: tf.Tensor, harden_between_levels=False, beta_for_hardening=None):
        B = tf.shape(field_logits)[0]
        comm_logits = tf.ones((B, 1), tf.float32) * 2.0

        # Keep simple consistent widths; adapter only thresholds.
        meas_list = [tf.ones((B, 4), tf.float32) * 3.0, tf.ones((B, 2), tf.float32) * -3.0]
        out_list  = [tf.ones((B, 4), tf.float32) * -1.0, tf.ones((B, 2), tf.float32) *  1.0]
        return comm_logits, meas_list, out_list


class DummyInternalB:
    """Internal B with compute_with_internal. Returns (shoot_logit,)."""
    def compute_with_internal(self, gun_logits, comm_logits, prev_meas_logits, prev_out_logits,
                              harden_between_levels=False, beta_for_hardening=None):
        B = tf.shape(gun_logits)[0]
        shoot_logit = tf.ones((B, 1), tf.float32) * 1.5  # positive => shoot=1 deterministically
        return (shoot_logit,)


class DummyLayout:
    field_size = 2   # n2 = 4
    comms_size = 1


# -------------------------
# Adapter A tests
# -------------------------

def test_adapter_a_call_default_returns_3():
    adapter = GameplayModelAAdapter(DummyInternalA(), beta=10.0)
    field_bits = tf.constant([[0, 1, 0, 1]], dtype=tf.float32)  # (1,4)

    comm_bits, meas_bits, out_bits = adapter(field_bits)

    assert comm_bits.shape == (1, 1)
    assert comm_bits.dtype == tf.float32
    assert len(meas_bits) == 2
    assert len(out_bits) == 2
    assert np.all(np.isin(comm_bits.numpy(), [0.0, 1.0]))


def test_adapter_a_call_return_comm_logits_returns_4():
    adapter = GameplayModelAAdapter(DummyInternalA(), beta=10.0)
    field_bits = tf.constant([[0, 1, 0, 1]], dtype=tf.float32)

    comm_bits, meas_bits, out_bits, comm_logits = adapter(field_bits, return_comm_logits=True)

    assert comm_logits.shape == (1, 1)
    assert comm_bits.shape == (1, 1)


def test_adapter_a_deprecated_compute_with_internal_matches_call():
    adapter = GameplayModelAAdapter(DummyInternalA(), beta=10.0)
    field_bits = tf.constant([[0, 1, 0, 1]], dtype=tf.float32)

    out1 = adapter(field_bits)
    out2 = adapter.compute_with_internal(field_bits)

    # compare comm bits exactly
    tf.debugging.assert_equal(out1[0], out2[0])
    assert len(out1[1]) == len(out2[1])
    assert len(out1[2]) == len(out2[2])


def test_adapter_a_explore_kwarg_runs_and_preserves_shape():
    adapter = GameplayModelAAdapter(DummyInternalA(), beta=10.0)
    field_bits = tf.constant([[0, 1, 0, 1]], dtype=tf.float32)

    tf.random.set_seed(123)
    comm_bits, _, _ = adapter(field_bits, explore=True)

    assert comm_bits.shape == (1, 1)
    assert np.all(np.isin(comm_bits.numpy(), [0.0, 1.0]))


# -------------------------
# Adapter B tests
# -------------------------

def test_adapter_b_call_default_returns_1():
    adapter = GameplayModelBAdapter(DummyInternalB(), beta=10.0)

    gun_bits  = tf.constant([[0, 1, 0, 0]], dtype=tf.float32)
    comm_bits = tf.constant([[1]], dtype=tf.float32)

    prev_meas = [tf.constant([[0, 1]], dtype=tf.float32)]
    prev_out  = [tf.constant([[1, 0]], dtype=tf.float32)]

    shoot_bit = adapter([gun_bits, comm_bits, prev_meas, prev_out])

    assert shoot_bit.shape == (1, 1)
    assert np.all(np.isin(shoot_bit.numpy(), [0.0, 1.0]))


def test_adapter_b_call_return_shoot_logit_returns_2():
    adapter = GameplayModelBAdapter(DummyInternalB(), beta=10.0)

    gun_bits  = tf.constant([[0, 1, 0, 0]], dtype=tf.float32)
    comm_bits = tf.constant([[1]], dtype=tf.float32)

    prev_meas = [tf.constant([[0, 1]], dtype=tf.float32)]
    prev_out  = [tf.constant([[1, 0]], dtype=tf.float32)]

    shoot_bit, shoot_logit = adapter([gun_bits, comm_bits, prev_meas, prev_out], return_shoot_logit=True)

    assert shoot_bit.shape == (1, 1)
    assert shoot_logit.shape == (1, 1)


def test_adapter_b_deprecated_compute_with_internal_matches_call():
    adapter = GameplayModelBAdapter(DummyInternalB(), beta=10.0)

    gun_bits  = tf.constant([[0, 1, 0, 0]], dtype=tf.float32)
    comm_bits = tf.constant([[1]], dtype=tf.float32)

    prev_meas = [tf.constant([[0, 1]], dtype=tf.float32)]
    prev_out  = [tf.constant([[1, 0]], dtype=tf.float32)]

    shoot1 = adapter([gun_bits, comm_bits, prev_meas, prev_out])
    shoot2 = adapter.compute_with_internal([gun_bits, comm_bits, prev_meas, prev_out])

    tf.debugging.assert_equal(shoot1, shoot2)


def test_adapter_b_rejects_nonbinary_comm_bits_policy_A():
    """Policy A regression guard: adapter must reject DRU-style floats."""
    adapter = GameplayModelBAdapter(DummyInternalB(), beta=10.0)

    gun_bits  = tf.constant([[0, 1, 0, 0]], dtype=tf.float32)
    comm_bad  = tf.constant([[0.3]], dtype=tf.float32)  # not binary

    prev_meas = [tf.constant([[0, 1]], dtype=tf.float32)]
    prev_out  = [tf.constant([[1, 0]], dtype=tf.float32)]

    # _assert_binary_tensor should trip (InvalidArgumentError typically)
    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = adapter([gun_bits, comm_bad, prev_meas, prev_out])


# -------------------------
# Player ↔ Adapter integration tests
# -------------------------

def test_player_a_with_adapter_sets_previous_and_returns_comm():
    layout = DummyLayout()
    adapter = GameplayModelAAdapter(DummyInternalA(), beta=10.0)
    player_a = TrainableAssistedPlayerA(layout, model_a=adapter)

    class Parent: pass
    parent = Parent()
    parent.previous = None
    player_a.parent = parent

    comm = player_a.decide(field=np.array([0, 1, 0, 1], dtype=np.int32))

    assert comm.shape == (layout.comms_size,)
    assert parent.previous is not None


def test_player_b_with_adapter_returns_int_and_uses_previous():
    layout = DummyLayout()
    adapter_b = GameplayModelBAdapter(DummyInternalB(), beta=10.0)
    player_b = TrainableAssistedPlayerB(layout, model_b=adapter_b)

    class Parent: pass
    parent = Parent()
    parent.previous = (
        [tf.constant([[0, 1]], tf.float32)],
        [tf.constant([[1, 0]], tf.float32)],
    )
    player_b.parent = parent

    shoot = player_b.decide(
        gun=np.array([0, 1, 0, 0], dtype=np.int32),
        comm=np.array([1], dtype=np.int32),
    )

    assert shoot in (0, 1)


def test_player_b_with_adapter_rejects_dru_float_comm_policy_A():
    """Policy A regression guard at player boundary: comm must be binary when using adapter."""
    layout = DummyLayout()
    adapter_b = GameplayModelBAdapter(DummyInternalB(), beta=10.0)
    player_b = TrainableAssistedPlayerB(layout, model_b=adapter_b)

    class Parent: pass
    parent = Parent()
    parent.previous = (
        [tf.constant([[0, 1]], tf.float32)],
        [tf.constant([[1, 0]], tf.float32)],
    )
    player_b.parent = parent

    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = player_b.decide(
            gun=np.array([0, 1, 0, 0], dtype=np.int32),
            comm=np.array([0.3], dtype=np.float32),  # should fail under A
        )