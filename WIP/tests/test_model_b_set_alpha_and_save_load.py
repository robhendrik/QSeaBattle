# test_model_b_set_alpha_and_save_load.py
import os
import tempfile
import numpy as np
import tensorflow as tf

from Q_Sea_Battle_New.pyr_internal_model_a import PyrInternalModelA
from Q_Sea_Battle_New.pyr_internal_model_b import PyrInternalModelB


class _Layout:
    def __init__(self, n2: int, comms_size: int = 1):
        self.n2 = n2
        self.comms_size = comms_size


def _build_model_a_replay(model_a: PyrInternalModelA, n2: int, batch: int = 1):
    """
    Build model_a in sr_mode='replay' by calling compute_with_internal with a
    correctly-shaped replay_out_a_logits_list for every level.

    IMPORTANT: do NOT call model_a(x) in replay-mode; that routes through call()
    which uses replay_out_a_logits_list=None and will (correctly) raise.
    """
    x = tf.zeros((batch, n2), dtype=tf.float32) - 0.5  # scaled input in {-0.5,+0.5}

    replay_out = []
    state = x
    for d in range(model_a.depth):
        # build measurement output shape (B,k_d)
        meas = model_a.measure_layers[d](state, training=False)
        replay_out.append(tf.zeros_like(meas))  # teacher logits for SR replay
        # advance state (just to build combine layer output shape)
        state = model_a.combine_layers[d](state, tf.zeros_like(meas), training=False)

    # This builds SR layers too (the part that previously failed).
    _ = model_a.compute_with_internal(x, replay_out_a_logits_list=replay_out, training=False)
    return x, replay_out


def _build_model_b(model_b: PyrInternalModelB, n2: int, batch: int = 1):
    """
    Build model_b (replay mode) with valid float32 shapes/dtypes.
    """
    depth = int(np.log2(n2))

    b = np.zeros((batch, n2), dtype=np.float32)  # float32
    gun = np.float32(b)  # float32
    c = np.zeros((batch, 1), dtype=np.float32)  # float32
    comm = np.float32(c)  # float32

    k = n2 // 2
    prev_meas = []
    prev_out = []
    for _ in range(depth):
        prev_meas.append(tf.zeros((batch, k), tf.float32))
        prev_out.append(tf.zeros((batch, k), tf.float32))
        k //= 2

    # Either route is fine; compute_with_internal is the most explicit.
    _ = model_b.compute_with_internal(gun, comm, prev_meas, prev_out, training=False)
    return gun, comm, prev_meas, prev_out


def test_model_b_set_alpha_updates_all_sr_layers():
    n2 = 4
    layout = _Layout(n2=n2, comms_size=1)
    model_b = PyrInternalModelB(layout, sr_mode="replay", alpha=1.0, beta=10.0, seed=123)

    assert hasattr(model_b, "set_alpha"), "Expected PyrInternalModelB.set_alpha(alpha)"

    _build_model_b(model_b, n2)

    model_b.set_alpha(7.5)

    for sr in model_b.sr_layers:
        # Your intended implementation can store alpha either in sr._alpha (tf.Variable)
        # or inside sr._cfg.alpha. Accept both.
        if hasattr(sr, "_alpha"):
            val = float(sr._alpha.numpy())
        elif hasattr(sr, "_cfg") and hasattr(sr._cfg, "alpha"):
            val = float(sr._cfg.alpha)
        else:
            raise AssertionError("SR layer has neither _alpha nor _cfg.alpha; cannot verify set_alpha()")
        assert abs(val - 7.5) < 1e-6


def test_model_a_save_load_roundtrip_preserves_outputs(tmp_path):
    n2 = 4
    layout = _Layout(n2=n2, comms_size=1)

    model_a = PyrInternalModelA(layout, sr_mode="replay", beta=10.0, seed=123)
    x, replay_out = _build_model_a_replay(model_a, n2, batch=2)

    comm1, meas1, out1 = model_a.compute_with_internal(x, replay_out_a_logits_list=replay_out, training=False)

    save_path = tmp_path / "model_a.weights.h5"
    model_a.save_weights_to(str(save_path))

    model_a2 = PyrInternalModelA(layout, sr_mode="replay", beta=10.0, seed=999)
    x2, replay_out2 = _build_model_a_replay(model_a2, n2, batch=2)

    model_a2.load_weights_from(str(save_path))

    comm2, meas2, out2 = model_a2.compute_with_internal(x2, replay_out_a_logits_list=replay_out2, training=False)
    print(comm1, comm2)
    tf.debugging.assert_near(comm1, comm2, atol=1e-6, rtol=0.0)
    for a, b in zip(meas1, meas2):
        tf.debugging.assert_near(a, b, atol=1e-6, rtol=0.0)
    for a, b in zip(out1, out2):
        tf.debugging.assert_near(a, b, atol=1e-6, rtol=0.0)


def test_model_b_save_load_roundtrip_preserves_outputs():
    n2 = 4
    layout = _Layout(n2=n2, comms_size=1)
    model_b = PyrInternalModelB(layout, sr_mode="replay", alpha=3.0, beta=10.0, seed=123)

    _build_model_b(model_b, n2, batch=2)

    gun = tf.random.uniform((2, n2), minval=-1.0, maxval=1.0, dtype=tf.float32, seed=2)
    comm = tf.random.uniform((2, 1), minval=-1.0, maxval=1.0, dtype=tf.float32, seed=3)

    depth = int(np.log2(n2))
    k = n2 // 2
    prev_meas = []
    prev_out = []
    for d in range(depth):
        prev_meas.append(tf.random.uniform((2, k), minval=-1.0, maxval=1.0, dtype=tf.float32, seed=10 + d))
        prev_out.append(tf.random.uniform((2, k), minval=-1.0, maxval=1.0, dtype=tf.float32, seed=20 + d))
        k //= 2

    y_before = model_b.compute_with_internal(gun, comm, prev_meas, prev_out, training=False)[0]

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "model_b.weights.h5")
        model_b.save_weights_to(path)

        # perturb weights
        for v in model_b.trainable_variables:
            v.assign(tf.zeros_like(v))

        y_zeroed = model_b.compute_with_internal(gun, comm, prev_meas, prev_out, training=False)[0]
        assert not tf.reduce_all(tf.equal(y_before, y_zeroed)), "Weights perturbation did not change output"

        model_b.load_weights_from(path)
        y_after = model_b.compute_with_internal(gun, comm, prev_meas, prev_out, training=False)[0]

    tf.debugging.assert_near(y_before, y_after, atol=1e-6)

