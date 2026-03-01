"""
Tests for pyr_internal_model_a.py / pyr_internal_model_b.py (contract-aligned, logits-only outputs).

Covers:
- Shapes match the contract.
- In replay mode, Model A's outcomes equal the provided replay outcomes by construction.
- Models accept converter outputs (cropped lists k_d).
- One gradient step runs for Model A (sanity that model is trainable).
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]  # QSeaBattle/
WIP_SRC = ROOT / "WIP" / "src"
sys.path.insert(0, str(WIP_SRC))

from Q_Sea_Battle_New.pyr_dataset_generation_utilities import generate_pyr_dataset
from Q_Sea_Battle_New.pyr_dataset_conversion_utilities import (
    convert_internal_model_a,
    convert_internal_model_b,
)
from Q_Sea_Battle_New.pyr_internal_model_a import PyrInternalModelA
from Q_Sea_Battle_New.pyr_internal_model_b import PyrInternalModelB

class _Layout:
    def __init__(self, n2: int):
        self.n2 = n2
        self.comms_size = 1

def test_internal_model_a_replay_outcomes_identity():
    ds = generate_pyr_dataset(n2=16, num_games=3, seed=0, validate=True)
    field_0, comm_final_t, meas_t_list, out_t_list = convert_internal_model_a(
        ds,
        rep_field="scaled",
        rep_comm_target="hard_logit",
        rep_meas_target="hard_logit",
        rep_out_target="hard_logit",
        beta=10.0,
    )
    layout = _Layout(16)
    model_a = PyrInternalModelA(layout, sr_mode="replay", beta=10.0, alpha=5.0)

    comm_logits, meas_list, out_list = model_a.compute_with_internal(
        tf.convert_to_tensor(field_0),
        replay_out_a_logits_list=[tf.convert_to_tensor(t) for t in out_t_list],
        training=True,
    )

    assert comm_logits.shape == (3, 1)
    assert comm_logits.dtype == tf.float32
    assert len(meas_list) == model_a.depth
    assert len(out_list) == model_a.depth
    assert len(meas_list) == len(out_list) == model_a.depth
    # Out list must equal provided replay list exactly (identity) in replay mode.
    for d in range(model_a.depth):
        np.testing.assert_allclose(out_list[d].numpy(), out_t_list[d], rtol=0, atol=0)

def test_internal_model_b_accepts_converter_outputs_and_shapes():
    ds = generate_pyr_dataset(n2=16, num_games=4, seed=1, validate=True)

    gun_0, comm_0, prev_meas_list, prev_out_list, _, _, shoot_t = convert_internal_model_b(
        ds,
        rep_gun="scaled",
        rep_comm_in="hard_logit",
        rep_prev_meas="hard_logit",
        rep_prev_out="hard_logit",
        rep_shoot_target="bits",
        beta=10.0,
    )
    layout = _Layout(16)
    model_b = PyrInternalModelB(layout, sr_mode="replay", beta=10.0, alpha=5.0)
    print("input:",gun_0)
    shoot_logit = model_b(
        [
            tf.convert_to_tensor(gun_0),
            tf.convert_to_tensor(comm_0),
            [tf.convert_to_tensor(t) for t in prev_meas_list],
            [tf.convert_to_tensor(t) for t in prev_out_list],
        ],
        training=False,
    )

    assert shoot_logit.shape == (4, 1)
    assert shoot_logit.dtype == tf.float32
    # logits not hardened: allow outside [-10,10] but at least not constrained to [0,1]
    mx = float(tf.reduce_max(shoot_logit).numpy())
    mn = float(tf.reduce_min(shoot_logit).numpy())
    assert (mx > 1.0) or (mn < 0.0)

def test_internal_model_a_one_gradient_step_runs():
    ds = generate_pyr_dataset(n2=16, num_games=8, seed=2, validate=True)
    field_0, comm_final_t, meas_t_list, out_t_list = convert_internal_model_a(
        ds,
        rep_field="scaled",
        rep_comm_target="hard_logit",
        rep_meas_target="hard_logit",
        rep_out_target="hard_logit",
        beta=10.0,
    )
    layout = _Layout(16)
    model_a = PyrInternalModelA(layout, sr_mode="replay", beta=10.0, alpha=5.0)

    opt = tf.keras.optimizers.Adam(learning_rate=0.01)

    # Simple loss on final comm logit vs target (hard_logit teacher)
    with tf.GradientTape() as tape:
        comm_logits, _, _ = model_a.compute_with_internal(
            tf.convert_to_tensor(field_0),
            replay_out_a_logits_list=[tf.convert_to_tensor(t) for t in out_t_list],
            training=True,
        )
        loss = tf.reduce_mean(tf.square(comm_logits - tf.convert_to_tensor(comm_final_t)))
    grads = tape.gradient(loss, model_a.trainable_variables)
    assert all(g is not None for g in grads), "Some gradients are None."

    opt.apply_gradients(zip(grads, model_a.trainable_variables))
    # Recompute loss using replay outcomes again (PRAssistedReplay in replay mode requires replay_outcome_logits).
    comm_logits2, _, _ = model_a.compute_with_internal(
        tf.convert_to_tensor(field_0),
        replay_out_a_logits_list=[tf.convert_to_tensor(t) for t in out_t_list],
        training=False,
    )
    loss2 = tf.reduce_mean(tf.square(comm_logits2 - tf.convert_to_tensor(comm_final_t)))
    assert np.isfinite(loss.numpy())
    assert np.isfinite(loss2.numpy())