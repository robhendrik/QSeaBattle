"""
Tests for dataset_conversion_utility.py.

These tests validate:
- correct d-ranges (0..depth-1 for per-level converters)
- cropping to L_d / k_d
- representation mappings (scaled, hard_logit)
- output types float32 and shapes expected by trainable objects

Repo layout expected:
QSeaBattle/
  WIP/
    src/Q_Sea_Battle_New/pyr_dataset_conversion_utility.py
    src/Q_Sea_Battle_New/pyr_dataset_generation_utilities.py
  WIP/tests/test_pyr_dataset_conversion_utility.py
"""

import sys
from pathlib import Path
import numpy as np

# Ensure WIP/src is on sys.path
ROOT = Path(__file__).resolve().parents[2]  # QSeaBattle/
WIP_SRC = ROOT / "WIP" / "src"
sys.path.insert(0, str(WIP_SRC))

from Q_Sea_Battle_New.pyr_dataset_generation_utilities import generate_pyr_dataset
from Q_Sea_Battle_New.pyr_dataset_conversion_utilities import (
    level_sizes,
    convert_layer_measure_a,
    convert_layer_combine_a,
    convert_layer_measure_b,
    convert_layer_combine_b,
    convert_internal_model_a,
    convert_internal_model_b,
    convert_full_system,
    apply_rep,
)

def _assert_float32(x: np.ndarray) -> None:
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float32

def _assert_allclose(x, y):
    np.testing.assert_allclose(x, y, rtol=0, atol=0)

def test_rep_mappings():
    bits = np.array([[0.0, 1.0]], dtype=np.float32)
    scaled = apply_rep(bits, "scaled", beta=10.0)
    hard = apply_rep(bits, "hard_logit", beta=10.0)
    _assert_allclose(scaled, np.array([[-0.5, 0.5]], dtype=np.float32))
    _assert_allclose(hard, np.array([[-10.0, 10.0]], dtype=np.float32))

def test_apply_rep_mixed_beta_hard_logit_interleave():
    # N=5, K=2 -> np.array_split gives [0,1,2] and [3,4]
    # Interleave should be: 0,3,1,4,2
    bits = np.array([[0.0], [1.0], [0.0], [1.0], [1.0]], dtype=np.float32)  # shape (5,1)
    out = apply_rep(bits, "hard_logit", beta=(1.0, 2.0))

    # Chunk0 beta=1: [ -1, +1, -1 ] ; Chunk1 beta=2: [ +2, +2 ]
    # Interleave: idx0(-1), idx3(+2), idx1(+1), idx4(+2), idx2(-1)
    expected = np.array([[-1.0], [ 2.0], [ 1.0], [ 2.0], [-1.0]], dtype=np.float32)
    _assert_allclose(out, expected)
    _assert_float32(out)
    assert out.shape == (5, 1)

def test_apply_rep_mixed_beta_bits_and_scaled_interleave_shapes():
    # bits/scaled should also support mixed-beta (beta ignored for bits/scaled),
    # but ordering must follow the same deterministic split+interleave.
    bits = np.arange(12, dtype=np.float32).reshape(6, 2)
    # Make it 0/1 like canonical bits by thresholding
    bits = (bits % 2).astype(np.float32)

    out_bits = apply_rep(bits, "bits", beta=(0.5, 3.0, 10.0))
    out_scaled = apply_rep(bits, "scaled", beta=(0.5, 3.0, 10.0))

    _assert_float32(out_bits); _assert_float32(out_scaled)
    assert out_bits.shape == bits.shape
    assert out_scaled.shape == bits.shape

    # scaled must be bits - 0.5 elementwise (regardless of beta)
    _assert_allclose(out_scaled, out_bits - np.float32(0.5))

def test_apply_rep_beta_sequence_len1_matches_scalar():
    bits = np.array([[0.0, 1.0, 1.0]], dtype=np.float32)
    a = apply_rep(bits, "hard_logit", beta=3.0)
    b = apply_rep(bits, "hard_logit", beta=(3.0,))
    _assert_allclose(a, b)

def test_apply_rep_empty_beta_sequence_raises():
    bits = np.array([[0.0, 1.0]], dtype=np.float32)
    try:
        _ = apply_rep(bits, "hard_logit", beta=())
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty beta sequence")

def test_converters_shapes_and_cropping_n2_4_16_64():
    for n2 in (4, 16, 64):
        ds = generate_pyr_dataset(n2=n2, num_games=3, seed=0, validate=True)
        depth = ds["meas_in_a_bits"].shape[1]
        assert depth == int(np.log2(n2))

        # --- Measure A ---
        mA = convert_layer_measure_a(ds, rep_x="scaled", rep_y="hard_logit", beta=10.0)
        assert set(mA.keys()) == set(range(depth))
        for d, (X, Y) in mA.items():
            L_d, k_d = level_sizes(n2, d)
            _assert_float32(X); _assert_float32(Y)
            assert X.shape == (3, L_d)
            assert Y.shape == (3, k_d)

        # --- Combine A ---
        cA = convert_layer_combine_a(ds, rep_field="scaled", rep_outcome="hard_logit", rep_target="hard_logit", beta=10.0)
        assert set(cA.keys()) == set(range(depth))
        for d, ((field_d, out_a_d), field_d1) in cA.items():
            L_d, k_d = level_sizes(n2, d)
            L_d1, _ = level_sizes(n2, d + 1)
            assert field_d.shape == (3, L_d)
            assert out_a_d.shape == (3, k_d)
            assert field_d1.shape == (3, L_d1)

        # --- Measure B ---
        mB = convert_layer_measure_b(ds, rep_x="scaled", rep_y="hard_logit", beta=10.0)
        assert set(mB.keys()) == set(range(depth))
        for d, (X, Y) in mB.items():
            L_d, k_d = level_sizes(n2, d)
            assert X.shape == (3, L_d)
            assert Y.shape == (3, k_d)

        # --- Combine B ---
        cB = convert_layer_combine_b(ds, rep_gun="scaled", rep_outcome_b="hard_logit",
                                     rep_comm_in="hard_logit", rep_gun_next="hard_logit",
                                     rep_comm_next="hard_logit", beta=10.0)
        assert set(cB.keys()) == set(range(depth))
        for d, ((gun_d, out_b_d, comm_d), (gun_d1, comm_d1)) in cB.items():
            L_d, k_d = level_sizes(n2, d)
            L_d1, _ = level_sizes(n2, d + 1)
            assert gun_d.shape == (3, L_d)
            assert out_b_d.shape == (3, k_d)
            assert comm_d.shape == (3, 1)
            assert gun_d1.shape == (3, L_d1)
            assert comm_d1.shape == (3, 1)

        # --- Internal Model A ---
        field_0, comm_final, meas_list, out_list = convert_internal_model_a(ds, rep_field="scaled",
                                                                            rep_comm_target="hard_logit",
                                                                            rep_meas_target="hard_logit",
                                                                            rep_out_target="hard_logit",
                                                                            beta=10.0)
        assert field_0.shape == (3, n2)
        assert comm_final.shape == (3, 1)
        assert len(meas_list) == depth
        assert len(out_list) == depth
        for d in range(depth):
            _, k_d = level_sizes(n2, d)
            assert meas_list[d].shape == (3, k_d)
            assert out_list[d].shape == (3, k_d)

        # --- Internal Model B ---
        gun_0, comm_0, prev_meas, prev_out, _, _,shoot = convert_internal_model_b(ds, rep_gun="scaled",
                                                                             rep_comm_in="hard_logit",
                                                                             rep_prev_meas="hard_logit",
                                                                             rep_prev_out="hard_logit",
                                                                             rep_shoot_target="bits",
                                                                             beta=10.0)
        assert gun_0.shape == (3, n2)
        assert comm_0.shape == (3, 1)
        assert shoot.shape == (3, 1)
        assert len(prev_meas) == depth
        assert len(prev_out) == depth
        for d in range(depth):
            _, k_d = level_sizes(n2, d)
            assert prev_meas[d].shape == (3, k_d)
            assert prev_out[d].shape == (3, k_d)

        # --- Full system ---
        field0, gun0, comm_trace, t_meas, t_out, shoot_t = convert_full_system(
            ds, rep_field="scaled", rep_gun="scaled",
            rep_teacher_comm_trace="bits",
            rep_teacher_meas_a="bits",
            rep_teacher_out_a="bits",
            rep_shoot_target="bits",
            beta=10.0,
        )
        assert field0.shape == (3, n2)
        assert gun0.shape == (3, n2)
        assert comm_trace.shape == (3, depth + 1, 1)
        assert len(t_meas) == depth
        assert len(t_out) == depth
        assert shoot_t.shape == (3, 1)
        for d in range(depth):
            _, k_d = level_sizes(n2, d)
            assert t_meas[d].shape == (3, k_d)
            assert t_out[d].shape == (3, k_d)

def test_cropping_matches_prefix_for_n2_16_level0_level1():
    # Stronger check: ensure cropped arrays equal canonical prefix exactly (bits rep).
    n2 = 16
    ds = generate_pyr_dataset(n2=n2, num_games=2, seed=1, validate=True)
    depth = ds["meas_in_a_bits"].shape[1]

    mA_bits = convert_layer_measure_a(ds, rep_x="bits", rep_y="bits", beta=10.0)
    # d=0 => L_0=16, k_0=8
    X0, Y0 = mA_bits[0]
    assert np.array_equal(X0, ds["field_bits"][:, 0, :16])
    assert np.array_equal(Y0, ds["meas_in_a_bits"][:, 0, :8])

    # d=1 => L_1=8, k_1=4
    X1, Y1 = mA_bits[1]
    assert np.array_equal(X1, ds["field_bits"][:, 1, :8])
    assert np.array_equal(Y1, ds["meas_in_a_bits"][:, 1, :4])
