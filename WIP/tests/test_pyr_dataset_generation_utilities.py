import numpy as np
import pytest

# Utility under test lives in WIP/src/Q_Sea_Battle_New
from Q_Sea_Battle_New.pyr_dataset_generation_utilities import (
    generate_one_game_trace_pyr,
    pyr_sizes,
    teacher_measure_a,
    teacher_combine_a,
    teacher_measure_b,
    teacher_combine_b,
)


def _one_hot_index(x: np.ndarray, L: int) -> int:
    prefix = x[:L]
    idx = np.flatnonzero(prefix == 1.0)
    assert idx.size == 1, f"expected one-hot in prefix length {L}, got indices {idx}"
    return int(idx[0])


def _assert_binary(arr: np.ndarray, name: str) -> None:
    u = np.unique(arr)
    assert np.all((u == 0.0) | (u == 1.0)), f"{name} not binary: unique={u}"


def _assert_padded_zeros(vec: np.ndarray, L: int, name: str) -> None:
    if L < vec.size:
        tail = vec[L:]
        assert np.all(tail == 0.0), f"{name} padding not zero beyond L={L}"


@pytest.mark.parametrize("n2", [4, 16, 64])
def test_shapes_dtypes_depth_and_binary(n2: int) -> None:
    s = pyr_sizes(n2)
    g = generate_one_game_trace_pyr(n2=n2, seed=123, validate=True)

    depth = s.depth
    assert depth == int(np.log2(n2))
    assert g["field_bits"].shape == (depth + 1, n2)
    assert g["gun_bits"].shape == (depth + 1, n2)
    assert g["comms_bits"].shape == (depth + 1, 1)

    assert g["meas_in_a_bits"].shape == (depth, n2)
    assert g["meas_out_a_bits"].shape == (depth, n2)
    assert g["meas_in_b_bits"].shape == (depth, n2)
    assert g["meas_out_b_bits"].shape == (depth, n2)

    assert g["shoot"].shape == (1,)

    for k, v in g.items():
        assert isinstance(v, np.ndarray), f"{k} must be np.ndarray"
        assert v.dtype == np.float32, f"{k} must be float32, got {v.dtype}"
        _assert_binary(v, k)


@pytest.mark.parametrize("n2", [4, 16, 64])
def test_shoot_is_winning_bit(n2: int) -> None:
    s = pyr_sizes(n2)
    g = generate_one_game_trace_pyr(n2=n2, seed=7, validate=True)

    # gun at level 0 must be one-hot in prefix length n2
    idx = _one_hot_index(g["gun_bits"][0], s.L[0])
    assert g["shoot"][0] == g["field_bits"][0, idx], "shoot must match field bit at gun index"


@pytest.mark.parametrize("n2", [4, 16, 64])
def test_transition_logic_and_padding(n2: int) -> None:
    s = pyr_sizes(n2)
    g = generate_one_game_trace_pyr(n2=n2, seed=999, validate=True)

    field_bits = g["field_bits"]
    gun_bits = g["gun_bits"]
    comms_bits = g["comms_bits"]
    meas_in_a_bits = g["meas_in_a_bits"]
    meas_out_a_bits = g["meas_out_a_bits"]
    meas_in_b_bits = g["meas_in_b_bits"]
    meas_out_b_bits = g["meas_out_b_bits"]

    depth = s.depth

    # --- A-side reduction: new_field from old_field and meas_out_a ---
    for d in range(depth):
        Ld = s.L[d]
        kd = s.k[d]

        fp = field_bits[d, :Ld]
        _assert_padded_zeros(field_bits[d], Ld, f"field_bits[{d}]")

        # meas_in_a must be teacher_measure_a(fp) (length kd) and padded to n2
        expected_in_a = teacher_measure_a(fp)
        assert expected_in_a.shape == (kd,)
        assert np.array_equal(meas_in_a_bits[d, :kd], expected_in_a)
        _assert_padded_zeros(meas_in_a_bits[d], kd, f"meas_in_a_bits[{d}]")

        # meas_out_a stored as-is (binary), padded to n2
        out_a = meas_out_a_bits[d, :kd]
        _assert_padded_zeros(meas_out_a_bits[d], kd, f"meas_out_a_bits[{d}]")

        # new field is teacher_combine_a(fp, out_a) (length kd) and padded
        expected_next_field = teacher_combine_a(fp, out_a)
        assert expected_next_field.shape == (kd,)
        assert np.array_equal(field_bits[d + 1, :kd], expected_next_field)
        _assert_padded_zeros(field_bits[d + 1], kd, f"field_bits[{d+1}]")

    # comms_bits[0] must equal the final reduced field bit (field_bits[depth][0])
    assert comms_bits[0].shape == (1,)
    assert comms_bits[0, 0] == field_bits[depth, 0]

    # --- B-side: meas_out_b PR-rule and combine_b updates gun and comms ---
    for d in range(depth):
        Ld = s.L[d]
        kd = s.k[d]

        gp = gun_bits[d, :Ld]
        _assert_padded_zeros(gun_bits[d], Ld, f"gun_bits[{d}]")
        _one_hot_index(gun_bits[d], Ld)

        expected_in_b = teacher_measure_b(gp)
        assert expected_in_b.shape == (kd,)
        assert np.array_equal(meas_in_b_bits[d, :kd], expected_in_b)
        _assert_padded_zeros(meas_in_b_bits[d], kd, f"meas_in_b_bits[{d}]")

        # PR rule: out_b = out_a XOR (in_a AND in_b)
        out_a = meas_out_a_bits[d, :kd]
        in_a = meas_in_a_bits[d, :kd]
        in_b = meas_in_b_bits[d, :kd]
        expected_out_b = np.logical_xor(out_a > 0.5, (in_a > 0.5) & (in_b > 0.5)).astype(np.float32)

        out_b = meas_out_b_bits[d, :kd]
        assert np.array_equal(out_b, expected_out_b)
        _assert_padded_zeros(meas_out_b_bits[d], kd, f"meas_out_b_bits[{d}]")

        # Combine B: next_gun and next_comm
        comm_old = comms_bits[d]
        assert comm_old.shape == (1,)

        expected_next_gun, expected_next_comm = teacher_combine_b(gp, out_b, comm_old)
        assert expected_next_gun.shape == (kd,)
        assert expected_next_comm.shape == (1,)

        assert np.array_equal(gun_bits[d + 1, :kd], expected_next_gun)
        _assert_padded_zeros(gun_bits[d + 1], kd, f"gun_bits[{d+1}]")

        # new gun remains one-hot in prefix length kd
        _one_hot_index(gun_bits[d + 1], kd)

        assert np.array_equal(comms_bits[d + 1], expected_next_comm)
