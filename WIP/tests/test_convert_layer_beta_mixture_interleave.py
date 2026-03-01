# (B) Create: tests/test_convert_layer_beta_mixture_interleave.py

import numpy as np

from Q_Sea_Battle_New.pyr_dataset_conversion_utilities import (
    convert_layer_measure_a,
    convert_layer_combine_a,
    convert_layer_measure_b,
    convert_layer_combine_b,
)


def _make_ds(N: int = 6, depth: int = 2, n2: int = 4):
    # Shapes:
    # field_bits: (N, depth+1, n2)
    # gun_bits:   (N, depth+1, n2)
    # comms_bits: (N, depth+1, 1)
    # meas_*:     (N, depth,   n2)
    # shoot:      (N, 1)
    field_bits = np.zeros((N, depth + 1, n2), dtype=np.float32)
    gun_bits = np.zeros((N, depth + 1, n2), dtype=np.float32)
    comms_bits = np.zeros((N, depth + 1, 1), dtype=np.float32)

    meas_in_a_bits = np.zeros((N, depth, n2), dtype=np.float32)
    meas_out_a_bits = np.zeros((N, depth, n2), dtype=np.float32)
    meas_in_b_bits = np.zeros((N, depth, n2), dtype=np.float32)
    meas_out_b_bits = np.zeros((N, depth, n2), dtype=np.float32)

    shoot = np.zeros((N, 1), dtype=np.float32)

    # Fill with simple deterministic patterns
    for i in range(N):
        for d in range(depth + 1):
            # alternate bits
            field_bits[i, d, :] = (np.arange(n2) + i + d) % 2
            gun_bits[i, d, :] = (np.arange(n2) + 2 * i + d) % 2
            comms_bits[i, d, 0] = float((i + d) % 2)

        for d in range(depth):
            meas_in_a_bits[i, d, :] = (np.arange(n2) + i) % 2
            meas_out_a_bits[i, d, :] = (np.arange(n2) + i + 1) % 2
            meas_in_b_bits[i, d, :] = (np.arange(n2) + i + 1) % 2
            meas_out_b_bits[i, d, :] = (np.arange(n2) + i) % 2

        shoot[i, 0] = float(i % 2)

    return {
        "field_bits": field_bits,
        "gun_bits": gun_bits,
        "comms_bits": comms_bits,
        "meas_in_a_bits": meas_in_a_bits,
        "meas_out_a_bits": meas_out_a_bits,
        "meas_in_b_bits": meas_in_b_bits,
        "meas_out_b_bits": meas_out_b_bits,
        "shoot": shoot,
    }


def _dict_array_equal(a, b) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a.keys():
        va = a[k]
        vb = b[k]
        if isinstance(va, tuple):
            # nested tuples in combine converters
            if not _tuple_array_equal(va, vb):
                return False
        else:
            if not np.array_equal(va, vb):
                return False
    return True


def _tuple_array_equal(ta, tb) -> bool:
    if len(ta) != len(tb):
        return False
    for xa, xb in zip(ta, tb):
        if isinstance(xa, tuple):
            if not _tuple_array_equal(xa, xb):
                return False
        else:
            if not np.array_equal(xa, xb):
                return False
    return True


def test_backward_compat_beta_scalar_vs_singleton_list():
    ds = _make_ds(N=6, depth=2, n2=4)

    a1 = convert_layer_measure_a(ds, beta=10.0)
    a2 = convert_layer_measure_a(ds, beta=[10.0])
    assert _dict_array_equal(a1, a2)

    ca1 = convert_layer_combine_a(ds, beta=10.0)
    ca2 = convert_layer_combine_a(ds, beta=[10.0])
    assert _dict_array_equal(ca1, ca2)

    b1 = convert_layer_measure_b(ds, beta=10.0)
    b2 = convert_layer_measure_b(ds, beta=[10.0])
    assert _dict_array_equal(b1, b2)

    cb1 = convert_layer_combine_b(ds, beta=10.0)
    cb2 = convert_layer_combine_b(ds, beta=[10.0])
    assert _dict_array_equal(cb1, cb2)


def test_mixture_changes_hard_logit_outputs_and_scaled_is_invariant():
    ds = _make_ds(N=6, depth=2, n2=4)

    # (1) Mixture affects hard_logit
    base = convert_layer_measure_a(ds, rep_x="scaled", rep_y="hard_logit", beta=10.0)
    mix = convert_layer_measure_a(ds, rep_x="scaled", rep_y="hard_logit", beta=[0.0, 10.0])

    # Check at least one level differs
    differs = False
    for d in base.keys():
        _, Y_base = base[d]
        _, Y_mix = mix[d]
        if np.any(Y_base != Y_mix):
            differs = True
            break
    assert differs, "Expected mixed betas to change hard_logit outputs."

    # Replace the scaled-only block in:
    # test_mixture_changes_hard_logit_outputs_and_scaled_is_invariant

    def _sort_rows_2d(a: np.ndarray) -> np.ndarray:
        """
        Deterministically sort rows of a 2D array.
        Uses lexsort on all columns (stable enough for our purpose).
        """
        if a.ndim != 2:
            raise ValueError(f"expected 2D, got shape={a.shape}")
        if a.shape[0] == 0:
            return a
        # lexsort uses last key as primary, so reverse columns
        keys = [a[:, c] for c in range(a.shape[1] - 1, -1, -1)]
        idx = np.lexsort(keys)
        return a[idx]


    def _dict_xy_equal_ignoring_row_order(d1, d2) -> bool:
        if d1.keys() != d2.keys():
            return False
        for k in d1.keys():
            X1, Y1 = d1[k]
            X2, Y2 = d2[k]
            if not np.array_equal(_sort_rows_2d(X1), _sort_rows_2d(X2)):
                return False
            if not np.array_equal(_sort_rows_2d(Y1), _sort_rows_2d(Y2)):
                return False
        return True


    def test_mixture_changes_hard_logit_outputs_and_scaled_is_invariant():
        ds = _make_ds(N=6, depth=2, n2=4)

        # (1) Mixture affects hard_logit (note: order differs, so compare values ignoring row order)
        base = convert_layer_measure_a(ds, rep_x="scaled", rep_y="hard_logit", beta=10.0)
        mix = convert_layer_measure_a(ds, rep_x="scaled", rep_y="hard_logit", beta=[0.0, 10.0])

        differs = False
        for d in base.keys():
            _, Y_base = base[d]
            _, Y_mix = mix[d]
            if not np.array_equal(np.sort(Y_base.reshape(-1)), np.sort(Y_mix.reshape(-1))):
                differs = True
                break
        assert differs, "Expected mixed betas to change hard_logit outputs (as a multiset, ignoring order)."

        # (2) Scaled-only outputs are invariant to beta VALUE, but order changes due to interleaving.
        # Therefore compare ignoring row order.
        s1 = convert_layer_measure_a(ds, rep_x="scaled", rep_y="scaled", beta=10.0)
        s2 = convert_layer_measure_a(ds, rep_x="scaled", rep_y="scaled", beta=[0.0, 10.0])
        assert _dict_xy_equal_ignoring_row_order(s1, s2), (
            "Expected scaled-only outputs to be identical as a multiset (ignoring order) regardless of beta."
        )


def test_interleaving_order_is_round_robin_contiguous_split():
    # N=6, K=2 -> contiguous chunks [0,1,2] and [3,4,5]
    # Interleave -> [0,3,1,4,2,5]
    N, depth, n2 = 6, 2, 4
    ds = _make_ds(N=N, depth=depth, n2=n2)

    # Make field_bits[:,0,:] easy to track (marker in first element).
    # This intentionally uses non-{0,1} markers to track ordering.
    ds["field_bits"][:, 0, :] = 0.0
    for i in range(N):
        ds["field_bits"][i, 0, 0] = float(i)

    out = convert_layer_measure_a(ds, rep_x="bits", rep_y="bits", beta=[10.0, 0.1])
    X, _ = out[0]  # level d=0, X comes from field_bits[:,0,:L_0]

    expected = np.array([0, 3, 1, 4, 2, 5], dtype=np.float32)
    got = X[:, 0]
    assert np.array_equal(got, expected), f"Interleaving mismatch. got={got}, expected={expected}"
