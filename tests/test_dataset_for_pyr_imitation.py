import numpy as np
import pytest

from Q_Sea_Battle.pyr_trainable_assisted_imitation_utils import (
    generate_measurement_dataset_a_level,
    generate_combine_dataset_a_level,
    generate_measurement_dataset_b_level,
    generate_combine_dataset_b_level,
)

# ---------- Independent reference rules ----------

def ref_measure_a(field):
    even = field[:, 0::2]
    odd  = field[:, 1::2]
    return (even != odd).astype(np.float32)

def ref_combine_a(field, sr):
    even = field[:, 0::2]
    return (even != sr).astype(np.float32)

def ref_measure_b(gun):
    even = gun[:, 0::2]
    odd  = gun[:, 1::2]
    return ((even == 0) & (odd == 1)).astype(np.float32)

def ref_next_gun(gun):
    even = gun[:, 0::2]
    odd  = gun[:, 1::2]
    return (even != odd).astype(np.float32)

def ref_next_comm(comm, next_gun, sr):
    # gun is one-hot → exactly one index is active
    idx = np.argmax(next_gun, axis=1)
    sr_bit = sr[np.arange(len(sr)), idx]
    return ((comm[:, 0] != sr_bit).astype(np.float32))[:, None]

# ---------- Tests ----------

@pytest.mark.parametrize("L", [8, 16])
def test_measurement_a_dataset(L):
    ds = generate_measurement_dataset_a_level(L=L, num_samples=128, seed=1)
    ref = ref_measure_a(ds["field"])
    np.testing.assert_array_equal(ds["meas_target"], ref)

@pytest.mark.parametrize("L", [8, 16])
def test_combine_a_dataset(L):
    ds = generate_combine_dataset_a_level(L=L, num_samples=128, seed=2)
    ref = ref_combine_a(ds["field"], ds["sr_outcome"])
    np.testing.assert_array_equal(ds["next_field_target"], ref)

@pytest.mark.parametrize("L", [8, 16])
def test_measurement_b_dataset(L):
    ds = generate_measurement_dataset_b_level(L=L, num_samples=128, seed=3)

    gun = ds["gun"]
    meas = ds["meas_target"]

    # gun must be one-hot
    np.testing.assert_array_equal(gun.sum(axis=1), np.ones(len(gun)))

    ref = ref_measure_b(gun)
    np.testing.assert_array_equal(meas, ref)

@pytest.mark.parametrize("L", [8, 16])
def test_combine_b_new_gun_and_comm(L):
    ds = generate_combine_dataset_b_level(L=L, num_samples=128, seed=4)

    gun = ds["gun"]
    sr  = ds["sr_outcome"]
    comm = ds["comm"]

    next_gun = ds["next_gun_target"]
    next_comm = ds["next_comm_target"]

    # gun must be one-hot
    np.testing.assert_array_equal(gun.sum(axis=1), np.ones(len(gun)))

    # next_gun must be one-hot
    np.testing.assert_array_equal(next_gun.sum(axis=1), np.ones(len(next_gun)))

    # gun downsampling rule
    ref_ng = ref_next_gun(gun)
    np.testing.assert_array_equal(next_gun, ref_ng)

    # comm update rule
    ref_nc = ref_next_comm(comm, next_gun, sr)
    np.testing.assert_array_equal(next_comm, ref_nc)
