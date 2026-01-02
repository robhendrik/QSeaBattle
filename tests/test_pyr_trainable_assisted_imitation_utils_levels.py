import numpy as np

from Q_Sea_Battle.pyr_trainable_assisted_imitation_utils import (
    generate_measurement_dataset_a_level,
    generate_combine_dataset_a_level,
    generate_measurement_dataset_b_level,
    generate_combine_dataset_b_level,
)


def _is_one_hot(x: np.ndarray) -> np.ndarray:
    # Returns boolean array (N,) indicating whether each row is one-hot (sum==1 and values in {0,1})
    return (np.isin(x, [0.0, 1.0]).all(axis=1)) & (np.isclose(x.sum(axis=1), 1.0))


def test_meas_a_shapes_and_values() -> None:
    L = 16
    N = 128
    ds = generate_measurement_dataset_a_level(L=L, num_samples=N, seed=1)
    field = ds["field"]
    meas = ds["meas_target"]
    assert field.shape == (N, L)
    assert meas.shape == (N, L // 2)
    assert np.isin(meas, [0.0, 1.0]).all()


def test_combine_a_shapes_and_logic() -> None:
    import numpy as np

    L = 16
    N = 128
    ds = generate_combine_dataset_a_level(L=L, num_samples=N, seed=2)

    field = ds["field"]
    sr = ds["sr_outcome"]
    next_field = ds["next_field_target"]

    # --- shape checks ---
    assert field.shape == (N, L)
    assert sr.shape == (N, L // 2)
    assert next_field.shape == (N, L // 2)

    # --- Combine A rule: next_field = even XOR sr ---
    even = field[:, ::2]
    expected_next_field = (even != sr).astype(np.float32)

    np.testing.assert_array_equal(next_field, expected_next_field)


def test_meas_b_shapes_and_values() -> None:
    L = 16
    N = 128
    ds = generate_measurement_dataset_b_level(L=L, num_samples=N, seed=3)
    gun = ds["gun"]
    meas = ds["meas_target"]
    assert gun.shape == (N, L)
    assert meas.shape == (N, L // 2)
    assert np.isin(meas, [0.0, 1.0]).all()


def test_combine_b_gun_is_one_hot_and_downsamples() -> None:
    """
    Combine-B dataset:
    - gun input must be one-hot
    - next_gun_target must remain one-hot
    - rule: next_gun = gun[::2] XOR gun[1::2]   (pair-index downsample)
    """
    L = 16
    N = 256
    SEED = 123

    ds = generate_combine_dataset_b_level(L=L, num_samples=N, seed=SEED)

    gun = ds["gun"]                  # (N, L)
    next_gun = ds["next_gun_target"] # (N, L//2)

    assert gun.shape == (N, L)
    assert next_gun.shape == (N, L // 2)

    assert _is_one_hot(gun).all(), "Input gun is not one-hot"
    assert _is_one_hot(next_gun).all(), "next_gun_target is not one-hot"

    expected_next_gun = np.logical_xor(gun[:, ::2] > 0.5, gun[:, 1::2] > 0.5).astype(np.float32)
    np.testing.assert_array_equal(next_gun, expected_next_gun)
