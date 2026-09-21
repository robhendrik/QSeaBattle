import importlib
from types import SimpleNamespace

import pytest


PYR_MODULES = [
    "Q_Sea_Battle.pr_assisted_replay",
    "Q_Sea_Battle.pyr_measurement_layer_a",
    "Q_Sea_Battle.pyr_measurement_layer_b",
    "Q_Sea_Battle.pyr_combine_layer_a",
    "Q_Sea_Battle.pyr_combine_layer_b",
    "Q_Sea_Battle.pyr_internal_model_a",
    "Q_Sea_Battle.pyr_internal_model_b",
    "Q_Sea_Battle.pyr_dataset_generation_utilities",
    "Q_Sea_Battle.pyr_dataset_conversion_utilities",
]

TF_MODULES = {
    "Q_Sea_Battle.pr_assisted_replay",
    "Q_Sea_Battle.pyr_measurement_layer_a",
    "Q_Sea_Battle.pyr_measurement_layer_b",
    "Q_Sea_Battle.pyr_combine_layer_a",
    "Q_Sea_Battle.pyr_combine_layer_b",
    "Q_Sea_Battle.pyr_internal_model_a",
    "Q_Sea_Battle.pyr_internal_model_b",
}


@pytest.mark.parametrize("module_name", PYR_MODULES)
def test_each_pyr_module_imports(module_name):
    if module_name in TF_MODULES:
        pytest.importorskip("tensorflow")
    importlib.import_module(module_name)


@pytest.mark.parametrize(
    "export_name",
    [
        "PRAssistedReplay",
        "PyrMeasurementLayerA",
        "PyrMeasurementLayerB",
        "PyrCombineLayerA",
        "PyrCombineLayerB",
        "PyrInternalModelA",
        "PyrInternalModelB",
        "generate_pyr_dataset",
        "save_npz",
        "convert_layer_measure_a",
        "convert_layer_combine_a",
        "convert_layer_measure_b",
        "convert_layer_combine_b",
        "convert_internal_model_a",
        "convert_internal_model_b",
        "convert_full_system",
    ],
)
def test_pyr_exports_available_from_package_root(export_name):
    if export_name in {
        "PRAssistedReplay",
        "PyrMeasurementLayerA",
        "PyrMeasurementLayerB",
        "PyrCombineLayerA",
        "PyrCombineLayerB",
        "PyrInternalModelA",
        "PyrInternalModelB",
    }:
        pytest.importorskip("tensorflow")

    import Q_Sea_Battle as qsb

    assert export_name in qsb.__all__
    assert getattr(qsb, export_name) is not None


def test_pyr_internal_models_smoke_forward():
    pytest.importorskip("tensorflow")
    import tensorflow as tf

    from Q_Sea_Battle.pyr_internal_model_a import PyrInternalModelA
    from Q_Sea_Battle.pyr_internal_model_b import PyrInternalModelB

    layout = SimpleNamespace(field_size=2, comms_size=1)

    model_a = PyrInternalModelA(layout, sr_mode="replay", seed=0)
    x = tf.zeros((1, 4), dtype=tf.float32) - 0.5
    replay = [tf.zeros((1, 2), dtype=tf.float32), tf.zeros((1, 1), dtype=tf.float32)]
    comm_logits, meas_a, out_a = model_a.compute_with_internal(x, replay_out_a_logits_list=replay, training=False)

    assert tuple(comm_logits.shape) == (1, 1)
    assert len(meas_a) == 2
    assert len(out_a) == 2

    model_b = PyrInternalModelB(layout, sr_mode="replay", seed=0)
    gun = tf.zeros((1, 4), dtype=tf.float32)
    comm = tf.zeros((1, 1), dtype=tf.float32)
    shoot_logits, meas_b, out_b, comm_trace, gun_trace = model_b.compute_with_internal(
        gun, comm, meas_a, out_a, training=False
    )

    assert tuple(shoot_logits.shape) == (1, 1)
    assert len(meas_b) == 2
    assert len(out_b) == 2
    assert len(comm_trace) == 3
    assert len(gun_trace) == 3


def test_pyr_dataset_utilities_smoke_conversion_shapes():
    from Q_Sea_Battle.pyr_dataset_generation_utilities import generate_pyr_dataset
    from Q_Sea_Battle.pyr_dataset_conversion_utilities import (
        convert_layer_measure_a,
        convert_internal_model_a,
    )

    ds = generate_pyr_dataset(n2=4, num_games=2, seed=3, validate=True)

    by_level = convert_layer_measure_a(ds, rep_x="scaled", rep_y="hard_logit", beta=10.0)
    assert set(by_level.keys()) == {0, 1}

    field0, comm_t, meas_t, out_t = convert_internal_model_a(ds)
    assert field0.shape == (2, 4)
    assert comm_t.shape == (2, 1)
    assert len(meas_t) == 2
    assert len(out_t) == 2