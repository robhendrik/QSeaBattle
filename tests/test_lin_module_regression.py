import importlib
from types import SimpleNamespace

import pytest


LIN_MODULES = [
    "Q_Sea_Battle.lin_measurement_layer_a",
    "Q_Sea_Battle.lin_measurement_layer_b",
    "Q_Sea_Battle.lin_combine_layer_a",
    "Q_Sea_Battle.lin_combine_layer_b",
    "Q_Sea_Battle.lin_internal_model_a",
    "Q_Sea_Battle.lin_internal_model_b",
    "Q_Sea_Battle.lin_trainable_assisted_model_a",
    "Q_Sea_Battle.lin_trainable_assisted_model_b",
    "Q_Sea_Battle.lin_dataset_generation_utilities",
    "Q_Sea_Battle.lin_dataset_conversion_utilities",
]

TF_MODULES = {
    "Q_Sea_Battle.lin_measurement_layer_a",
    "Q_Sea_Battle.lin_measurement_layer_b",
    "Q_Sea_Battle.lin_combine_layer_a",
    "Q_Sea_Battle.lin_combine_layer_b",
    "Q_Sea_Battle.lin_internal_model_a",
    "Q_Sea_Battle.lin_internal_model_b",
    "Q_Sea_Battle.lin_trainable_assisted_model_a",
    "Q_Sea_Battle.lin_trainable_assisted_model_b",
}


@pytest.mark.parametrize("module_name", LIN_MODULES)
def test_each_lin_module_imports(module_name):
    if module_name in TF_MODULES:
        pytest.importorskip("tensorflow")
    importlib.import_module(module_name)


@pytest.mark.parametrize(
    "export_name",
    [
        "LinMeasurementLayerA",
        "LinMeasurementLayerB",
        "LinCombineLayerA",
        "LinCombineLayerB",
        "LinInternalModelA",
        "LinInternalModelB",
        "LinTrainableAssistedModelA",
        "LinTrainableAssistedModelB",
        "generate_lin_dataset",
        "convert_lin_layer_measure_a",
        "convert_lin_layer_combine_a",
        "convert_lin_layer_measure_b",
        "convert_lin_layer_combine_b",
        "convert_lin_internal_model_a",
        "convert_lin_internal_model_b",
        "convert_lin_full_system",
    ],
)
def test_lin_exports_available_from_package_root(export_name):
    if export_name in {
        "LinMeasurementLayerA",
        "LinMeasurementLayerB",
        "LinCombineLayerA",
        "LinCombineLayerB",
        "LinInternalModelA",
        "LinInternalModelB",
        "LinTrainableAssistedModelA",
        "LinTrainableAssistedModelB",
    }:
        pytest.importorskip("tensorflow")

    import Q_Sea_Battle as qsb

    assert export_name in qsb.__all__
    assert getattr(qsb, export_name) is not None


def test_lin_internal_models_smoke_forward_with_m_gt_1():
    pytest.importorskip("tensorflow")
    import tensorflow as tf

    from Q_Sea_Battle.lin_internal_model_a import LinInternalModelA
    from Q_Sea_Battle.lin_internal_model_b import LinInternalModelB

    layout = SimpleNamespace(field_size=2, comms_size=2)

    model_a = LinInternalModelA(layout, sr_mode="replay", seed=0)
    field_logits = tf.zeros((2, 4), dtype=tf.float32)
    replay = [tf.zeros((2, 4), dtype=tf.float32)]
    comm_logits, meas_a, out_a = model_a.compute_with_internal(
        field_logits,
        replay_out_a_logits_list=replay,
        training=False,
    )

    assert tuple(comm_logits.shape) == (2, 2)
    assert len(meas_a) == 1
    assert len(out_a) == 1
    assert tuple(meas_a[0].shape) == (2, 4)
    assert tuple(out_a[0].shape) == (2, 4)

    model_b = LinInternalModelB(layout, sr_mode="replay", seed=0)
    gun_logits = tf.zeros((2, 4), dtype=tf.float32)
    shoot_logits, meas_b, out_b, comm_trace, gun_trace = model_b.compute_with_internal(
        gun_logits,
        comm_logits,
        meas_a,
        out_a,
        training=False,
    )

    assert tuple(shoot_logits.shape) == (2, 1)
    assert len(meas_b) == 1
    assert len(out_b) == 1
    assert len(comm_trace) == 2
    assert len(gun_trace) == 2
    assert tuple(comm_trace[0].shape) == (2, 2)
    assert tuple(comm_trace[1].shape) == (2, 1)


def test_lin_dataset_utilities_smoke_conversion_shapes():
    from Q_Sea_Battle.lin_dataset_generation_utilities import generate_lin_dataset
    from Q_Sea_Battle.lin_dataset_conversion_utilities import (
        convert_internal_model_a,
        convert_internal_model_b,
        convert_layer_measure_a,
    )

    ds = generate_lin_dataset(n2=4, m=2, num_games=3, seed=11, validate=True)

    by_level = convert_layer_measure_a(ds, rep_x="scaled", rep_y="hard_logit", beta=10.0)
    assert set(by_level.keys()) == {0}

    field0, comm0, meas_list, out_list = convert_internal_model_a(ds)
    assert field0.shape == (3, 4)
    assert comm0.shape == (3, 2)
    assert len(meas_list) == 1
    assert len(out_list) == 1

    gun0, comm_in, prev_meas, prev_out, meas_b, out_b, shoot = convert_internal_model_b(ds)
    assert gun0.shape == (3, 4)
    assert comm_in.shape == (3, 2)
    assert len(prev_meas) == 1 and prev_meas[0].shape == (3, 4)
    assert len(prev_out) == 1 and prev_out[0].shape == (3, 4)
    assert len(meas_b) == 1 and meas_b[0].shape == (3, 4)
    assert len(out_b) == 1 and out_b[0].shape == (3, 4)
    assert shoot.shape == (3, 1)
