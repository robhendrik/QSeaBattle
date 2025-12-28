import sys
sys.path.append("./src")

import numpy as np
import tensorflow as tf
import pytest

from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.lin_measurement_layer_a import LinMeasurementLayerA
from Q_Sea_Battle.lin_measurement_layer_b import LinMeasurementLayerB
from Q_Sea_Battle.lin_combine_layer_a import LinCombineLayerA
from Q_Sea_Battle.lin_combine_layer_b import LinCombineLayerB

from Q_Sea_Battle.lin_trainable_assisted_imitation_utils import (
    generate_measurement_dataset_a,
    generate_measurement_dataset_b,
    generate_combine_dataset_a,
    generate_combine_dataset_b,
    to_tf_dataset,
)


def test_reproducibility_measurement_a():
    layout = GameLayout(field_size=4, comms_size=1)
    d1 = generate_measurement_dataset_a(layout, 32, seed=123)
    d2 = generate_measurement_dataset_a(layout, 32, seed=123)
    for k in d1:
        np.testing.assert_allclose(d1[k], d2[k])


@pytest.mark.parametrize("layer_kind", ["meas_a", "meas_b", "comb_a", "comb_b"])
def test_tiny_training_succeeds(layer_kind):
    layout = GameLayout(field_size=4, comms_size=1)
    n2 = layout.field_size ** 2
    m = layout.comms_size

    if layer_kind == "meas_a":
        data = generate_measurement_dataset_a(layout, 64, seed=10)
        ds = to_tf_dataset(data, x_keys=["field"], y_key="meas_target",
                           batch_size=16, shuffle=True, seed=0)
        inp = tf.keras.Input(shape=(n2,), dtype=tf.float32)
        out = LinMeasurementLayerA(n2=n2)(inp)
        model = tf.keras.Model(inp, out)
        model.compile(optimizer="adam", loss="binary_crossentropy")

    elif layer_kind == "meas_b":
        data = generate_measurement_dataset_b(layout, 64, seed=11)
        ds = to_tf_dataset(data, x_keys=["gun"], y_key="meas_target",
                           batch_size=16, shuffle=True, seed=0)
        inp = tf.keras.Input(shape=(n2,), dtype=tf.float32)
        out = LinMeasurementLayerB(n2=n2)(inp)
        model = tf.keras.Model(inp, out)
        model.compile(optimizer="adam", loss="binary_crossentropy")

    elif layer_kind == "comb_a":
        data = generate_combine_dataset_a(layout, 64, seed=12)
        ds = to_tf_dataset(data, x_keys=["outcomes_a"], y_key="comm_target",
                           batch_size=16, shuffle=True, seed=0)
        inp = tf.keras.Input(shape=(n2,), dtype=tf.float32)
        out = LinCombineLayerA(comms_size=m)(inp)
        model = tf.keras.Model(inp, out)
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        )

    else:
        data = generate_combine_dataset_b(layout, 64, seed=13)
        ds = to_tf_dataset(data,
                           x_keys=["outcomes_b", "comm"],
                           y_key="shoot_target",
                           batch_size=16, shuffle=True, seed=0)
        inp_outcomes = tf.keras.Input(shape=(n2,), dtype=tf.float32)
        inp_comm = tf.keras.Input(shape=(m,), dtype=tf.float32)
        layer = LinCombineLayerB(comms_size=m)
        out = layer(inp_outcomes, inp_comm)
        model = tf.keras.Model([inp_outcomes, inp_comm], out)
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        )

    model.fit(ds, epochs=1, verbose=0)
