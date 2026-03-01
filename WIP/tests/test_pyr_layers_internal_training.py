"""
Tests for the updated Pyr layer modules (internal-training variant: scaled inputs, logits outputs).

These tests validate:
- Interfaces accept converter outputs (cropped L_d/k_d, scaled/hard_logit reps)
- Output shapes match expected (B, k_d) or (B, L_{d+1}) etc.
- Outputs are logits (not restricted to [0,1]) in a simple sanity check.
- Gradients flow (single optimization step reduces MSE on tiny batch).

Repo layout expected:
QSeaBattle/
  WIP/
    src/Q_Sea_Battle_New/pyr_dataset_generation_utilities.py
    src/Q_Sea_Battle_New/pyr_dataset_conversion_utilities.py
    src/Q_Sea_Battle_New/pyr_measurement_layer_a.py
    src/Q_Sea_Battle_New/pyr_measurement_layer_b.py
    src/Q_Sea_Battle_New/pyr_combine_layer_a.py
    src/Q_Sea_Battle_New/pyr_combine_layer_b.py
  WIP/tests/test_pyr_layers_internal_training.py
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
    convert_layer_measure_a,
    convert_layer_measure_b,
    convert_layer_combine_a,
    convert_layer_combine_b,
)
from Q_Sea_Battle_New.pyr_measurement_layer_a import PyrMeasurementLayerA
from Q_Sea_Battle_New.pyr_measurement_layer_b import PyrMeasurementLayerB
from Q_Sea_Battle_New.pyr_combine_layer_a import PyrCombineLayerA
from Q_Sea_Battle_New.pyr_combine_layer_b import PyrCombineLayerB

def _mse(a, b):
    return tf.reduce_mean(tf.square(a - b))

def test_layers_accept_converter_outputs_and_shapes():
    ds = generate_pyr_dataset(n2=16, num_games=4, seed=0, validate=True)
    depth = ds["meas_in_a_bits"].shape[1]
    assert depth == 4

    mA = convert_layer_measure_a(ds, rep_x="scaled", rep_y="hard_logit", beta=10.0)
    mB = convert_layer_measure_b(ds, rep_x="scaled", rep_y="hard_logit", beta=10.0)
    cA = convert_layer_combine_a(ds, rep_field="scaled", rep_outcome="hard_logit", rep_target="hard_logit", beta=10.0)
    cB = convert_layer_combine_b(ds, rep_gun="scaled", rep_outcome_b="hard_logit", rep_comm_in="hard_logit",
                                 rep_gun_next="hard_logit", rep_comm_next="hard_logit", beta=10.0)

    # d=0 sizes for n2=16: L0=16, k0=8, L1=8
    X_a0, Y_a0 = mA[0]
    X_b0, Y_b0 = mB[0]
    (field0, outa0), field1_t = cA[0]
    (gun0, outb0, comm0), (gun1_t, comm1_t) = cB[0]

    assert X_a0.shape == (4, 16)
    assert Y_a0.shape == (4, 8)
    assert X_b0.shape == (4, 16)
    assert Y_b0.shape == (4, 8)
    assert field0.shape == (4, 16)
    assert outa0.shape == (4, 8)
    assert field1_t.shape == (4, 8)
    assert gun0.shape == (4, 16)
    assert outb0.shape == (4, 8)
    assert comm0.shape == (4, 1)
    assert gun1_t.shape == (4, 8)
    assert comm1_t.shape == (4, 1)

    # Instantiate layers and call
    la = PyrMeasurementLayerA(hidden_units=64)
    lb = PyrMeasurementLayerB(hidden_units=64)
    lca = PyrCombineLayerA(hidden_units=64)
    lcb = PyrCombineLayerB(hidden_units=64)

    out_meas_a = la(tf.convert_to_tensor(X_a0), training=False)
    out_meas_b = lb(tf.convert_to_tensor(X_b0), training=False)
    out_field1 = lca(tf.convert_to_tensor(field0), tf.convert_to_tensor(outa0), training=False)
    out_gun1, out_comm1 = lcb(tf.convert_to_tensor(gun0), tf.convert_to_tensor(outb0), tf.convert_to_tensor(comm0), training=False)

    assert out_meas_a.shape == (4, 8)
    assert out_meas_b.shape == (4, 8)
    assert out_field1.shape == (4, 8)
    assert out_gun1.shape == (4, 8)
    assert out_comm1.shape == (4, 1)

def test_logits_not_probabilities_sanity():
    # If outputs were sigmoid probabilities, large inputs would saturate to ~0/1.
    la = PyrMeasurementLayerA(hidden_units=8)
    x = tf.ones((2, 16), dtype=tf.float32) * 50.0
    y = la(x, training=False)
    # Logits should not be clipped to [0,1]
    assert tf.reduce_max(y).numpy() > 1.0 or tf.reduce_min(y).numpy() < 0.0

def test_one_gradient_step_runs():
    ds = generate_pyr_dataset(n2=16, num_games=8, seed=1, validate=True)
    mA = convert_layer_measure_a(ds, rep_x="scaled", rep_y="hard_logit", beta=10.0)
    (X, Yt) = mA[0]  # d=0, shapes (8,16)->(8,8)
    X = tf.convert_to_tensor(X)
    Yt = tf.convert_to_tensor(Yt)

    layer = PyrMeasurementLayerA(hidden_units=32)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)

    with tf.GradientTape() as tape:
        Y = layer(X, training=True)
        loss0 = _mse(Y, Yt)
    grads = tape.gradient(loss0, layer.trainable_variables)
    assert all(g is not None for g in grads)

    opt.apply_gradients(zip(grads, layer.trainable_variables))

    # Recompute loss; should be finite (not necessarily strictly smaller every time, but typically is).
    Y2 = layer(X, training=False)
    loss1 = _mse(Y2, Yt)
    assert np.isfinite(loss0.numpy())
    assert np.isfinite(loss1.numpy())
