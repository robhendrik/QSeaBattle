
from __future__ import annotations

import os
import sys
from pathlib import Path

def change_to_repo_root(marker: str = "src") -> None:
    """Change CWD to the repository root (parent of `src`)."""
    here = Path.cwd()
    for parent in [here] + list(here.parents):
        if (parent / marker).is_dir():
            os.chdir(parent)
            break

change_to_repo_root("src")
sys.path.append("./src")


import numpy as np
import tensorflow as tf

from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.game_env import GameEnv
from Q_Sea_Battle.tournament import Tournament

from Q_Sea_Battle.lin_trainable_models import LinTrainableAssistedModelA
from Q_Sea_Battle.lin_trainable_models import LinTrainableAssistedModelB
from Q_Sea_Battle.trainable_assisted_players import TrainableAssistedPlayers

from Q_Sea_Battle.lin_trainable_assisted_imitation_utilities import (
    generate_measurement_dataset_a,
    generate_measurement_dataset_b,
    generate_combine_dataset_a,
    generate_combine_dataset_b,
    to_tf_dataset,
    transfer_assisted_model_a_layer_weights,
    transfer_assisted_model_b_layer_weights,
    train_layer
)




FIELD_SIZE = 4
COMMS_SIZE = 1

# shared resource (SR) correlation parameter used by your task
P_HIGH = 1.0

# Dataset / training sizes
DATASET_SIZE = 500
BATCH_SIZE = 16
EPOCHS_MEAS = 1 # we can use smaller number of epochs for measurement training, since it is an easier task
EPOCHS_COMB = 5

# DIAL/DRU training settings
SR_MODE_BOOTSTRAP_EVAL = "sample"
SR_MODE_DIAL_TRAIN = "expected"
SR_MODE_DIAL_EVAL = "sample"

SEED = 123
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Folders
data_dir = Path("tests/data_for_tests")
models_dir = Path("tests/data_for_tests")
data_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

n2 = FIELD_SIZE * FIELD_SIZE
print("n2:", n2, "m:", COMMS_SIZE)


def test_tutorial_lin_trainable_assisted_imitation():
    # Build layout for data generation (enemy_probability/channel_noise not used by these generators)
    layout = GameLayout(field_size=FIELD_SIZE, comms_size=COMMS_SIZE)

    # --- Generate datasets ---
    ds_meas_a = generate_measurement_dataset_a(layout, num_samples=DATASET_SIZE, seed=SEED)
    ds_comb_a = generate_combine_dataset_a(layout, num_samples=DATASET_SIZE, seed=SEED + 1)
    ds_meas_b = generate_measurement_dataset_b(layout, num_samples=DATASET_SIZE, seed=SEED + 2)
    ds_comb_b = generate_combine_dataset_b(layout, num_samples=DATASET_SIZE, seed=SEED + 3)

    tfds_meas_a = to_tf_dataset(ds_meas_a, x_keys=["field"], y_key="meas_target", batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
    tfds_comb_a = to_tf_dataset(ds_comb_a, x_keys=["outcomes_a"], y_key="comm_target", batch_size=BATCH_SIZE, shuffle=True, seed=SEED+1)
    tfds_meas_b = to_tf_dataset(ds_meas_b, x_keys=["gun"], y_key="meas_target", batch_size=BATCH_SIZE, shuffle=True, seed=SEED+2)
    tfds_comb_b = to_tf_dataset(ds_comb_b, x_keys=["outcomes_b","comm"], y_key="shoot_target", batch_size=BATCH_SIZE, shuffle=True, seed=SEED+3)



    # --- Train layers ---
    from Q_Sea_Battle.lin_teacher_layers import LinMeasurementLayerA
    from Q_Sea_Battle.lin_teacher_layers import LinMeasurementLayerB
    from Q_Sea_Battle.lin_teacher_layers import LinCombineLayerA
    from Q_Sea_Battle.lin_teacher_layers import LinCombineLayerB

    # --- Build layers ---
    n2 = FIELD_SIZE * FIELD_SIZE
    model_a = LinTrainableAssistedModelA(
        field_size=FIELD_SIZE,
        comms_size=COMMS_SIZE,
        sr_mode="sample",   # evaluation mode
        seed=SEED,
        p_high=P_HIGH,
    )
    meas_layer_a = model_a.measure_layer
    comb_layer_a = model_a.combine_layer

    model_b = LinTrainableAssistedModelB(
        field_size=FIELD_SIZE,
        comms_size=COMMS_SIZE,
        sr_mode="sample",   # evaluation mode
        seed=SEED,
        p_high=P_HIGH,
    )
    meas_layer_b = model_b.measure_layer
    comb_layer_b = model_b.combine_layer

    _ = train_layer(meas_layer_a, tfds_meas_a, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), epochs=EPOCHS_MEAS,
                    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)])
    _ = train_layer(comb_layer_a, tfds_comb_a, loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), epochs=EPOCHS_COMB,
                    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0)])

    _ = train_layer(meas_layer_b, tfds_meas_b, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), epochs=EPOCHS_MEAS,
                    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)])
    _ = train_layer(comb_layer_b, tfds_comb_b, loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), epochs=EPOCHS_COMB,
                    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0)])



    # %%
    # --- Install into full models ---
    model_a = LinTrainableAssistedModelA(field_size=FIELD_SIZE, comms_size=COMMS_SIZE, sr_mode=SR_MODE_BOOTSTRAP_EVAL, seed=SEED, p_high=P_HIGH)
    model_b = LinTrainableAssistedModelB(field_size=FIELD_SIZE, comms_size=COMMS_SIZE, sr_mode=SR_MODE_BOOTSTRAP_EVAL, seed=SEED, p_high=P_HIGH)

    # Build models (required before weight transfer)
    _ = model_a(tf.zeros((1, n2), tf.float32))
    _dummy_gun = tf.zeros((1, n2), tf.float32)
    _dummy_comm = tf.zeros((1, COMMS_SIZE), tf.float32)
    _dummy_prev_meas_list = [tf.zeros((1, n2), tf.float32)]
    _dummy_prev_out_list  = [tf.zeros((1, n2), tf.float32)]
    _ = model_b([_dummy_gun, _dummy_comm, _dummy_prev_meas_list, _dummy_prev_out_list])

    transfer_assisted_model_a_layer_weights(meas_layer_a, comb_layer_a, model_a)
    transfer_assisted_model_b_layer_weights(meas_layer_b, comb_layer_b, model_b)


    # Force SR sample explicitly for evaluation
    model_a.sr_layer.mode = "sample"

    model_b.sr_layer.mode = "sample"

    print("model_a sr_mode:", model_a.sr_layer.mode)
    print("model_b sr_mode:", model_b.sr_layer.mode)
    players = TrainableAssistedPlayers(layout, model_a=model_a, model_b=model_b)

    layout_eval = GameLayout(
        field_size=FIELD_SIZE,
        comms_size=COMMS_SIZE,
        enemy_probability=0.5,
        channel_noise=0.0,
        number_of_games_in_tournament=2_0,
    )
    env = GameEnv(layout_eval)
    t = Tournament(env, players, layout_eval)
    log = t.tournament()
    mean_reward, std_err = log.outcome()
    

    model_a_path = models_dir / f"lin_model_a_bootstrap_f{FIELD_SIZE}_m{COMMS_SIZE}_p{P_HIGH:.2f}.weights.h5"
    model_b_path = models_dir / f"lin_model_b_bootstrap_f{FIELD_SIZE}_m{COMMS_SIZE}_p{P_HIGH:.2f}.weights.h5"

    model_a.save_weights(model_a_path)
    model_b.save_weights(model_b_path)

    assert True

    


