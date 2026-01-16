

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

change_to_repo_root()


src_path = Path("src").resolve()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))



# %%

import random
import math
import numpy as np
import pandas as pd
import tensorflow as tf

import Q_Sea_Battle as qsb
from Q_Sea_Battle.dru_utilities import dru_train
from Q_Sea_Battle.reference_performance_utilities import limit_from_mutual_information

SEED = 1232
np.random.seed(SEED)
random.seed(SEED * 2)
tf.random.set_seed(SEED * 4)

tf.config.run_functions_eagerly(True)




FIELD_SIZES = [4, 32]
COMMS_SIZES = [1,8]

# RL hyperparameters (can be tuned)
NUM_EPOCHS = 16
BATCHES_PER_EPOCH = 4
BATCH_SIZE = 2048
SIGMA_TRAIN = 2.0
CLIP_RANGE = (-10.0, 10.0)
LEARNING_RATE = 1e-3

# Sigma annealing for DRU: high noise -> low noise over epochs
SIGMA_START = 2.0
SIGMA_END = 0.3

N_GAMES_TOURNAMENT = 10



from Q_Sea_Battle.tournament import Tournament
from Q_Sea_Battle.majority_players import MajorityPlayers
from Q_Sea_Battle.neural_net_players import NeuralNetPlayers


def sample_batch(layout: qsb.GameLayout, batch_size: int):
    """Sample (fields, guns, cell_values) for the given layout.

    - fields: shape (B, n²), Bernoulli(enemy_probability)
    - guns:   shape (B, n²), one-hot
    - cell_values: shape (B, 1), field value at the gun index
    """
    n2 = layout.field_size ** 2
    p = layout.enemy_probability

    fields = np.random.binomial(1, p, size=(batch_size, n2)).astype("float32")

    guns = np.zeros((batch_size, n2), dtype="float32")
    gun_indices = np.random.randint(0, n2, size=(batch_size,))
    guns[np.arange(batch_size), gun_indices] = 1.0

    cell_values = (fields * guns).sum(axis=1, keepdims=True).astype("float32")
    return fields, guns, cell_values


def evaluate_players_in_tournament(
    layout: qsb.GameLayout,
    players_factory,
    label: str = "",
) -> float:
    """Run a tournament and return mean reward.

    `players_factory` is any Players subclass instance, e.g.
    MajorityPlayers(layout) or NeuralNetPlayers(layout, ...).
    """
    game_env = qsb.GameEnv(game_layout=layout)
    tournament = Tournament(game_env=game_env, players=players_factory, game_layout=layout)
    log = tournament.tournament()
    mean_reward, std_err = log.outcome()

    if label:
        print(
            f"{label}: mean reward = {mean_reward:.4f} ± {std_err:.4f} "
        )
    else:
        print(
            f"mean reward = {mean_reward:.4f} ± {std_err:.4f} "
        )

    return mean_reward


# %%

def dial_pg_update(
    model_a: tf.keras.Model,
    model_b: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    layout: qsb.GameLayout,
    batch_size: int = 512,
    sigma: float = 2.0,
    clip_range=(-10.0, 10.0),
    entropy_coeff: float = 0.01,
    normalize_adv: bool = True,
) -> tuple[float, float]:
    """Improved DIAL-style policy-gradient update step with DRU.

    Compared to the earlier version, this adds:
    - optional advantage normalization (variance reduction),
    - an entropy bonus on the shoot policy.
    """
    fields_np, guns_np, cell_values_np = sample_batch(layout, batch_size)

    fields_tf = tf.convert_to_tensor(fields_np, dtype=tf.float32)
    fields_scaled = fields_tf - 0.5  # scale {0,1} -> [-0.5, 0.5] to match NeuralNetPlayerA
    guns_tf = tf.convert_to_tensor(guns_np, dtype=tf.float32)
    cell_values_tf = tf.convert_to_tensor(cell_values_np, dtype=tf.float32)

    n2 = layout.field_size ** 2
    denom = float(max(1, n2 - 1))
    gun_indices = tf.argmax(guns_tf, axis=1, output_type=tf.int32)
    gun_idx_norm = tf.cast(gun_indices, tf.float32) / denom
    gun_idx_norm = tf.reshape(gun_idx_norm, (-1, 1))


    eps = 1e-8

    with tf.GradientTape() as tape:
        # A produces communication logits
        comm_logits = model_a(fields_scaled, training=True)          # (B, m)

        # DRU (train mode): logits + noise -> logistic
        comm_cont = dru_train(comm_logits, sigma=sigma, clip_range=clip_range)
        comm_cont = tf.cast(comm_cont, tf.float32)               # (B, m) in (0,1)

        # B receives gun + continuous comm
        x_b = tf.concat([gun_idx_norm, comm_cont], axis=1)          # (B, 1 + m)
        shoot_logits = model_b(x_b, training=True)               # (B, 1)

        probs = tf.nn.sigmoid(shoot_logits)
        rnd = tf.random.uniform(tf.shape(probs))
        actions = tf.cast(rnd < probs, tf.float32)               # (B, 1) in {0,1}

        # Team reward: 1 if correct guess of the field bit at the gun index
        rewards = tf.cast(tf.equal(actions, cell_values_tf), tf.float32)

        # Baseline and advantage
        baseline = tf.reduce_mean(rewards)
        advantages = rewards - baseline

        if normalize_adv:
            adv_std = tf.math.reduce_std(advantages) + 1e-8
            advantages = advantages / adv_std

        advantages = tf.stop_gradient(advantages)

        # Log-prob of the sampled action under Bernoulli(probs)
        log_probs = (
            actions * tf.math.log(probs + eps)
            + (1.0 - actions) * tf.math.log(1.0 - probs + eps)
        )

        # Policy entropy for Bernoulli(probs)
        entropy = -(
            probs * tf.math.log(probs + eps)
            + (1.0 - probs) * tf.math.log(1.0 - probs + eps)
        )

        # REINFORCE loss with entropy regularization
        loss_pg = -tf.reduce_mean(log_probs * advantages)
        loss_ent = -tf.reduce_mean(entropy)   # negative so adding pushes up entropy
        loss = loss_pg + entropy_coeff * loss_ent

    params = model_a.trainable_variables + model_b.trainable_variables
    grads = tape.gradient(loss, params)
    optimizer.apply_gradients(zip(grads, params))

    mean_reward = float(tf.reduce_mean(rewards).numpy())
    loss_value = float(loss.numpy())
    return mean_reward, loss_value

# %%
def sigma_for_epoch(epoch: int, num_epochs: int) -> float:
    """Linearly interpolate sigma from SIGMA_START to SIGMA_END over epochs."""
    t = epoch / max(1, num_epochs)
    return SIGMA_START * (1.0 - t) + SIGMA_END * t

def test_tutorial_DIAL_DRU_training_neural_net_models():
    results_updated = []

    for n in FIELD_SIZES:
        for m in COMMS_SIZES:
            print(f"Field_size = {n}, comms_size = {m}")

            layout = qsb.GameLayout(
                field_size=n,
                comms_size=m,
                enemy_probability=0.5,
                channel_noise=0.0,
                number_of_games_in_tournament=N_GAMES_TOURNAMENT,
            )

            # Majority players as before
            majority_players = MajorityPlayers(layout)
            maj_mean = evaluate_players_in_tournament(
                layout, majority_players,
                label="\tMajorityPlayers"
            )

            # Fresh NeuralNetPlayers and models
            nn_players = NeuralNetPlayers(game_layout=layout, explore=True)
            player_a, player_b = nn_players.players()
            model_a = nn_players.model_a
            model_b = nn_players.model_b
            assert model_a is not None and model_b is not None

            optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

            for epoch in range(1, NUM_EPOCHS + 1):
                sigma_now = sigma_for_epoch(epoch, NUM_EPOCHS)
                epoch_rewards = []
                epoch_losses = []

                for _ in range(BATCHES_PER_EPOCH):
                    r, l = dial_pg_update(
                        model_a=model_a,
                        model_b=model_b,
                        optimizer=optimizer,
                        layout=layout,
                        batch_size=BATCH_SIZE,
                        sigma=sigma_now,
                        clip_range=CLIP_RANGE,
                        entropy_coeff=0.01,
                        normalize_adv=True,
                    )
                    epoch_rewards.append(r)
                    epoch_losses.append(l)

            # Evaluate trained neural nets (greedy play)
            nn_players_eval = NeuralNetPlayers(
                game_layout=layout,
                model_a=model_a,
                model_b=model_b,
                explore=False,
            )
            nn_mean = evaluate_players_in_tournament(
                layout, nn_players_eval,
                label="\tNeuralNetPlayers (DIAL+DRU)"
            )

            info_limit = limit_from_mutual_information(
                field_size=n,
                comms_size=m,
                channel_noise=0.0,
                accuracy_in_digits=10,
            )

            print(
                f"\tInfo-theoretic upper bound (noiseless channel): {info_limit:.4f}"
            )

            results_updated.append(
                {
                    "field_size": n,
                    "comms_size": m,
                    "maj_mean": maj_mean,
                    "nn_dial_mean": nn_mean,
                    "info_limit": info_limit,
                }
            )

            # Store trained models for this (field_size, comms_size) setting
            filename_a = f"notebooks/models/neural_net_model_a_f{n}_c{m}.keras"
            filename_b = f"notebooks/models/neural_net_model_b_f{n}_c{m}.keras"
            nn_players.store_models(filename_a, filename_b)
            print(f"Saved models to {filename_a} and {filename_b}")



    # %%

    df_results_updated = pd.DataFrame(results_updated)
    assert df_results_updated is not None
