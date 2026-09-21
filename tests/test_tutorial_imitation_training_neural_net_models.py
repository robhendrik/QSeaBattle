

import numpy as np
import pandas as pd
import sys
sys.path.append("./src")
import Q_Sea_Battle as qsb 
# Core QSeaBattle imports.
# Adjust these imports if your package layout is different.
from Q_Sea_Battle import GameLayout
from Q_Sea_Battle import GameEnv
from Q_Sea_Battle import Tournament
from Q_Sea_Battle import MajorityPlayers, NeuralNetPlayers

from Q_Sea_Battle import neural_net_imitation_utilities as imitation_utils

def test_tutorial_imitation_training_neural_net_models():

    # Debug-friendly settings — use only during development.
    import tensorflow as tf
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()

    print("Debug mode enabled: eager execution + eager tf.data")
    # Uncomment these lines only for debugging. For large runs or sweeps, disable to restore full performance.



    # %%
    def inspect_imitation_datasets(layout, dataset_a, dataset_b, sample_size=50_000):
        """
        Inspect dataset_a (field -> comm) and dataset_b (gun+comm -> shoot)
        to confirm that the imitation data matches the intended distribution
        and the MajorityPlayers logic.

        layout: GameLayout
        dataset_a: DataFrame with columns ["field", "comm", ...]
        dataset_b: DataFrame with columns ["field", "gun", "comm", "shoot", ...]
        sample_size: number of rows to subsample for stats (for speed).
        """
        import numpy as np
        import pandas as pd
        import Q_Sea_Battle as qsb

        print("=== Imitation Dataset Inspection ===")

        # -----------------------------
        # Subsample for speed
        # -----------------------------
        if len(dataset_a) > sample_size:
            da = dataset_a.sample(sample_size, random_state=0)
        else:
            da = dataset_a

        if len(dataset_b) > sample_size:
            db = dataset_b.sample(sample_size, random_state=0)
        else:
            db = dataset_b

        n2 = layout.field_size ** 2
        m  = layout.comms_size

        # -----------------------------
        # A1 — Field distribution p_one
        # -----------------------------
        fields = np.stack(da["field"].to_numpy(), axis=0).astype(float)
        p_emp = fields.mean()
        print(f" A1: Field mean p_one (empirical): {p_emp:.4f}")

        # -----------------------------
        # A2 — Comm bit frequencies
        # -----------------------------
        comms_a = np.stack(da["comm"].to_numpy(), axis=0)
        print(" A2: Comm bit frequencies (dataset A):")
        for j in range(m):
            print(f"  bit {j}: mean={comms_a[:, j].mean():.4f}")

        # -----------------------------
        # B1 — Gun index uniformity
        # -----------------------------
        guns_b = np.stack(db["gun"].to_numpy(), axis=0)
        gun_idx = guns_b.argmax(axis=1)
        print(" B1: Gun index statistics (dataset B):")
        print(f"  min idx={gun_idx.min()}, max idx={gun_idx.max()}")
        print(f"  approx std(index) = {gun_idx.std():.1f} (uniform-ish if large)")

        # -----------------------------
        # B2 — Shoot distribution
        # -----------------------------
        shoots = np.array(db["shoot"].to_numpy(), dtype=float)
        print(f" B2: Shoot distribution: mean(shoot)={shoots.mean():.4f}")

        # -----------------------------
        # C1 — Cross-check B vs MajorityPlayers
        # -----------------------------
        print(" C1: Cross-check dataset B vs MajorityPlayers.playerB ...")

        maj = qsb.MajorityPlayers(layout)
        player_a_maj, player_b_maj = maj.players()

        subset_b = db.sample(min(2000, len(db)), random_state=1)

        mismatches_b = 0
        for _, row in subset_b.iterrows():
            gun   = row["gun"]
            comm  = row["comm"]
            shoot_ds = int(row["shoot"])

            shoot_maj = int(player_b_maj.decide(gun=gun, comm=comm))
            if shoot_ds != shoot_maj:
                mismatches_b += 1

        mismatch_rate_b = mismatches_b / len(subset_b)
        print(f"  Majority vs Dataset-B shoot mismatch rate: {mismatch_rate_b:.4f}")

        # -----------------------------
        # C2 — Cross-check A vs MajorityPlayers
        # -----------------------------
        print(" C2: Cross-check dataset A vs MajorityPlayers.playerA ...")

        subset_a = da.sample(min(2000, len(da)), random_state=2)

        mismatches_a = 0
        for _, row in subset_a.iterrows():
            field = row["field"]
            comm  = row["comm"]

            comm_maj = player_a_maj.decide(field)
            if not np.array_equal(comm, comm_maj):
                mismatches_a += 1

        mismatch_rate_a = mismatches_a / len(subset_a)
        print(f"  Majority vs Dataset-A comm mismatch rate: {mismatch_rate_a:.4f}")

        print(" === Inspection Completed ===")


    # %%
    # ----------------------------
    # Global configuration
    # ----------------------------

    FIELD_SIZES = [64,16,4]
    COMMS_SIZES = [8,1]

    # Number of synthetic samples per imitation batch.
    # We keep the *total* number of synthetic samples roughly comparable to before,
    # but avoid a single huge DataFrame in memory.
    NUM_SAMPLES_A = 10   # for model_a (field -> comm), per batch
    NUM_SAMPLES_B = 10   # for model_b (gun + comm -> shoot), per batch

    # How many independent imitation batches to generate and train on.
    NUM_IM_BATCHES_A = 5
    NUM_IM_BATCHES_B = 5

    # Training hyper-parameters for imitation
    TRAINING_SETTINGS_A = {
        "epochs": 5,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "verbose": 0,
        "use_sample_weight": False,
    }

    TRAINING_SETTINGS_B = {
        "epochs": 25,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "verbose": 0,
        "use_sample_weight": False,
    }

    # Tournament configuration
    NUM_GAMES_TOURNAMENT = 5_0

    # Base seed for reproducibility
    BASE_SEED = 12345


    # %%
    
    def run_single_experiment(field_size: int, comms_size: int, seed: int = 0): 
        """ 
        Run imitation-training and tournaments for a single (field_size, comms_size). 
    
        Returns a dict with summary statistics. 
        """ 
        n2 = field_size ** 2 
        if comms_size > n2: 
            raise ValueError("comms_size cannot exceed field_size**2 in this setup.") 
    
        # 1) Build layout and environment 
        layout = GameLayout( 
            field_size=field_size, 
            comms_size=comms_size, 
            number_of_games_in_tournament=NUM_GAMES_TOURNAMENT, 
        ) 
    
        game_env = GameEnv(layout) 
    
        # 2) Evaluate MajorityPlayers teacher 
        majority_players = MajorityPlayers(layout) 
        tournament_teacher = Tournament(game_env, majority_players, layout) 
        log_teacher = tournament_teacher.tournament() 
        maj_mean, maj_stderr = log_teacher.outcome() 
    
        # 3) Build NeuralNetPlayers student 
        nn_players = NeuralNetPlayers(layout) 
    
        # 4) Imitation training with on-the-fly batch generation 
        # 
        # Instead of generating one *huge* imitation dataset and training on it multiple 
        # epochs, we now: 
        #   - generate a fresh synthetic imitation batch for each step, and 
        #   - run a short training phase on that batch. 
        # 
        # This keeps the peak memory usage bounded by NUM_SAMPLES_A / NUM_SAMPLES_B, 
        # while the *total* number of synthetic samples seen during training is 
        # NUM_IM_BATCHES_* × NUM_SAMPLES_*. 
    
        # 4a) Train model_a (field -> comm) via imitation on multiple small batches 
        for k in range(NUM_IM_BATCHES_A): 
            dataset_a, _ = imitation_utils.generate_majority_imitation_datasets( 
                layout=layout, 
                num_samples_a=NUM_SAMPLES_A, 
                num_samples_b=NUM_SAMPLES_B, 
                seed=BASE_SEED + k, 
            ) 
            nn_players.train_model_a(dataset_a, TRAINING_SETTINGS_A) 
    
        # 4b) Train model_b (gun + comm -> shoot) via imitation on multiple small batches 
        for k in range(NUM_IM_BATCHES_B): 
            _, dataset_b = imitation_utils.generate_majority_imitation_datasets( 
                layout=layout, 
                num_samples_a=NUM_SAMPLES_A, 
                num_samples_b=NUM_SAMPLES_B, 
                seed=BASE_SEED + 100 + k, 
            ) 
            nn_players.train_model_b(dataset_b, TRAINING_SETTINGS_B) 
    
        # 5) Evaluate trained NeuralNetPlayers in a fresh tournament 
        #    (re-use the same layout but reset the environment) 
        game_env_nn = GameEnv(layout) 
        tournament_student = Tournament(game_env_nn, nn_players, layout) 
        log_student = tournament_student.tournament() 
        nn_mean, nn_stderr = log_student.outcome() 
    
        return { 
            "field_size": field_size, 
            "comms_size": comms_size, 
            "maj_mean": maj_mean, 
            "maj_stderr": maj_stderr, 
            "nn_mean": nn_mean, 
            "nn_stderr": nn_stderr, 
        } 

    # %%

    results = []

    for i, field_size in enumerate(FIELD_SIZES):
        for comms_size in COMMS_SIZES:
            # Only run configurations where comms_size is not trivially impossible
            if comms_size > field_size ** 2:
                continue

            print(f"\n=== Running experiment: field_size={field_size}, comms_size={comms_size}, sample_size={NUM_SAMPLES_A} ===")
            summary = run_single_experiment(field_size, comms_size, seed=i)
            results.append(summary)
            print(
                f"Majority: mean={summary['maj_mean']:.4f}, stderr={summary['maj_stderr']:.4f} | "
                f"Neural: mean={summary['nn_mean']:.4f}, stderr={summary['nn_stderr']:.4f}"
            )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["field_size", "comms_size"]).reset_index(drop=True)

    assert results_df is not None

