# P_HIGH sweep results
# ======================================================
#   P_HIGH |   win_rate | source
# ------------------------------------------------------
#     1.00 |     1.0000 | summary.mean_reward_fallback
#     0.99 |     0.9710 | summary.mean_reward_fallback
#     0.98 |     0.9330 | summary.mean_reward_fallback
#     0.97 |     0.8900 | summary.mean_reward_fallback
#     0.96 |     0.8420 | summary.mean_reward_fallback
#     0.95 |     0.8150 | summary.mean_reward_fallback
#     0.94 |     0.7950 | summary.mean_reward_fallback
#     0.93 |     0.7930 | summary.mean_reward_fallback
#     0.92 |     0.7500 | summary.mean_reward_fallback
#     0.91 |     0.7250 | summary.mean_reward_fallback
#     0.90 |     0.6790 | summary.mean_reward_fallback
#     0.89 |     0.6640 | summary.mean_reward_fallback
#     0.88 |     0.6870 | summary.mean_reward_fallback
#     0.87 |     0.6720 | summary.mean_reward_fallback
#     0.86 |     0.6300 | summary.mean_reward_fallback
#     0.85 |     0.6140 | summary.mean_reward_fallback
#     0.84 |     0.6080 | summary.mean_reward_fallback
#     0.83 |     0.6060 | summary.mean_reward_fallback
#     0.82 |     0.5900 | summary.mean_reward_fallback
#     0.81 |     0.5770 | summary.mean_reward_fallback
#     0.80 |     0.5680 | summary.mean_reward_fallback
#     0.79 |     0.5380 | summary.mean_reward_fallback
#     0.78 |     0.5420 | summary.mean_reward_fallback
#     0.77 |     0.5380 | summary.mean_reward_fallback
#     0.76 |     0.5290 | summary.mean_reward_fallback
#     0.75 |     0.5140 | summary.mean_reward_fallback
#     0.74 |     0.5210 | summary.mean_reward_fallback
#     0.73 |     0.5400 | summary.mean_reward_fallback
#     0.72 |     0.5110 | summary.mean_reward_fallback
#     0.71 |     0.5120 | summary.mean_reward_fallback
#     0.70 |     0.5200 | summary.mean_reward_fallback
#     0.69 |     0.5020 | summary.mean_reward_fallback
#     0.68 |     0.4870 | summary.mean_reward_fallback
#     0.67 |     0.5190 | summary.mean_reward_fallback
#     0.66 |     0.5010 | summary.mean_reward_fallback
#     0.65 |     0.4950 | summary.mean_reward_fallback
#     0.64 |     0.5030 | summary.mean_reward_fallback
#     0.63 |     0.4690 | summary.mean_reward_fallback
#     0.62 |     0.4960 | summary.mean_reward_fallback
#     0.61 |     0.5150 | summary.mean_reward_fallback
#     0.60 |     0.5150 | summary.mean_reward_fallback
#     0.59 |     0.5150 | summary.mean_reward_fallback
#     0.58 |     0.5210 | summary.mean_reward_fallback
#     0.57 |     0.5060 | summary.mean_reward_fallback
#     0.56 |     0.4900 | summary.mean_reward_fallback
#     0.55 |     0.5220 | summary.mean_reward_fallback
#     0.54 |     0.5060 | summary.mean_reward_fallback
#     0.53 |     0.5170 | summary.mean_reward_fallback
#     0.52 |     0.4940 | summary.mean_reward_fallback
#     0.51 |     0.4850 | summary.mean_reward_fallback
#     0.50 |     0.5070 | summary.mean_reward_fallback

from __future__ import annotations

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent
TRAINING_SCRIPT = THIS_DIR / "train_Phase_pre_gameplay_STE - CONFIGURABLE_multi_phase.py"
TOURNAMENT_SCRIPT = THIS_DIR / "pyr_tournament_eval_standalone.py"
ROOT = THIS_DIR.parent


# -----------------------------------------------------------------------------
# Default configs
# -----------------------------------------------------------------------------
DEFAULT_MODEL_CFG: dict[str, Any] = {
    "P_HIGH": 1.0,
    "MODEL_BETA": 10.0,
    "ALPHA_FOR_PR_LAYERS": 0.3,
}

DEFAULT_TEACHER_DATASET_CFG: dict[str, Any] = {
    "NUM_GAMES_DATASET": 250_000,
    "BETA_INPUT": 10.0,
}

DEFAULT_TRAINING_CFG: dict[str, Any] = {
    "N2": 16,
    "FIELD_SIZE": 4,
    "COMMS_SIZE": 1,
    "DEPTH": 4,
    "CHANNEL_NOISE": 0.0,
    "ENEMY_PROBABILITY": 0.5,
    "SEED": 42,
    "BATCH": 32,
    "EPOCHS": 1000,
    "START_EPOCH": 0,
    "ROOT": ROOT,
    "WIP": ROOT / "WIP",
    "DATASET_DIR": ROOT / "WIP" / "dataset",
    "CHECKPOINT_DIR": ROOT / "WIP" / "checkpoints",
    "LOG_DIR": ROOT / "WIP" / "logs",
    "SAVE_WEIGHTS_EVERY": 0,
    "SAVE_WEIGHTS_AT_END": False,
    "LOG_FLUSH_EVERY_EPOCHS": 0,
    "VERBOSITY": 1,
}

# Exact B-only training settings requested by user, mapped to the current
# configurable trainer format.
# - w_meas_in_a=[0.02,0.02,0.02,0.02]  -> global 1.0, per-level exact list
# - w_comms_a=0.05                     -> w_comm_a_global
# - w_comms_b_123=[0.0,0.02,0.02,0.02]
#   The current trainer expects only levels 1..3, so we use the last three values with "w_comms_b_global"=0.0 to achieve the same effect.
DEFAULT_STAGE_CFG: dict[str, Any] = {
    "update_a": False,
    "update_b": True,
    "lr_a": 1e-5,
    "lr_b": 1e-5,
}

DEFAULT_LOSS_CFG: dict[str, Any] = {
    "w_meas_a_global": 1.0,
    "w_comms_b_global": 0.0,
    "w_meas_a_per_level": [0.02, 0.02, 0.02, 0.02],
    "w_comms_b_per_level": [0.02, 0.02, 0.02],
    "w_comm_a_global": 0.05,
    "w_shoot": 2.0,
    "w_comms_b_mag": float(0.0001),
    "w_comm_a_mag": float(0.0001),
    "w_meas_a_mag": float(0.0001),
    "w_meas_b_mag": float(0.0001),
    "mag_target": 6.0,
}

# Frozen A -> B interface:
#   comm_fixed = stop_gradient(comm_logits)
#   meas_fixed = stop_gradient(meas_list)
#   out_fixed  = stop_gradient(out_list)
#   comm/meas hard forward with no gradients, out unchanged with no gradients.
DEFAULT_INTERFACE_CFG: dict[str, Any] = {
    "comm": {
        "stop_gradient": True,
        "hardening": {"mode": 1, "beta": 10.0, "lambda": 0.0},
        "noise_std": 0.0,
    },
    "meas": {
        "stop_gradient": True,
        "hardening": {"mode": 1, "beta": 10.0, "lambda": 0.0},
        "noise_std": 0.0,
    },
    "out": {
        "stop_gradient": True,
        "hardening": {"mode": 0, "beta": 10.0, "lambda": 0.0},
        "noise_std": 0.0,
    },
}

DEFAULT_TOURNAMENT_CFG: dict[str, Any] = {
    "ROOT": str(ROOT),
    "FIELD_SIZE": 4,
    "COMMS_SIZE": 1,
    "GAMES_IN_EVAL_TOURNAMENT": 1000,
    "CHANNEL_NOISE": 0.0,
    "ENEMY_PROBABILITY": 0.5,
    "GAMEPLAY_ADAPTER_BETA": 10.0,
    "HARDENING_BETWEEN_LEVELS_FOR_MODEL_A": False,
    "HARDENING_BETWEEN_LEVELS_FOR_MODEL_B": False,
    "MODEL_A_WEIGHTS_IN": "",
    "MODEL_B_WEIGHTS_IN": "",
}

DEFAULT_START_WEIGHTS: dict[str, str] = {
    "model_a": r"C:\Users\nly99857\OneDrive - Philips\SW Projects\QSeaBattle\WIP\checkpoints\model_a_step100.weights.h5",
    "model_b": r"C:\Users\nly99857\OneDrive - Philips\SW Projects\QSeaBattle\WIP\checkpoints\model_b_step100.weights.h5",
}


# -----------------------------------------------------------------------------
# Loading helper modules
# -----------------------------------------------------------------------------

def _load_module(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name!r} from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _effective_teacher_cfg(training_cfg: dict[str, Any], teacher_dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        **teacher_dataset_cfg,
        "DATASET_DIR": str(training_cfg["DATASET_DIR"]),
        "BATCH": training_cfg["BATCH"],
        "SEED": training_cfg["SEED"],
    }


def _effective_model_cfg(training_mod, training_cfg: dict[str, Any], model_cfg: dict[str, Any], p_high: float) -> dict[str, Any]:
    layout = training_mod.GameLayout(
        field_size=training_cfg["FIELD_SIZE"],
        comms_size=training_cfg["COMMS_SIZE"],
        number_of_games_in_tournament=1000,
        channel_noise=training_cfg["CHANNEL_NOISE"],
        enemy_probability=training_cfg["ENEMY_PROBABILITY"],
    )
    return {
        **model_cfg,
        "FIELD_SIZE": training_cfg["FIELD_SIZE"],
        "COMMS_SIZE": training_cfg["COMMS_SIZE"],
        "SEED": training_cfg["SEED"],
        "LAYOUT": layout,
        "P_HIGH": float(p_high),
    }


def _make_tf_constants(training_mod, stage_cfg: dict[str, Any], loss_cfg: dict[str, Any], interface_cfg: dict[str, Any]):
    tf = training_mod.tf
    return {
        "learning_rate_a": tf.constant(stage_cfg["lr_a"], dtype=tf.float32),
        "learning_rate_b": tf.constant(stage_cfg["lr_b"], dtype=tf.float32),
        "update_a": stage_cfg["update_a"],
        "update_b": stage_cfg["update_b"],
        "w_comm_a_global": tf.constant(loss_cfg["w_comm_a_global"], dtype=tf.float32),
        "w_meas_a_global": tf.constant(loss_cfg["w_meas_a_global"], dtype=tf.float32),
        "w_meas_a_per_level": tf.constant(loss_cfg["w_meas_a_per_level"], dtype=tf.float32),
        "w_comms_b_global": tf.constant(loss_cfg["w_comms_b_global"], dtype=tf.float32),
        "w_comms_b_per_level": tf.constant(loss_cfg["w_comms_b_per_level"], dtype=tf.float32),
        "w_shoot": tf.constant(loss_cfg["w_shoot"], dtype=tf.float32),
        "w_comm_a_mag": tf.constant(loss_cfg["w_comm_a_mag"], dtype=tf.float32),
        "w_comms_b_mag": tf.constant(loss_cfg["w_comms_b_mag"], dtype=tf.float32),
        "w_meas_a_mag": tf.constant(loss_cfg["w_meas_a_mag"], dtype=tf.float32),
        "w_meas_b_mag": tf.constant(loss_cfg["w_meas_b_mag"], dtype=tf.float32),
        "beta_target": tf.constant(loss_cfg["mag_target"], dtype=tf.float32),
        "comm_stop_gradient": tf.constant(interface_cfg["comm"]["stop_gradient"]),
        "comm_hard_mode": tf.constant(interface_cfg["comm"]["hardening"]["mode"], dtype=tf.int32),
        "comm_beta": tf.constant(interface_cfg["comm"]["hardening"]["beta"], dtype=tf.float32),
        "comm_lambda": tf.constant(interface_cfg["comm"]["hardening"]["lambda"], dtype=tf.float32),
        "comm_noise_std": tf.constant(interface_cfg["comm"]["noise_std"], dtype=tf.float32),
        "meas_stop_gradient": tf.constant(interface_cfg["meas"]["stop_gradient"]),
        "meas_hard_mode": tf.constant(interface_cfg["meas"]["hardening"]["mode"], dtype=tf.int32),
        "meas_beta": tf.constant(interface_cfg["meas"]["hardening"]["beta"], dtype=tf.float32),
        "meas_lambda": tf.constant(interface_cfg["meas"]["hardening"]["lambda"], dtype=tf.float32),
        "meas_noise_std": tf.constant(interface_cfg["meas"]["noise_std"], dtype=tf.float32),
        "out_stop_gradient": tf.constant(interface_cfg["out"]["stop_gradient"], dtype=tf.bool),
        "out_hard_mode": tf.constant(interface_cfg["out"]["hardening"]["mode"], dtype=tf.int32),
        "out_beta": tf.constant(interface_cfg["out"]["hardening"]["beta"], dtype=tf.float32),
        "out_lambda": tf.constant(interface_cfg["out"]["hardening"]["lambda"], dtype=tf.float32),
        "out_noise_std": tf.constant(interface_cfg["out"]["noise_std"], dtype=tf.float32),
    }


def _tag_for_phigh(p_high: float) -> str:
    return f"phigh_{p_high:.2f}".replace(".", "p") + "_best"


def _extract_tournament_win_rate(result: dict[str, Any]) -> tuple[float, str]:
    summary = result.get("summary", {})
    if "win_rate" in summary:
        return float(summary["win_rate"]), "summary.win_rate"

    log = result.get("objects", {}).get("log")
    if log is not None:
        for attr in ("win_rate", "mean_win_rate"):
            value = getattr(log, attr, None)
            if isinstance(value, (int, float)):
                return float(value), f"log.{attr}"
            if callable(value):
                out = value()
                if isinstance(out, (int, float)):
                    return float(out), f"log.{attr}()"
        wins = getattr(log, "wins", None)
        games = getattr(log, "games", None)
        if isinstance(wins, (int, float)) and isinstance(games, (int, float)) and games:
            return float(wins) / float(games), "log.wins/log.games"

    if "mean_reward" in summary:
        return float(summary["mean_reward"]), "summary.mean_reward_fallback"

    raise KeyError("Could not extract tournament win rate from tournament result.")


def _print_table(rows: list[dict[str, Any]]) -> None:
    print("\nP_HIGH sweep results")
    print("=" * 54)
    print(f"{'P_HIGH':>8} | {'win_rate':>10} | {'source':<24}")
    print("-" * 54)
    for row in rows:
        print(f"{row['p_high']:8.2f} | {row['win_rate']:10.4f} | {row['win_rate_source']:<24}")
    print("=" * 54)


def _plot_results(rows: list[dict[str, Any]]) -> None:
    p_vals = [row["p_high"] for row in rows]
    wins = [row["win_rate"] for row in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(p_vals, wins, marker="o")
    plt.xlabel("P_HIGH")
    plt.ylabel("win_rate")
    plt.title("Tournament win rate vs P_HIGH")
    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.show()


def run_p_high_sweep(
    *,
    model_cfg: dict[str, Any] | None = None,
    teacher_dataset_cfg: dict[str, Any] | None = None,
    training_cfg: dict[str, Any] | None = None,
    stage_cfg: dict[str, Any] | None = None,
    loss_cfg: dict[str, Any] | None = None,
    interface_cfg: dict[str, Any] | None = None,
    tournament_cfg: dict[str, Any] | None = None,
    start_weights: dict[str, str] | None = None,
    patience_epochs_since_best: int = 10,
    print_progress: bool = True,
    plot: bool = True,
) -> dict[str, Any]:
    training_mod = _load_module("qsb_training_mod", TRAINING_SCRIPT)
    tournament_mod = _load_module("qsb_tournament_mod", TOURNAMENT_SCRIPT)

    model_cfg = deepcopy(model_cfg or DEFAULT_MODEL_CFG)
    teacher_dataset_cfg = deepcopy(teacher_dataset_cfg or DEFAULT_TEACHER_DATASET_CFG)
    training_cfg = deepcopy(training_cfg or DEFAULT_TRAINING_CFG)
    stage_cfg = deepcopy(stage_cfg or DEFAULT_STAGE_CFG)
    loss_cfg = deepcopy(loss_cfg or DEFAULT_LOSS_CFG)
    interface_cfg = deepcopy(interface_cfg or DEFAULT_INTERFACE_CFG)
    tournament_cfg = deepcopy(tournament_cfg or DEFAULT_TOURNAMENT_CFG)
    start_weights = deepcopy(start_weights or DEFAULT_START_WEIGHTS)

    # Make module-level verbosity-controlled helpers use this configuration.
    training_mod.training_cfg = training_cfg

    effective_teacher_cfg = _effective_teacher_cfg(training_cfg, teacher_dataset_cfg)
    raw_ds = training_mod.load_dataset(effective_teacher_cfg)
    tfds_train = training_mod.build_train_pipeline(raw_ds, effective_teacher_cfg)
    tfds_train = tfds_train.prefetch(training_mod.tf.data.AUTOTUNE)

    first_batch = next(iter(tfds_train))
    p_values = [round(x / 100.0, 2) for x in range(100, 49, -1)]
    results: list[dict[str, Any]] = []

    model_a = None
    model_b = None
    prev_best_a = None
    prev_best_b = None

    for round_idx, p_high in enumerate(p_values):
        effective_model_cfg = _effective_model_cfg(training_mod, training_cfg, model_cfg, p_high)

        if model_a is None or model_b is None:
            training_mod.tf.random.set_seed(training_cfg["SEED"])
            model_a = training_mod.build_model_a(effective_model_cfg)
            model_b = training_mod.build_model_b(effective_model_cfg)
            training_mod._force_build_models(model_a, model_b, first_batch)
        else:
            if hasattr(model_a, "set_p_high"):
                model_a.set_p_high(p_high)
            if hasattr(model_b, "set_p_high"):
                model_b.set_p_high(p_high)

        # Load round start weights.
        if round_idx == 0:
            a_in = Path(start_weights["model_a"])
            b_in = Path(start_weights["model_b"])
        else:
            a_in = Path(prev_best_a)
            b_in = Path(prev_best_b)

        loaded = training_mod.load_model_weights(model_a, model_b, a_in, b_in)
        if not loaded:
            raise FileNotFoundError(f"Failed to load round-start weights for P_HIGH={p_high:.2f}: {a_in}, {b_in}")

        opt_a = training_mod.tf.keras.optimizers.Adam(stage_cfg["lr_a"])
        opt_b = training_mod.tf.keras.optimizers.Adam(stage_cfg["lr_b"])
        opt_a.build(model_a.trainable_variables)
        opt_b.build(model_b.trainable_variables)
        train_step = training_mod.make_train_steps(depth=training_cfg["DEPTH"])

        best_shoot_acc = float("-inf")
        epochs_since_best = 0
        best_tag = _tag_for_phigh(p_high)
        best_a = None
        best_b = None
        stopped_early = False

        for epoch in range(int(training_cfg["EPOCHS"])):
            metrics_total = training_mod.tf.keras.metrics.Mean()
            metrics_shoot_acc = training_mod.tf.keras.metrics.Mean()
            tf_args = _make_tf_constants(training_mod, stage_cfg, loss_cfg, interface_cfg)

            for batch in tfds_train:
                out = train_step(
                    batch=batch,
                    model_a=model_a,
                    model_b=model_b,
                    opt_a=opt_a,
                    opt_b=opt_b,
                    **tf_args,
                )
                metrics_total.update_state(out["total_loss"])
                metrics_shoot_acc.update_state(out["shoot_acc"])

            shoot_acc = float(metrics_shoot_acc.result().numpy())
            total_loss = float(metrics_total.result().numpy())

            if shoot_acc > best_shoot_acc:
                best_shoot_acc = shoot_acc
                epochs_since_best = 0
                path_a, path_b = training_mod.save_model_weights(
                    model_a,
                    model_b,
                    Path(training_cfg["CHECKPOINT_DIR"]),
                    tag=best_tag,
                )
                best_a = str(path_a)
                best_b = str(path_b)
            else:
                epochs_since_best += 1

            epoch_metrics = {
                "epoch": epoch,
                "shoot_acc": shoot_acc,
                "total_loss": total_loss,
                "best_shoot_acc": best_shoot_acc,
                "epochs_since_best": float(epochs_since_best),
                "p_high": float(p_high),
            }

            if print_progress:
                print(
                    f"[train] P_HIGH={p_high:.2f} epoch={epoch:04d} "
                    f"shoot_acc={shoot_acc:.4f} best={best_shoot_acc:.4f} "
                    f"epochs_since_best={epochs_since_best} total_loss={total_loss:.4f}",
                    flush=True,
                )

            if epoch_metrics["epochs_since_best"] > float(patience_epochs_since_best):
                stopped_early = True
                if print_progress:
                    print(
                        f"[stop] P_HIGH={p_high:.2f} stopping because epochs_since_best="
                        f"{epoch_metrics['epochs_since_best']:.0f} > {patience_epochs_since_best}",
                        flush=True,
                    )
                break

        if best_a is None or best_b is None:
            raise RuntimeError(f"No best checkpoint was saved for P_HIGH={p_high:.2f}.")

        prev_best_a = best_a
        prev_best_b = best_b

        # Tournament uses best weights from this round.
        round_model_cfg = {
            **model_cfg,
            "N2": training_cfg["N2"],
            "FIELD_SIZE": training_cfg["FIELD_SIZE"],
            "COMMS_SIZE": training_cfg["COMMS_SIZE"],
            "SEED": training_cfg["SEED"],
            "P_HIGH": p_high,
            "BETA_INPUT": teacher_dataset_cfg["BETA_INPUT"],
        }
        round_dataset_cfg = {
            **teacher_dataset_cfg,
            "DATASET_DIR": str(training_cfg["DATASET_DIR"]),
            "BATCH": training_cfg["BATCH"],
            "SEED": training_cfg["SEED"],
        }
        round_tournament_cfg = {
            **tournament_cfg,
            "ROOT": str(training_cfg["ROOT"]),
            "FIELD_SIZE": training_cfg["FIELD_SIZE"],
            "COMMS_SIZE": training_cfg["COMMS_SIZE"],
            "MODEL_A_WEIGHTS_IN": best_a,
            "MODEL_B_WEIGHTS_IN": best_b,
        }
        tournament_result = tournament_mod.run_tournament_evaluation(
            model_cfg=round_model_cfg,
            tournament_cfg=round_tournament_cfg,
            dataset_cfg=round_dataset_cfg,
            verbose=False,
        )
        win_rate, source = _extract_tournament_win_rate(tournament_result)

        row = {
            "p_high": float(p_high),
            "win_rate": float(win_rate),
            "win_rate_source": source,
            "best_shoot_acc": float(best_shoot_acc),
            "best_model_a_path": best_a,
            "best_model_b_path": best_b,
            "stopped_early": bool(stopped_early),
        }
        results.append(row)

        if print_progress:
            print(f"[tournament] P_HIGH={p_high:.2f} win_rate={win_rate:.4f}", flush=True)

    _print_table(results)
    if plot:
        _plot_results(results)

    return {
        "results": results,
        "start_weights": start_weights,
        "training_cfg": training_cfg,
        "stage_cfg": stage_cfg,
        "loss_cfg": loss_cfg,
        "interface_cfg": interface_cfg,
        "tournament_cfg": tournament_cfg,
        "teacher_dataset_cfg": teacher_dataset_cfg,
        "patience_epochs_since_best": int(patience_epochs_since_best),
    }


if __name__ == "__main__":
    run_p_high_sweep()
