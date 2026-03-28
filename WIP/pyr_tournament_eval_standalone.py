from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Keep before tensorflow import
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf


# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------

def _ensure_repo_paths(root: Path) -> None:
    """Make sure WIP/src, src, and WIP are importable."""
    root = root.resolve()
    wip_src = root / "WIP" / "src"
    core_src = root / "src"
    wip = root / "WIP"
    for p in (wip_src, core_src, wip):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def _find_repo_root(start: Path | None = None) -> Path:
    """Find repository root by looking for the WIP and src folders."""
    here = (start or Path.cwd()).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "WIP").is_dir() and (parent / "src").is_dir():
            return parent
    raise FileNotFoundError("Could not locate repository root containing both 'WIP' and 'src'.")


ROOT = _find_repo_root(Path(__file__).resolve().parent)
_ensure_repo_paths(ROOT)


# -----------------------------------------------------------------------------
# Imports that depend on repo paths
# -----------------------------------------------------------------------------
from Q_Sea_Battle.game_env import GameEnv
from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.gameplay_adapters import GameplayModelAAdapter, GameplayModelBAdapter
from Q_Sea_Battle.tournament import Tournament
from Q_Sea_Battle.trainable_assisted_players import TrainableAssistedPlayers
from Q_Sea_Battle_New.pyr_internal_model_a import PyrInternalModelA
from Q_Sea_Battle_New.pyr_internal_model_b import PyrInternalModelB
from data_build import build_train_pipeline, load_dataset
from helpers import load_ab_weights


# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------

def _resolve_path(root: Path, value: str | os.PathLike[str] | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return (root / p).resolve()



def _unpack_batch(batch: Any) -> tuple[
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
    list[tf.Tensor],
    list[tf.Tensor],
    list[tf.Tensor],
]:
    field0, gun0, shoot_tgt_logits, meas_in_a_tgt_list, meas_out_a_tgt_list, comms_tgt_list = batch
    return (
        tf.cast(field0, tf.float32),
        tf.cast(gun0, tf.float32),
        tf.cast(shoot_tgt_logits, tf.float32),
        [tf.cast(x, tf.float32) for x in meas_in_a_tgt_list],
        [tf.cast(x, tf.float32) for x in meas_out_a_tgt_list],
        [tf.cast(x, tf.float32) for x in comms_tgt_list],
    )



def build_model_a(config: dict[str, Any]) -> tf.keras.Model:
    n2 = int(config["N2"])
    field_size = int(config.get("FIELD_SIZE", int(np.sqrt(n2))))
    comms_size = int(config.get("COMMS_SIZE", 1))
    p_high = float(config.get("P_HIGH", 1.0))
    beta_input = float(config["BETA_INPUT"])
    seed = int(config["SEED"])
    alpha = float(config["ALPHA_FOR_PR_LAYERS"])

    layout = GameLayout(
        field_size=field_size,
        comms_size=comms_size,
        number_of_games_in_tournament=1000,
        channel_noise=0.0,
        enemy_probability=0.5,
    )

    return PyrInternalModelA(
        layout,
        sr_mode="stochastic",
        p_high=p_high,
        beta=beta_input,
        alpha=alpha,
        seed=seed + 10,
    )



def build_model_b(config: dict[str, Any]) -> tf.keras.Model:
    n2 = int(config["N2"])
    field_size = int(config.get("FIELD_SIZE", int(np.sqrt(n2))))
    comms_size = int(config.get("COMMS_SIZE", 1))
    p_high = float(config.get("P_HIGH", 1.0))
    beta_input = float(config["BETA_INPUT"])
    alpha = float(config["ALPHA_FOR_PR_LAYERS"])

    layout = GameLayout(
        field_size=field_size,
        comms_size=comms_size,
        number_of_games_in_tournament=1000,
        channel_noise=0.0,
        enemy_probability=0.5,
    )

    return PyrInternalModelB(
        layout,
        sr_mode="stochastic",
        p_high=p_high,
        beta=beta_input,
        alpha=alpha,
    )



def _force_build_models(model_a: tf.keras.Model, model_b: tf.keras.Model, sample_batch: Any) -> None:
    """Warmup build via real forward calls to create variables once."""
    field_logits, gun_logits, _, meas_in_a_tgt_list, meas_out_a_tgt_list, comms_tgt_list = _unpack_batch(sample_batch)

    _ = model_a.compute_with_internal(
        field_logits=field_logits,
        replay_out_a_logits_list=meas_out_a_tgt_list,
        harden_between_levels=False,
        training=False,
    )
    if hasattr(model_a, "_ensure_built"):
        model_a._ensure_built()

    comm0 = tf.cast(comms_tgt_list[0], tf.float32)
    _ = model_b.compute_with_internal(
        gun_logits=gun_logits,
        comm_in_logits=comm0,
        prev_meas_list=meas_in_a_tgt_list,
        prev_out_list=meas_out_a_tgt_list,
        training=False,
    )
    if hasattr(model_b, "_ensure_built"):
        model_b._ensure_built()



def _load_build_dataset(dataset_cfg: dict[str, Any]) -> tf.data.Dataset:
    raw_ds = load_dataset(dataset_cfg)
    tfds_train = build_train_pipeline(raw_ds, dataset_cfg)
    return tfds_train.prefetch(tf.data.AUTOTUNE)



def _make_layout_from_tournament_cfg(tournament_cfg: dict[str, Any]) -> GameLayout:
    return GameLayout(
        field_size=int(tournament_cfg["FIELD_SIZE"]),
        comms_size=int(tournament_cfg["COMMS_SIZE"]),
        number_of_games_in_tournament=int(tournament_cfg["GAMES_IN_EVAL_TOURNAMENT"]),
        channel_noise=float(tournament_cfg.get("CHANNEL_NOISE", 0.0)),
        enemy_probability=float(tournament_cfg.get("ENEMY_PROBABILITY", 0.5)),
    )



def run_tournament_evaluation(
    model_cfg: dict[str, Any],
    tournament_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run a tournament evaluation.

    Returns a conservative result schema with:
    - summary: aggregate scalar outputs that were previously printed
    - weight_info: resolved weight paths and load status
    - config: resolved config snapshots used for the run
    - objects: live Python objects that may be useful for callers
    """
    root = Path(tournament_cfg.get("ROOT", ROOT)).resolve()
    _ensure_repo_paths(root)

    effective_model_cfg = dict(model_cfg)
    effective_tournament_cfg = dict(tournament_cfg)
    effective_dataset_cfg = dict(dataset_cfg)

    effective_model_cfg.setdefault("ROOT", str(root))
    effective_model_cfg.setdefault("FIELD_SIZE", effective_tournament_cfg["FIELD_SIZE"])
    effective_model_cfg.setdefault("COMMS_SIZE", effective_tournament_cfg["COMMS_SIZE"])
    effective_model_cfg.setdefault("N2", int(effective_tournament_cfg["FIELD_SIZE"]) ** 2)

    effective_dataset_cfg.setdefault("ROOT", str(root))
    if "DATASET_DIR" in effective_dataset_cfg:
        effective_dataset_cfg["DATASET_DIR"] = str(_resolve_path(root, effective_dataset_cfg["DATASET_DIR"]))

    tf.random.set_seed(int(effective_model_cfg["SEED"]))

    if verbose:
        print("Building dataset and models...")
    tfds_train = _load_build_dataset(effective_dataset_cfg)
    internal_model_a = build_model_a(effective_model_cfg)
    internal_model_b = build_model_b(effective_model_cfg)

    first_batch = next(iter(tfds_train))
    _force_build_models(internal_model_a, internal_model_b, first_batch)

    a_path = _resolve_path(root, effective_tournament_cfg["MODEL_A_WEIGHTS_IN"])
    b_path = _resolve_path(root, effective_tournament_cfg["MODEL_B_WEIGHTS_IN"])
    loaded = load_ab_weights(internal_model_a, internal_model_b, a_path, b_path)
    if not loaded:
        raise FileNotFoundError(
            f"Failed to load model weights: model_a={a_path}, model_b={b_path}"
        )

    harden_a = bool(effective_tournament_cfg.get("HARDENING_BETWEEN_LEVELS_FOR_MODEL_A", False))
    harden_b = bool(effective_tournament_cfg.get("HARDENING_BETWEEN_LEVELS_FOR_MODEL_B", False))
    adapter_beta = float(effective_tournament_cfg.get("GAMEPLAY_ADAPTER_BETA", 10.0))

    model_a = GameplayModelAAdapter(
        internal_model_a=internal_model_a,
        beta=adapter_beta,
        harden_between_levels=harden_a,
    )
    model_b = GameplayModelBAdapter(
        internal_model_b=internal_model_b,
        beta=adapter_beta,
        harden_between_levels=harden_b,
    )

    layout_eval = _make_layout_from_tournament_cfg(effective_tournament_cfg)
    env = GameEnv(layout_eval)
    players = TrainableAssistedPlayers(layout_eval, model_a=model_a, model_b=model_b)

    if verbose:
        print("Running tournament...")
    tournament = Tournament(game_env=env, players=players, game_layout=layout_eval)
    log = tournament.tournament()
    mean_reward, std_err = log.outcome()
    if verbose:
        print("Tournament finished.")

    result = {
        "summary": {
            "mean_reward": float(mean_reward),
            "std_err": float(std_err),
            "games": int(layout_eval.number_of_games_in_tournament),
            "field_size": int(layout_eval.field_size),
            "comms_size": int(layout_eval.comms_size),
            "channel_noise": float(layout_eval.channel_noise),
            "enemy_probability": float(layout_eval.enemy_probability),
            "alpha_for_sr_layer": float(effective_model_cfg["ALPHA_FOR_PR_LAYERS"]),
            "p_high": float(effective_model_cfg.get("P_HIGH", 1.0)),
            "adapter_beta": float(adapter_beta),
            "harden_between_levels_model_a": bool(harden_a),
            "harden_between_levels_model_b": bool(harden_b),
        },
        "weight_info": {
            "loaded": bool(loaded),
            "model_a_path": str(a_path),
            "model_b_path": str(b_path),
        },
        "config": {
            "model_cfg": effective_model_cfg,
            "tournament_cfg": effective_tournament_cfg,
            "dataset_cfg": effective_dataset_cfg,
        },
        "objects": {
            "internal_model_a": internal_model_a,
            "internal_model_b": internal_model_b,
            "model_a_adapter": model_a,
            "model_b_adapter": model_b,
            "layout_eval": layout_eval,
            "env": env,
            "players": players,
            "tournament": tournament,
            "log": log,
        },
    }
    return result



def print_tournament_summary(result: dict[str, Any]) -> None:
    s = result["summary"]
    w = result["weight_info"]
    print(
        f"Pyramid bootstrap tournament over {s['games']}: "
        f"{s['mean_reward']:.4f} ± {s['std_err']:.4f}"
    )
    print(f"Weights used: model_a from {w['model_a_path']}, model_b from {w['model_b_path']}")
    print(f"Alpha for SR layer: {s['alpha_for_sr_layer']}")
    print(f"PR-assisted correlation parameter (P_HIGH): {s['p_high']}")
    print(
        "Harden between levels: "
        f"model_a={s['harden_between_levels_model_a']}, "
        f"model_b={s['harden_between_levels_model_b']}"
    )


# -----------------------------------------------------------------------------
# Hard-coded defaults for standalone use
# -----------------------------------------------------------------------------
DEFAULT_MODEL_CFG: dict[str, Any] = {
    "N2": 16,
    "FIELD_SIZE": 4,
    "COMMS_SIZE": 1,
    "SEED": 1234,
    "P_HIGH": 1.0,
    "BETA_INPUT": 10.0,
    "ALPHA_FOR_PR_LAYERS": 0.3,
}

DEFAULT_DATASET_CFG: dict[str, Any] = {
    "DATASET_DIR": "WIP/dataset",
    "NUM_GAMES_DATASET": 250_000,
    "BATCH": 32,
    "SEED": 1234,
    "BETA_INPUT": 10.0,
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
    "MODEL_A_WEIGHTS_IN": "WIP/checkpoints/combined_ab/model_a_latest.weights.h5",
    "MODEL_B_WEIGHTS_IN": "WIP/checkpoints/combined_ab/model_b_latest.weights.h5",
}


if __name__ == "__main__":
    result = run_tournament_evaluation(
        model_cfg=DEFAULT_MODEL_CFG,
        tournament_cfg=DEFAULT_TOURNAMENT_CFG,
        dataset_cfg=DEFAULT_DATASET_CFG,
        verbose=True,
    )
    print_tournament_summary(result)
