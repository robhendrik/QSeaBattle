from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import sys
import os
from copy import deepcopy

from WIP.train_modelAB_combined_total_script import LOG_DIR

# Logging / performance knobs (set BEFORE importing tensorflow)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# User requested oneDNN enabled.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

import tensorflow as tf
import io
from contextlib import redirect_stdout

from config import get_config
from data_build import load_dataset, build_train_pipeline
from helpers import (
    _iso_mtime,
    _latest_epoch_pair,
    _wfile,
    append_epoch_log,
    flush_run_log,
    init_run_log,
    load_ab_weights,
    save_ab_weights,
    weighted_per_level_bce,
    add_logit_noise_tf,
    harden_ste,
    magnitude_margin_loss,
    magnitude_target_loss,
)

def _ensure_repo_paths(config: dict[str, Any]) -> None:
    """Make sure WIP/src, src, and WIP are importable."""
    root = Path(config["ROOT"]).resolve()
    wip_src = root / "WIP" / "src"
    core_src = root / "src"
    wip = root / "WIP"
    for p in (wip_src, core_src, wip):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def _suppress_tf_output(func):
    """Decorator to suppress TensorFlow verbose output during function execution."""
    def wrapper(*args, **kwargs):
        with redirect_stdout(io.StringIO()):
            return func(*args, **kwargs)
    return wrapper


_ensure_repo_paths(get_config())

ROOT = Path(__file__).resolve().parent.parent   # WIP/config.py -> ROOT

from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle_New.pyr_internal_model_b import PyrInternalModelB
from Q_Sea_Battle_New.pyr_internal_model_a import PyrInternalModelA

# TODO: Add print verbosity settings level 0 (Nothing) 1 skip details epoch info 2 full info including losses, 3+ add debug info like weight norms, gradient norms, etc.
early_stop_cfg = {
        "consecutive": 3,
        "rules": [
            {
                "shoot_acc": {"comparator": "gt", "value": 0.999}
            }
        ]
    }
stage_cfg = {
    "update_a" : False,
    "update_b" : True,
    "lr_a" : 1e-5,
    "lr_b" : 1e-5,
    
}
      
loss_cfg = {
    "w_meas_a_global": 0.2,
    "w_comms_b_global": 0.02,
    "w_meas_a_per_level" : [1.0, 1.0, 1.0, 1.0], 
    "w_comms_b_per_level" : [0.02, 0.02, 0.02], 
    "w_comm_a_global" : 0.05,
    "w_shoot" : 2.0,
    "w_comms_b_mag": float(0.0001),
    "w_comm_a_mag": float(0.0001),
    "w_meas_a_mag": float(0.0001),
    "w_meas_b_mag": float(0.0001),
    "mag_target" : 6.0,
}
interface_cfg = {
        # "mode": 2 for interp, "mode": 0 for none, "mode": 1 for hard.
        "comm": {"stop_gradient": False, "hardening": {"mode": 2, "beta": 10.0, "lambda": 0.0}, "noise_std": 0.30},
        "meas": {"stop_gradient": False, "hardening": {"mode": 0, "beta": 10.0, "lambda": 0.0}, "noise_std": 0.30},
        "out":  {"stop_gradient": False, "hardening": {"mode": 0, "beta": 10.0, "lambda": 0.0}, "noise_std": 0.30},
    }
# Change the interface_cfg mode values to integers: "mode": 2 for interp, "mode": 0 for none, "mode": 1 for hard.
model_cfg = {
        "P_HIGH": 1.0,
        "MODEL_BETA": 10.0,
        "ALPHA_FOR_PR_LAYERS": 0.3,
    }
teacher_dataset_cfg = {
        "NUM_GAMES_DATASET": 250_000,
        "BETA_INPUT": 10.0
    }
training_cfg = {
        # Geometry, this should not change in a training run.
        "N2": 16,
        "FIELD_SIZE": 4,
        "COMMS_SIZE": 1,
        "DEPTH": 4,
        "CHANNEL_NOISE": 0.0,
        "ENEMY_PROBABILITY": 0.5,

        # Training setup
        "SEED": 42,
        "BATCH": 32,
        "EPOCHS": 1000,
        "START_EPOCH": 0,

        # Directory structure
        "ROOT": ROOT,
        "WIP": ROOT / "WIP",
        "DATASET_DIR": ROOT / "WIP" /"dataset_10k",
        "CHECKPOINT_DIR": ROOT / "WIP" / "checkpoints",
        "LOG_DIR": ROOT / "WIP" / "logs",
 
        # Training settings
        "LOAD_WEIGHTS_ON_START": True,
        "MODEL_A_WEIGHTS_IN": "checkpoints\\combined_ab\\model_a_latest.weights.h5",
        "MODEL_B_WEIGHTS_IN": "checkpoints\\combined_ab\\model_b_latest.weights.h5",
        "SAVE_WEIGHTS_EVERY": 100,
        "SAVE_WEIGHTS_AT_END": True,
        "LOG_FLUSH_EVERY_EPOCHS": 10,

        # Verbosity settings
        "VERBOSITY": 1, # 0 = no print, 1 = epoch summary, 2 = epoch summary + losses, 3 = detailed info including weight norms, gradient norms, etc.
        
        # LOG settings
        "LOG_NAME": None, # if None, will be auto-generated with timestamp
    }
rollback_cfg = {
        "ENABLE_ROLLBACK": False,
        "LOOKBACK": 5,
        "DROP_ABS": 0.02,
        "MIN_EPOCH": 50,
        "COOLDOWN_EPOCHS": 10,
    }

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
    
    return PyrInternalModelA(
        config["LAYOUT"],
        sr_mode="replay",
        p_high=config["P_HIGH"],
        beta=config["MODEL_BETA"],
        alpha=config["ALPHA_FOR_PR_LAYERS"], # scale this to |logit| = 10.0
        seed=config["SEED"], # not used in replay mode
    )


def build_model_b(config: dict[str, Any]) -> tf.keras.Model:
    
    return PyrInternalModelB(
        config["LAYOUT"],
        sr_mode="replay",
        p_high=config["P_HIGH"],
        beta=config["MODEL_BETA"],
        alpha=config["ALPHA_FOR_PR_LAYERS"], # scale this to |logit| = 10.0
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

def evaluate_early_stop(epoch_metrics: dict, cfg: dict) -> bool:
    """
    Evaluate early stop rule for a single epoch.

    OR across rule dicts
    AND within each rule dict
    """

    rules = cfg["rules"]

    for rule in rules:  # OR across rules
        rule_ok = True

        for metric, cond in rule.items():  # AND within rule
            val = epoch_metrics.get(metric)

            if val is None:
                rule_ok = False
                break

            ref = cond["value"]
            cmp = cond["comparator"]

            if cmp == "gt":
                ok = val > ref
            elif cmp == "ge":
                ok = val >= ref
            elif cmp == "lt":
                ok = val < ref
            elif cmp == "le":
                ok = val <= ref
            else:
                raise ValueError(f"Unknown comparator: {cmp}")

            if not ok:
                rule_ok = False
                break

        if rule_ok:
            return True

    return False

def _best_recent_epoch(log_entries: list[dict[str, Any]], *, metric: str, lookback: int) -> dict[str, Any] | None:
    if not log_entries:
        return None
    recent = log_entries[-int(lookback):]
    valid = [e for e in recent if metric in e]
    if not valid:
        return None
    return max(valid, key=lambda e: float(e[metric]))

def print_verbosity(message: str, level: int = 1, config: dict[str, Any] = training_cfg) -> None:
    """Print message if verbosity level is sufficient."""
    if config.get("VERBOSITY", 0) >= level:
        print(message, flush=True)

def save_model_weights(model_a: tf.keras.Model, model_b: tf.keras.Model, checkpoint_dir: Path, tag: str) -> tuple[Path, Path]:
    path_a, path_b =save_ab_weights(model_a = model_a, model_b = model_b, base_dir = checkpoint_dir, tag = tag)
    if path_a is not None and path_b is not None:
        print_verbosity(f"[weights] saved: {path_a.name}, {path_b.name}", level=1, config=training_cfg)
    else:
        print_verbosity(f"[weights] save failed ({tag})", level=1, config=training_cfg)
    return path_a, path_b

def load_model_weights(
    model_a: tf.keras.Model,
    model_b: tf.keras.Model,
    a_path: Path | None,
    b_path: Path | None,
) -> bool:
    # Suppress helper-level prints so verbosity is controlled only here.
    with redirect_stdout(io.StringIO()):
        success = load_ab_weights(model_a, model_b, a_path, b_path)

    if success:
        a_name = Path(a_path).name if a_path is not None else "<none>"
        b_name = Path(b_path).name if b_path is not None else "<none>"
        print_verbosity(f"[weights] loaded: {a_name}, {b_name}", level=1, config=training_cfg)
    else:
        print_verbosity("[weights] load failed", level=1, config=training_cfg)
    return success

def _should_rollback(
    run_log: dict[str, Any],
    *,
    current_epoch: int,
    lookback: int,
    metric: str,
    drop_abs: float,
    min_epoch: int,
    cooldown_epochs: int,
    last_rollback_epoch: int,
) -> dict[str, Any]:
    epochs = list(run_log.get("epochs", []))
    if current_epoch < min_epoch:
        return {"action": "none", "reason": "min_epoch"}
    if last_rollback_epoch >= 0 and (current_epoch - last_rollback_epoch) < cooldown_epochs:
        return {"action": "none", "reason": "cooldown"}
    if len(epochs) < max(2, lookback):
        return {"action": "none", "reason": "insufficient_history"}

    current = epochs[-1]
    if metric not in current:
        return {"action": "none", "reason": "metric_missing_current"}

    best_recent = _best_recent_epoch(epochs[:-1], metric=metric, lookback=lookback)
    if best_recent is None:
        return {"action": "none", "reason": "metric_missing_recent"}

    current_val = current[metric]
    best_val = best_recent[metric]
    drop = best_val - current_val
    if drop >= drop_abs:
        return {
            "action": "rollback_best",
            "reason": f"{metric}_drop",
            "metric": metric,
            "current": current_val,
            "best_recent": best_val,
            "best_recent_epoch": best_recent["epoch"],
            "drop": drop,
        }
    return {"action": "none", "reason": "within_threshold"}

def _transform_tensor_for_b(
    x: tf.Tensor,
    *,
    stop_gradient: tf.Tensor,   # bool scalar tensor
    hard_mode: tf.Tensor,       # int scalar tensor: 0=none, 1=hard, 2=interp
    beta: tf.Tensor,            # float scalar tensor
    lam: tf.Tensor,             # float scalar tensor
    noise_std: tf.Tensor,       # float scalar tensor
) -> tf.Tensor:
    """
    TF-safe A->B transform for a single tensor.

    hard_mode:
      0 -> none
      1 -> hard
      2 -> interp
    """
    x = tf.cast(x, tf.float32)
    beta = tf.cast(beta, tf.float32)
    lam = tf.cast(lam, tf.float32)
    noise_std = tf.cast(noise_std, tf.float32)
    stop_gradient = tf.cast(stop_gradient, tf.bool)
    hard_mode = tf.cast(hard_mode, tf.int32)

    x_base = tf.cond(
        stop_gradient,
        lambda: tf.stop_gradient(x),
        lambda: x,
    )

    x_hard = harden_ste(x_base, beta=beta)

    y = tf.case(
        [
            (tf.equal(hard_mode, 0), lambda: x_base),
            (tf.equal(hard_mode, 1), lambda: x_hard),
            (tf.equal(hard_mode, 2), lambda: (1.0 - lam) * x_base + lam * x_hard),
        ],
        default=lambda: x_base,
        exclusive=True,
    )
    y = tf.cond(
        noise_std > 0.0,
        lambda: add_logit_noise_tf(y, noise_std, training=True),
        lambda: y,
    )
    return y


def _transform_list_for_b(
    xs: list[tf.Tensor],
    *,
    stop_gradient: tf.Tensor,
    hard_mode: tf.Tensor,
    beta: tf.Tensor,
    lam: tf.Tensor,
    noise_std: tf.Tensor,
) -> list[tf.Tensor]:
    return [
        _transform_tensor_for_b(
            x,
            stop_gradient=stop_gradient,
            hard_mode=hard_mode,
            beta=beta,
            lam=lam,
            noise_std=noise_std,
        )
        for x in xs
    ]


def _mean_over_scalars(values: list[tf.Tensor]) -> tf.Tensor:
    if not values:
        return tf.constant(0.0, dtype=tf.float32)
    return tf.add_n(values) / tf.cast(len(values), tf.float32)

def make_train_steps(depth: int):
    depth = int(depth)

    def _core_forward(
        batch: Any,                      # raw batch
        model_a: tf.keras.Model,
        model_b: tf.keras.Model,

        # whether B runs in training mode
        b_training: tf.Tensor,          # bool scalar tensor

        # interface config: comm
        comm_stop_gradient: tf.Tensor,   # bool scalar tensor
        comm_hard_mode: tf.Tensor,       # int scalar tensor: 0=none, 1=hard, 2=interp
        comm_beta: tf.Tensor,            # scalar tensor
        comm_lambda: tf.Tensor,          # scalar tensor
        comm_noise_std: tf.Tensor,       # scalar tensor

        # interface config: meas
        meas_stop_gradient: tf.Tensor,   # bool scalar tensor
        meas_hard_mode: tf.Tensor,       # int scalar tensor: 0=none, 1=hard, 2=interp
        meas_beta: tf.Tensor,            # scalar tensor
        meas_lambda: tf.Tensor,          # scalar tensor
        meas_noise_std: tf.Tensor,       # scalar tensor

        # interface config: out
        out_stop_gradient: tf.Tensor,    # bool scalar tensor
        out_hard_mode: tf.Tensor,        # int scalar tensor: 0=none, 1=hard, 2=interp
        out_beta: tf.Tensor,             # scalar tensor
        out_lambda: tf.Tensor,           # scalar tensor
        out_noise_std: tf.Tensor,        # scalar tensor
    ) -> tuple:
        (
            field_logits,
            gun_logits,
            shoot_tgt_logits,
            meas_a_tgt_logits_list,
            meas_out_a_tgt_list,
            comms_tgt_logits_list,
        ) = _unpack_batch(batch)

        # A forward
        comm_a_logits, meas_a_logits_list, out_a_logits_list = model_a.compute_with_internal(
            field_logits=field_logits,
            replay_out_a_logits_list=meas_out_a_tgt_list,
            harden_between_levels=False,
            training=True,
        )

        # A -> B interface transforms
        comm_logits_for_b = _transform_tensor_for_b(
            comm_a_logits,
            stop_gradient=comm_stop_gradient,
            hard_mode=comm_hard_mode,
            beta=comm_beta,
            lam=comm_lambda,
            noise_std=comm_noise_std,
        )

        meas_list_for_b = _transform_list_for_b(
            list(meas_a_logits_list),
            stop_gradient=meas_stop_gradient,
            hard_mode=meas_hard_mode,
            beta=meas_beta,
            lam=meas_lambda,
            noise_std=meas_noise_std,
        )

        out_list_for_b = _transform_list_for_b(
            list(out_a_logits_list),
            stop_gradient=out_stop_gradient,
            hard_mode=out_hard_mode,
            beta=out_beta,
            lam=out_lambda,
            noise_std=out_noise_std,
        )

        # B forward
        shoot_logit, meas_b_logits_list, out_b_logits_list, comms_b_logits_list, gun_logits_list = model_b.compute_with_internal(
            gun_logits,
            comm_logits_for_b,
            list(meas_list_for_b),
            list(out_list_for_b),
            harden_between_levels=False,
            training=False, # Not wired in model_b.compute_with_internal
        )

        return (
            comm_a_logits, # comm_a logits from model A
            meas_a_logits_list, # meas_a logits from model A
            comms_b_logits_list, # comms_b logits from model B, including comms_b at all levels (0, 1, ..., depth-1)
            meas_b_logits_list, # meas_b logits from model B, including meas_b at all levels (0, 1, ..., depth-1)
            gun_logits_list, # gun logits from model B, including gun logits at all levels (0, 1, ..., depth-1)
            shoot_logit, # shoot logit from model B
            meas_a_tgt_logits_list, # meas_a teacher target logits for all levels
            comms_tgt_logits_list, # comms teacher target logits for all levels. Element [0] is teacher target for comm_a (output of model A), elements [1:] are teacher targets for comms_b at levels 1, ..., depth-1. Element [-1] is teacher target for shoot (output of model B)
            shoot_tgt_logits, # shoot teacher target logit
        )

    def _losses(
        comm_a_logits: tf.Tensor, # comm_a logits from model A
        meas_a_logits_list: list[tf.Tensor], # meas_a logits from model A
        comms_b_logits_list: list[tf.Tensor], # comms_b logits from model B, including comms_b at all levels (0, 1, ..., depth-1)
        meas_b_logits_list: list[tf.Tensor], # meas_b logits from model B, including meas_b at all levels (0, 1, ..., depth-1)
        gun_logits_list: list[tf.Tensor], # gun logits from model B, including gun logits at all levels (0, 1, ..., depth-1)
        shoot_logit: tf.Tensor, # shoot logit from model B
        meas_a_tgt_logits_list: list[tf.Tensor], # meas_a teacher target logits for all levels
        comms_tgt_logits_list: list[tf.Tensor], # comms teacher target logits for all levels
                                                # Element [0] is teacher target for comm_a (output of model A), elements [1:] are teacher targets for comms_b at levels 1, ..., depth-1
                                                # Element [-1] is teacher target for shoot (output of model B)
        shoot_tgt_logits: tf.Tensor, # shoot teacher target logit
        w_comm_a_global: tf.Tensor,
        w_meas_a_global: tf.Tensor,
        w_meas_a_per_level: tf.Tensor, 
        w_comms_b_global: tf.Tensor, 
        w_comms_b_per_level: tf.Tensor,
        w_shoot: tf.Tensor,
        w_comm_a_mag: tf.Tensor,
        w_comms_b_mag: tf.Tensor,
        w_meas_a_mag: tf.Tensor,
        w_meas_b_mag: tf.Tensor,
        beta_target: tf.Tensor, # for magnitude losses, target is to bring |logit| towards this value
    ) -> dict[str, tf.Tensor | list[tf.Tensor]]:

        # === comm_a is the output of model A
        comm_a_tgt_bits = tf.cast(comms_tgt_logits_list[0] >= 0.0, tf.float32)
        comm_a_loss_raw = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=comm_a_tgt_bits, logits=comm_a_logits)
        )
        comm_a_loss = w_comm_a_global * comm_a_loss_raw
        comm_a_mag_loss_raw = magnitude_target_loss(comm_a_logits, beta_target=beta_target)
        comm_a_mag_loss = w_comm_a_mag * comm_a_mag_loss_raw
        comm_a_mean_abs = tf.reduce_mean(tf.abs(tf.cast(comm_a_logits, tf.float32)))

        # === shoot is the shoot output of model B
        shoot_tgt_bits = tf.cast(shoot_tgt_logits >= 0.0, tf.float32)
        shoot_loss_raw = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=shoot_tgt_bits, logits=shoot_logit)
        )
        shoot_loss = w_shoot * shoot_loss_raw
        shoot_acc = tf.reduce_mean(
            tf.cast(tf.equal(shoot_tgt_bits, tf.cast(shoot_logit >= 0.0, tf.float32)), tf.float32)
        )

        # === comms_b are the internal comms of model B. 
        # Note that first comms is same as output of model A, and last comm is the same as shoot.
        # We focus on comms from b at indices 1, ... depth-1
        comms_logits_list_123 = list(comms_b_logits_list[1:depth])
        comms_tgt_list_123 = list(comms_tgt_logits_list[1:depth])
        comms_b_loss_raw, comms_b_per = weighted_per_level_bce(
            comms_tgt_list_123,
            comms_logits_list_123,
            w_comms_b_per_level,
        )
        comms_b_loss = w_comms_b_global * comms_b_loss_raw

        comms_b_mag_loss_terms = [magnitude_target_loss(t, beta_target=beta_target) for t in comms_b_logits_list]
        comms_b_mag_loss_raw = _mean_over_scalars(comms_b_mag_loss_terms)
        comms_b_mag_loss = w_comms_b_mag * comms_b_mag_loss_raw
        comms_b_mean_abs_terms = [tf.reduce_mean(tf.abs(tf.cast(t, tf.float32))) for t in comms_b_logits_list]
        comms_b_mean_abs = _mean_over_scalars(comms_b_mean_abs_terms)

        # === meas_a is the measurement input to sr layers inside model A
        meas_a_loss_raw, meas_a_per = weighted_per_level_bce(
            list(meas_a_tgt_logits_list),
            list(meas_a_logits_list),
            w_meas_a_per_level,
        )
        meas_a_loss = w_meas_a_global * meas_a_loss_raw

        meas_a_mag_loss_terms = [magnitude_target_loss(t, beta_target=beta_target) for t in meas_a_logits_list]
        meas_a_mag_loss_raw = _mean_over_scalars(meas_a_mag_loss_terms)
        meas_a_mag_loss = w_meas_a_mag * meas_a_mag_loss_raw
        meas_a_mean_abs_terms = [tf.reduce_mean(tf.abs(tf.cast(t, tf.float32))) for t in meas_a_logits_list]
        meas_a_mean_abs = _mean_over_scalars(meas_a_mean_abs_terms)

        # === meas_b is the measurement input to sr layers inside model B
        meas_b_mag_loss_terms = [magnitude_target_loss(t, beta_target=beta_target) for t in meas_b_logits_list]
        meas_b_mag_loss_raw = _mean_over_scalars(meas_b_mag_loss_terms)
        meas_b_mag_loss = w_meas_b_mag * meas_b_mag_loss_raw
        meas_b_mean_abs_terms = [tf.reduce_mean(tf.abs(tf.cast(t, tf.float32))) for t in meas_b_logits_list]
        meas_b_mean_abs = _mean_over_scalars(meas_b_mean_abs_terms)

        # === gun logits are the per level guns for model B, outcome of combine layers.
        gun_logits_mean_abs_terms = [tf.reduce_mean(tf.abs(tf.cast(t, tf.float32))) for t in gun_logits_list]
        gun_logits_mean_abs = _mean_over_scalars(gun_logits_mean_abs_terms)

        total_loss = comm_a_loss + meas_a_loss + comms_b_loss + shoot_loss + comm_a_mag_loss + comms_b_mag_loss + meas_a_mag_loss + meas_b_mag_loss
        
        return {
            "total_loss": total_loss,
            "shoot_acc": shoot_acc,
            "comm_a_loss": comm_a_loss, # comm_a loss vs teacher target. Model A
            "comm_a_mag_loss": comm_a_mag_loss, # comm_a magnitude loss vs beta target. Model A
            "meas_a_loss": meas_a_loss, # meas_a loss vs teacher target. Model A 
            "meas_a_mag_loss": meas_a_mag_loss, # meas_a magnitude loss vs beta target. Model A
            "comms_b_loss": comms_b_loss, # comms_b loss vs teacher target. Model B 
            "comms_b_mag_loss": comms_b_mag_loss, # comms_b magnitude loss vs beta target. Model B 
            "meas_b_mag_loss": meas_b_mag_loss, # meas_b magnitude loss vs beta target. Model B
            "shoot_loss": shoot_loss, # shoot loss vs teacher target. Model B
            "comm_a_mean_abs": comm_a_mean_abs, # mean abs of comm_a logits. Model A
            "meas_a_mean_abs": meas_a_mean_abs, # mean abs of meas_a logits. Model A
            "comms_b_mean_abs": comms_b_mean_abs, # mean abs of comms_b logits. Model B
            "meas_b_mean_abs": meas_b_mean_abs, # mean abs of meas_b logits. Model B
            "mean_abs_gun_logits": gun_logits_mean_abs, # mean abs of gun logits. Model B
            "meas_a_per": meas_a_per, # per level accuracy of meas_a logits. Model A
            "comms_b_per": comms_b_per, # per level accuracy of comms_b logits. Model B
            "comm_a_loss_raw": comm_a_loss_raw, # comm_a loss vs teacher target, without weight. Model A
            "comm_a_mag_loss_raw": comm_a_mag_loss_raw, # comm_a magnitude loss vs beta target, without weight. Model A
            "meas_a_loss_raw": meas_a_loss_raw, # meas_a loss vs teacher target, without weight. Model A
            "meas_a_mag_loss_raw": meas_a_mag_loss_raw, # meas_a magnitude loss vs beta target, without weight. Model A
            "comms_b_loss_raw": comms_b_loss_raw, # comms_b loss vs teacher target, without weight. Model B
            "comms_b_mag_loss_raw": comms_b_mag_loss_raw, # comms_b magnitude loss vs beta target, without weight. Model B
            "meas_b_mag_loss_raw": meas_b_mag_loss_raw, # meas_b magnitude loss vs beta target, without weight. Model B
            "shoot_loss_raw": shoot_loss_raw, # shoot loss vs teacher target, without weight. Model B
        }

    @tf.function(reduce_retracing=True)
    def train_step(
        batch: Any,
        model_a: tf.keras.Model,
        model_b: tf.keras.Model,
        opt_a: tf.keras.optimizers.Optimizer,
        opt_b: tf.keras.optimizers.Optimizer,
        learning_rate_a: tf.Tensor,
        learning_rate_b: tf.Tensor,
        update_a: bool,
        update_b: bool,

        w_comm_a_global: tf.Tensor,
        w_meas_a_global: tf.Tensor, 
        w_meas_a_per_level: tf.Tensor,
        w_comms_b_global: tf.Tensor,  
        w_comms_b_per_level: tf.Tensor,
        w_shoot: tf.Tensor,
        w_comm_a_mag: tf.Tensor,
        w_comms_b_mag: tf.Tensor,
        w_meas_a_mag: tf.Tensor,
        w_meas_b_mag: tf.Tensor,
        beta_target: tf.Tensor, # for magnitude losses, target is to bring |logit| towards this value
        
        # interface: comm
        comm_stop_gradient: tf.Tensor,
        comm_hard_mode: tf.Tensor,
        comm_beta: tf.Tensor,
        comm_lambda: tf.Tensor,
        comm_noise_std: tf.Tensor,
        # interface: meas
        meas_stop_gradient: tf.Tensor,
        meas_hard_mode: tf.Tensor,
        meas_beta: tf.Tensor,
        meas_lambda: tf.Tensor,
        meas_noise_std: tf.Tensor,
        # interface: out
        out_stop_gradient: tf.Tensor,
        out_hard_mode: tf.Tensor,
        out_beta: tf.Tensor,
        out_lambda: tf.Tensor,
        out_noise_std: tf.Tensor,
        ) -> dict[str, tf.Tensor | list[tf.Tensor]]:
            
        vars_to_watch = []
        if update_a:
            vars_to_watch += model_a.trainable_variables
        if update_b:
            vars_to_watch += model_b.trainable_variables

        with tf.GradientTape() as tape:
            vals = _core_forward(
                        batch = batch, 
                        model_a = model_a,
                        model_b = model_b,
                        b_training = False, # This variable is not wired to anything in the current implementation, but we set it to False to be explicit that B forward is in eval mode during A step. We may want to experiment with setting this to True during A step in the future.
                        comm_stop_gradient = comm_stop_gradient,
                        comm_hard_mode = comm_hard_mode,
                        comm_beta = comm_beta,
                        comm_lambda = comm_lambda,
                        comm_noise_std = comm_noise_std,         
                        meas_stop_gradient = meas_stop_gradient, 
                        meas_hard_mode = meas_hard_mode, 
                        meas_beta = meas_beta, 
                        meas_lambda = meas_lambda,                    
                        meas_noise_std = meas_noise_std, 
                        out_stop_gradient = out_stop_gradient, 
                        out_hard_mode = out_hard_mode, 
                        out_beta = out_beta,
                        out_lambda = out_lambda, 
                        out_noise_std = out_noise_std, 
                    )
            out  = _losses(
                        *vals,
                        w_comm_a_global = w_comm_a_global,
                        w_meas_a_global = w_meas_a_global,        
                        w_meas_a_per_level = w_meas_a_per_level,
                        w_comms_b_global = w_comms_b_global,
                        w_comms_b_per_level = w_comms_b_per_level,
                        w_shoot = w_shoot,
                        w_comm_a_mag = w_comm_a_mag,    
                        w_comms_b_mag = w_comms_b_mag, 
                        w_meas_a_mag = w_meas_a_mag,     
                        w_meas_b_mag = w_meas_b_mag,
                        beta_target = beta_target,     
                    )

        grads = tape.gradient(out["total_loss"], vars_to_watch)

        if update_a:
            n_a = len(model_a.trainable_variables)
            gv_a = [(g, v) for g, v in zip(grads[:n_a], model_a.trainable_variables) if g is not None]
            if gv_a:
                opt_a.learning_rate.assign(learning_rate_a) # workaround for TF bug where learning rate change doesn't take effect unless you do a dummy assign to it after the first step
                opt_a.apply_gradients(gv_a)
        if update_b:
            n_a = len(model_a.trainable_variables) if update_a else 0
            gv_b = [(g, v) for g, v in zip(grads[n_a:], model_b.trainable_variables) if g is not None]
            if gv_b:
                opt_b.learning_rate.assign(learning_rate_b) # workaround for TF bug where learning rate change doesn't take effect unless you do a dummy assign to it after the first step
                opt_b.apply_gradients(gv_b)

        return out
    
    return train_step



def train(model_cfg, stage_cfg, interface_cfg, loss_cfg, early_stop_cfg) -> dict[str, Any]:

    
    WIP = Path(training_cfg["WIP"])
    DEPTH = training_cfg["DEPTH"]
    CHECKPOINT_DIR = Path(training_cfg["CHECKPOINT_DIR"])
    DATASET_DIR = Path(training_cfg["DATASET_DIR"])
    LOG_DIR = Path(training_cfg["LOG_DIR"])
    EPOCHS = training_cfg["EPOCHS"]
    BATCH = training_cfg["BATCH"]
    SEED = training_cfg["SEED"]

    SAVE_WEIGHTS_EVERY = training_cfg["SAVE_WEIGHTS_EVERY"]
    SAVE_WEIGHTS_AT_END = training_cfg["SAVE_WEIGHTS_AT_END"]
    LOAD_WEIGHTS_ON_START = training_cfg["LOAD_WEIGHTS_ON_START"]
    MODEL_A_WEIGHTS_IN = training_cfg["MODEL_A_WEIGHTS_IN"]
    MODEL_B_WEIGHTS_IN = training_cfg["MODEL_B_WEIGHTS_IN"]
    LOG_FLUSH_EVERY_EPOCHS = training_cfg["LOG_FLUSH_EVERY_EPOCHS"]

    # Rollback and magnitude settings are required config keys.
    ROLLBACK_LOOKBACK = rollback_cfg["LOOKBACK"]
    ROLLBACK_DROP_ABS = rollback_cfg["DROP_ABS"]
    ROLLBACK_MIN_EPOCH = rollback_cfg["MIN_EPOCH"]
    ROLLBACK_COOLDOWN_EPOCHS = rollback_cfg["COOLDOWN_EPOCHS"]
    ENABLE_ROLLBACK = rollback_cfg["ENABLE_ROLLBACK"]

    teacher_dataset_cfg["DATASET_DIR"] = str(DATASET_DIR)
    teacher_dataset_cfg["SEED"] = SEED
    teacher_dataset_cfg["BATCH"] = BATCH

    LAYOUT = GameLayout(
        field_size=training_cfg["FIELD_SIZE"],
        comms_size=training_cfg["COMMS_SIZE"],
        number_of_games_in_tournament=1000,
        channel_noise=training_cfg["CHANNEL_NOISE"],
        enemy_probability=training_cfg["ENEMY_PROBABILITY"],
    )
    model_cfg["FIELD_SIZE"] = training_cfg["FIELD_SIZE"]
    model_cfg["COMMS_SIZE"] = training_cfg["COMMS_SIZE"]
    model_cfg["SEED"] = training_cfg["SEED"]
    model_cfg["LAYOUT"] = LAYOUT
    
    
    
    tf.random.set_seed(SEED)

    # === Load dataset and build training pipeline
    raw_ds = load_dataset(teacher_dataset_cfg)
    tfds_train = build_train_pipeline(raw_ds, teacher_dataset_cfg)
    tfds_train = tfds_train.prefetch(tf.data.AUTOTUNE)
    print_verbosity(f"Dataset loaded and training pipeline built: {teacher_dataset_cfg['DATASET_DIR']}", level=1, config=training_cfg)

    # === Build models, optimizers, and do a warmup pass to build variables
    model_a = build_model_a(model_cfg)
    model_b = build_model_b(model_cfg)
    opt_a = tf.keras.optimizers.Adam(learning_rate=stage_cfg["lr_a"])
    opt_b = tf.keras.optimizers.Adam(learning_rate=stage_cfg["lr_b"])

    first_batch = next(iter(tfds_train))
    _force_build_models(model_a, model_b, first_batch)
    opt_a.build(model_a.trainable_variables)
    opt_b.build(model_b.trainable_variables)
    msg = f"[Model A+B] field_size={training_cfg['FIELD_SIZE']}, comms_size={training_cfg['COMMS_SIZE']}, p_high={model_cfg['P_HIGH']}, beta_input={model_cfg['MODEL_BETA']}, seed={training_cfg['SEED']}"
    print_verbosity(msg, level=1, config=training_cfg)
    
    run_load_event = {"start_mode": "fresh", "loaded_from": None}
    if LOAD_WEIGHTS_ON_START:
        if MODEL_A_WEIGHTS_IN:
            a_in = WIP / MODEL_A_WEIGHTS_IN
        else:
            a_in = _wfile(CHECKPOINT_DIR, "model_a", "latest")
        if MODEL_B_WEIGHTS_IN:
            b_in = WIP / MODEL_B_WEIGHTS_IN
        else:
            b_in = _wfile(CHECKPOINT_DIR, "model_b", "latest")

        loaded = load_model_weights(model_a, model_b, a_in, b_in)
        used_a, used_b = (a_in, b_in) if loaded else (None, None)

        if not loaded:
            a_auto, b_auto = _latest_epoch_pair(CHECKPOINT_DIR)
            loaded = load_model_weights(model_a, model_b, a_auto, b_auto)
            if loaded:
                used_a, used_b = a_auto, b_auto

        if loaded:
            run_load_event = {
                "start_mode": "loaded",
                "loaded_from": {
                    "model_a_path": str(used_a),
                    "model_b_path": str(used_b),
                    "model_a_mtime": _iso_mtime(used_a),
                    "model_b_mtime": _iso_mtime(used_b),
                },
            }
    # TODO: Add option to pass run_log and run_log_path from config and continue, also
    # add to output so we can chain consequetive runs together in a more flexible way.
    if training_cfg["LOG_NAME"] is not None:
        run_log, run_log_path = init_run_log(LOG_DIR, run_load_event, log_name=training_cfg["LOG_NAME"])
    else:
        run_log, run_log_path = init_run_log(LOG_DIR, run_load_event)
    print_verbosity(f"[log] run file: {run_log_path}", level=1, config=training_cfg)

    train_step = make_train_steps(depth=DEPTH)

    epoch = training_cfg["START_EPOCH"]
    early_stop_count = 0
          
    interrupted = False
    
    # === Initialize metrics for 
    best_shoot_acc = float("-inf")
    last_rollback_epoch = -10**9
    best_stage_cfg = deepcopy(stage_cfg)
    best_model_cfg = deepcopy(model_cfg)
    best_interface_cfg = deepcopy(interface_cfg)
    best_loss_cfg = deepcopy(loss_cfg)
    best_early_stop_cfg = deepcopy(early_stop_cfg)

    try:
        print_verbosity(f"Enable rollback: {ENABLE_ROLLBACK}", level=1, config=training_cfg)
        while epoch < EPOCHS:
            
            msg = f"\n[epoch {epoch}] Training mode: Model A {stage_cfg['update_a']}, Model B {stage_cfg['update_b']}"
            print_verbosity(msg , level=1, config=training_cfg)

            m_total = tf.keras.metrics.Mean()
            m_shoot_acc = tf.keras.metrics.Mean()
            m_comm_a = tf.keras.metrics.Mean()
            m_meas_in_a = tf.keras.metrics.Mean()
            m_comms_b = tf.keras.metrics.Mean()
            m_shoot_loss = tf.keras.metrics.Mean()
            m_comm_mag = tf.keras.metrics.Mean()
            m_comm_mag_raw = tf.keras.metrics.Mean()
            m_mean_abs_comm = tf.keras.metrics.Mean()
            m_comms_b_mag = tf.keras.metrics.Mean()
            m_comms_b_mag_raw = tf.keras.metrics.Mean()
            m_comms_b_mean_abs = tf.keras.metrics.Mean()
            m_meas_mag = tf.keras.metrics.Mean()
            m_meas_mag_raw = tf.keras.metrics.Mean()
            m_mean_abs_meas_in_a = tf.keras.metrics.Mean()
            m_meas_b_mag = tf.keras.metrics.Mean()
            m_meas_b_mag_raw = tf.keras.metrics.Mean()
            m_mean_abs_meas_b = tf.keras.metrics.Mean()
            m_mean_abs_gun_logits = tf.keras.metrics.Mean()
            m_meas_in_a_per = [tf.keras.metrics.Mean() for _ in range(DEPTH)]
            m_comms_b_per = [tf.keras.metrics.Mean() for _ in range(max(DEPTH - 1, 0))]

            _lr_a = tf.constant(stage_cfg["lr_a"], dtype=tf.float32)
            _lr_b = tf.constant(stage_cfg["lr_b"], dtype=tf.float32)
            _update_a = stage_cfg["update_a"]
            _update_b = stage_cfg["update_b"]
            _w_comm_a_global = tf.constant(loss_cfg["w_comm_a_global"], dtype=tf.float32)
            _w_meas_a_global = tf.constant(loss_cfg["w_meas_a_global"], dtype=tf.float32)
            _w_meas_a_per_level = tf.constant(loss_cfg["w_meas_a_per_level"], dtype=tf.float32)
            _w_comms_b_global = tf.constant(loss_cfg["w_comms_b_global"], dtype=tf.float32)
            _w_comms_b_per_level = tf.constant(loss_cfg["w_comms_b_per_level"], dtype=tf.float32)
            _w_shoot = tf.constant(loss_cfg["w_shoot"], dtype=tf.float32)
            _w_comm_a_mag = tf.constant(loss_cfg["w_comm_a_mag"], dtype=tf.float32)
            _w_comms_b_mag = tf.constant(loss_cfg["w_comms_b_mag"], dtype=tf.float32)
            _w_meas_a_mag = tf.constant(loss_cfg["w_meas_a_mag"], dtype=tf.float32)
            _w_meas_b_mag = tf.constant(loss_cfg["w_meas_b_mag"], dtype=tf.float32)
            _beta_target = tf.constant(loss_cfg["mag_target"], dtype=tf.float32)        
            _comm_stop_gradient = tf.constant(interface_cfg["comm"]["stop_gradient"])
            _comm_hard_mode = tf.constant(interface_cfg["comm"]["hardening"]["mode"], dtype=tf.int32)
            _comm_beta = tf.constant(interface_cfg["comm"]["hardening"]["beta"], dtype=tf.float32)
            _comm_lambda = tf.constant(interface_cfg["comm"]["hardening"]["lambda"], dtype=tf.float32)
            _comm_noise_std = tf.constant(interface_cfg["comm"]["noise_std"], dtype=tf.float32)
            _meas_stop_gradient = tf.constant(interface_cfg["meas"]["stop_gradient"])           
            _meas_hard_mode = tf.constant(interface_cfg["meas"]["hardening"]["mode"], dtype=tf.int32)
            _meas_beta = tf.constant(interface_cfg["meas"]["hardening"]["beta"], dtype=tf.float32)
            _meas_lambda = tf.constant(interface_cfg["meas"]["hardening"]["lambda"], dtype=tf.float32)
            _meas_noise_std = tf.constant(interface_cfg["meas"]["noise_std"], dtype=tf.float32)
            _out_stop_gradient = tf.constant(interface_cfg["out"]["stop_gradient"], dtype=tf.bool)
            _out_hard_mode = tf.constant(interface_cfg["out"]["hardening"]["mode"], dtype=tf.int32)
            _out_beta = tf.constant(interface_cfg["out"]["hardening"]["beta"], dtype=tf.float32)
            _out_lambda = tf.constant(interface_cfg["out"]["hardening"]["lambda"], dtype=tf.float32)
            _out_noise_std = tf.constant(interface_cfg["out"]["noise_std"], dtype=tf.float32)

            for batch in tfds_train:
                out = train_step(
                    batch = batch,
                    model_a = model_a,
                    model_b = model_b,
                    opt_a = opt_a,
                    opt_b = opt_b,
                    learning_rate_a = _lr_a,
                    learning_rate_b = _lr_b,
                    update_a = _update_a,
                    update_b = _update_b,
                    w_comm_a_global = _w_comm_a_global,
                    w_meas_a_global = _w_meas_a_global,
                    w_meas_a_per_level = _w_meas_a_per_level,
                    w_comms_b_global = _w_comms_b_global,
                    w_comms_b_per_level = _w_comms_b_per_level,
                    w_shoot = _w_shoot,
                    w_comm_a_mag = _w_comm_a_mag,
                    w_comms_b_mag = _w_comms_b_mag,
                    w_meas_a_mag = _w_meas_a_mag,
                    w_meas_b_mag = _w_meas_b_mag,
                    beta_target = _beta_target,
                    # interface: comm
                    comm_stop_gradient = _comm_stop_gradient,
                    comm_hard_mode = _comm_hard_mode,
                    comm_beta = _comm_beta,
                    comm_lambda = _comm_lambda,
                    comm_noise_std = _comm_noise_std,
                    # interface: meas
                    meas_stop_gradient = _meas_stop_gradient,
                    meas_hard_mode = _meas_hard_mode,
                    meas_beta = _meas_beta,
                    meas_lambda = _meas_lambda,
                    meas_noise_std = _meas_noise_std,
                    # interface: out
                    out_stop_gradient = _out_stop_gradient,
                    out_hard_mode = _out_hard_mode,
                    out_beta = _out_beta,
                    out_lambda = _out_lambda,
                    out_noise_std = _out_noise_std,
                 
                )
 
                m_total.update_state(out["total_loss"])
                m_shoot_acc.update_state(out["shoot_acc"])
                m_comm_a.update_state(out["comm_a_loss"])
                m_meas_in_a.update_state(out["meas_a_loss"])
                m_comms_b.update_state(out["comms_b_loss"])
                m_shoot_loss.update_state(out["shoot_loss"])
                m_comm_mag.update_state(out["comm_a_mag_loss"])
                m_comm_mag_raw.update_state(out["comm_a_mag_loss_raw"])
                m_mean_abs_comm.update_state(out["comm_a_mean_abs"])
                m_comms_b_mag.update_state(out["comms_b_mag_loss"])
                m_comms_b_mag_raw.update_state(out["comms_b_mag_loss_raw"])
                m_comms_b_mean_abs.update_state(out["comms_b_mean_abs"])
                m_meas_mag.update_state(out["meas_a_mag_loss"])
                m_meas_mag_raw.update_state(out["meas_a_mag_loss_raw"])
                m_mean_abs_meas_in_a.update_state(out["meas_a_mean_abs"])
                m_meas_b_mag.update_state(out["meas_b_mag_loss"])
                m_meas_b_mag_raw.update_state(out["meas_b_mag_loss_raw"])
                m_mean_abs_meas_b.update_state(out["meas_b_mean_abs"])
                m_mean_abs_gun_logits.update_state(out["mean_abs_gun_logits"])

                for i, t in enumerate(out["meas_a_per"]):
                    if i < len(m_meas_in_a_per):
                        m_meas_in_a_per[i].update_state(t)
                for i, t in enumerate(out["comms_b_per"]):
                    if i < len(m_comms_b_per):
                        m_comms_b_per[i].update_state(t)

            epoch_metrics: dict[str, Any] = {
                "epoch": epoch,
                "shoot_acc": float(m_shoot_acc.result().numpy()),
                "total_loss": float(m_total.result().numpy()),
                "comm_a_loss": float(m_comm_a.result().numpy()),
                "meas_in_a_loss": float(m_meas_in_a.result().numpy()),
                "comms_b_loss": float(m_comms_b.result().numpy()),
                "shoot_loss": float(m_shoot_loss.result().numpy()),
                "comm_mag_loss": float(m_comm_mag.result().numpy()),
                "comm_mag_loss_raw": float(m_comm_mag_raw.result().numpy()),
                "mean_abs_comm": float(m_mean_abs_comm.result().numpy()),
                "comms_b_mag_loss": float(m_comms_b_mag.result().numpy()),
                "comms_b_mag_loss_raw": float(m_comms_b_mag_raw.result().numpy()),
                "mean_abs_comms_b": float(m_comms_b_mean_abs.result().numpy()),
                "meas_mag_loss": float(m_meas_mag.result().numpy()),
                "meas_mag_loss_raw": float(m_meas_mag_raw.result().numpy()),
                "mean_abs_meas_in_a": float(m_mean_abs_meas_in_a.result().numpy()),
                "meas_b_mag_loss": float(m_meas_b_mag.result().numpy()),
                "meas_b_mag_loss_raw": float(m_meas_b_mag_raw.result().numpy()),
                "mean_abs_meas_b": float(m_mean_abs_meas_b.result().numpy()),
                "mean_abs_gun_logits": float(m_mean_abs_gun_logits.result().numpy()),
                "comm_mag_beta_target": loss_cfg["mag_target"],
                "training_model_A": stage_cfg["update_a"],
                "training_model_B": stage_cfg["update_b"],
                "lr_a*1000": float(opt_a.learning_rate.numpy())*1000.0,
                "lr_b*1000": float(opt_b.learning_rate.numpy())*1000.0,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            for i, mm in enumerate(m_meas_in_a_per):
                epoch_metrics[f"meas_in_a_per_{i}"] = float(mm.result().numpy())
            for i, mm in enumerate(m_comms_b_per):
                epoch_metrics[f"comms_b_per_{i}"] = float(mm.result().numpy())

            
            msg =  f"epoch {epoch:03d}  "
            msg += "  ".join([f"{k}={v:.4f}" for k, v in epoch_metrics.items() if isinstance(v, float)])
            msg += "\n"
            print_verbosity(msg, level=2, config=training_cfg)

            append_epoch_log(run_log, epoch_metrics)

            # === Store 'best' for roll back
            if epoch_metrics["shoot_acc"] > best_shoot_acc:
                best_shoot_acc = float(epoch_metrics["shoot_acc"])
                save_model_weights(model_a, model_b, CHECKPOINT_DIR, tag="best")
                best_stage_cfg = deepcopy(stage_cfg)
                best_model_cfg = deepcopy(model_cfg)
                best_interface_cfg = deepcopy(interface_cfg)
                best_loss_cfg = deepcopy(loss_cfg)
                best_early_stop_cfg = deepcopy(early_stop_cfg)

            rollback_event = {"action": "none", "reason": "disabled"}
            if ENABLE_ROLLBACK:
                rollback_event = _should_rollback(
                    run_log,
                    current_epoch=epoch,
                    lookback=ROLLBACK_LOOKBACK,
                    metric="shoot_acc",
                    drop_abs=ROLLBACK_DROP_ABS,
                    min_epoch=ROLLBACK_MIN_EPOCH,
                    cooldown_epochs=ROLLBACK_COOLDOWN_EPOCHS,
                    last_rollback_epoch=last_rollback_epoch,
                )

            if rollback_event["action"] == "rollback_best":
                best_a = _wfile(CHECKPOINT_DIR, "model_a", "best")
                best_b = _wfile(CHECKPOINT_DIR, "model_b", "best")
                rollback_loaded = load_model_weights(model_a, model_b, best_a, best_b)
                if rollback_loaded:
                    last_rollback_epoch = epoch
                    stage_cfg = best_stage_cfg
                    model_cfg = best_model_cfg
                    interface_cfg = best_interface_cfg
                    loss_cfg = best_loss_cfg
                    early_stop_cfg = best_early_stop_cfg
                    early_stop_count = 0
                    
                    if stage_cfg["update_a"] and not stage_cfg["update_b"]:
                        rollback_stage = "A"
                    elif stage_cfg["update_b"] and not stage_cfg["update_a"]:
                        rollback_stage = "B"
                    else:
                        rollback_stage = "A+B"
                    rollback_event["rollback_loaded"] = True
                    rollback_event["rollback_stage_mode"] = rollback_stage
                    rollback_event["rollback_lr_a"] = stage_cfg["lr_a"]
                    rollback_event["rollback_lr_b"] = stage_cfg["lr_b"]
                    opt_a.build(model_a.trainable_variables)
                    opt_b.build(model_b.trainable_variables)
                    print_verbosity(
                        "[rollback] restored best checkpoint and reduced stage learning rates "
                        f"to mode = {rollback_event['rollback_stage_mode']}, lr_a={rollback_event['rollback_lr_a']:.6g}, lr_b={rollback_event['rollback_lr_b']:.6g}"
                    , level=1, config=training_cfg)
                else:
                    rollback_event["rollback_loaded"] = False
                    print_verbosity("[rollback] requested but best checkpoint files were not available.", level=1, config=training_cfg)

            run_log.setdefault("events", []).append({
                "epoch": int(epoch),
                "type": "rollback_check",
                **rollback_event,
            })

            if LOG_FLUSH_EVERY_EPOCHS and ((epoch + 1) % LOG_FLUSH_EVERY_EPOCHS == 0):
                flush_run_log(run_log, run_log_path)
                print_verbosity(f"[log] flushed -> {run_log_path}", level=1, config=training_cfg)
            if SAVE_WEIGHTS_EVERY and ((epoch + 1) % SAVE_WEIGHTS_EVERY == 0):
                save_model_weights(model_a, model_b, CHECKPOINT_DIR, tag=f"epoch_{epoch + 1:04d}")
                save_model_weights(model_a, model_b, CHECKPOINT_DIR, tag="latest")
                flush_run_log(run_log, run_log_path)
                print_verbosity(f"[log] flushed -> {run_log_path}", level=1, config=training_cfg)
            # Early stop functionality and counter increment
            if evaluate_early_stop(epoch_metrics, early_stop_cfg):
                early_stop_count += 1
            else:
                early_stop_count = 0

            if early_stop_count >= early_stop_cfg["consecutive"]:
                msg = f"Early stopping criteria met at epoch {epoch} "
                msg += f"for {early_stop_cfg['consecutive']} consecutive epoch(s)."
                print_verbosity(msg, level=1, config=training_cfg) 
                run_log.setdefault("events", []).append({
                    "epoch": int(epoch),
                    "type": "early_stop",
                    "patience": early_stop_cfg["consecutive"],
                })
                break
            
            # Epoch increment
            epoch += 1

            # ========== END OF EPOCH LOOP =========

    except KeyboardInterrupt:
        interrupted = True
        print_verbosity("[train] interrupted by user.", level=1, config=training_cfg)

    finally:
        flush_run_log(run_log, run_log_path)
        print_verbosity(f"[log] flushed -> {run_log_path}", level=1, config=training_cfg)

        if SAVE_WEIGHTS_AT_END:
            end_tag = f"epoch_{epoch:04d}_{'interrupted' if interrupted else 'final'}"
            save_model_weights(model_a, model_b, CHECKPOINT_DIR, tag=end_tag)
            save_model_weights(model_a, model_b, CHECKPOINT_DIR, tag="latest")

    return {
        "log_path": str(run_log_path),
        "epochs_recorded": len(run_log.get("epochs", [])),
        "interrupted": interrupted,
    }


if __name__ == "__main__":
    result = train(model_cfg, stage_cfg, interface_cfg, loss_cfg, early_stop_cfg)
    print_verbosity(result, level=1, config=training_cfg)