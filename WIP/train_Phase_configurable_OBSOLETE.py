


from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import sys
import os

# Logging / performance knobs (set BEFORE importing tensorflow)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# User requested oneDNN enabled.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

import tensorflow as tf

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


_ensure_repo_paths(get_config())
from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle_New.pyr_internal_model_b import PyrInternalModelB
from Q_Sea_Battle_New.pyr_internal_model_a import PyrInternalModelA


MODE_A = 0
MODE_B = 1
MODE_A_SHOOT = 2 # A_SHOOT is obsolete
MODE_AB = 3

MODE_NAME = {
    MODE_A: "A",
    MODE_B: "B",
    MODE_A_SHOOT: "A_SHOOT",
    MODE_AB: "AB",
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
    n2 = int(config["N2"])
    field_size = int(config.get("FIELD_SIZE", int(np.sqrt(n2))))
    comms_size = int(config.get("COMMS_SIZE", 1))
    p_high = float(config.get("P_HIGH", 1.0))
    beta_input = float(config["BETA_INPUT"])
    seed = int(config["SEED"])

    layout = GameLayout(
        field_size=field_size,
        comms_size=comms_size,
        number_of_games_in_tournament=1000,
        channel_noise=0.0,
        enemy_probability=0.5,
    )

    return PyrInternalModelA(
        layout,
        sr_mode="replay",
        p_high=p_high,
        beta=beta_input,
        alpha=5.0,
        seed=seed + 10,
    )



def build_model_b(config: dict[str, Any]) -> tf.keras.Model:
    n2 = int(config["N2"])
    field_size = int(config.get("FIELD_SIZE", int(np.sqrt(n2))))
    comms_size = int(config.get("COMMS_SIZE", 1))
    p_high = float(config.get("P_HIGH", 1.0))
    beta_input = float(config["BETA_INPUT"])

    layout = GameLayout(
        field_size=field_size,
        comms_size=comms_size,
        number_of_games_in_tournament=1000,
        channel_noise=0.0,
        enemy_probability=0.5,
    )

    return PyrInternalModelB(
        layout,
        sr_mode="replay",
        p_high=p_high,
        beta=beta_input,
        alpha=5.0,
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


class TrainState:
    """Mutable training state stored as tf.Variables outside the graph."""

    def __init__(self, depth: int, *, lr_a: float, lr_b: float, w_comm_mag: float, w_meas_mag: float):
        self.depth = int(depth)
        self.W_MEAS_IN_A = tf.Variable([0.0, 0.0, 0.0, 0.0], dtype=tf.float32, trainable=False)
        self.W_COMMS_A = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.W_COMMS_B_123 = tf.Variable([1.0, 0.0, 0.0], dtype=tf.float32, trainable=False)
        self.W_SHOOT = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.W_COMM_MAG = tf.Variable(float(w_comm_mag), dtype=tf.float32, trainable=False)
        self.W_MEAS_MAG = tf.Variable(float(w_meas_mag), dtype=tf.float32, trainable=False)
        self.TRAINING_MODE = tf.Variable(MODE_B, dtype=tf.int32, trainable=False)
        # Keep A_SHOOT hardening identical to your prior script.
        self.BETA_HARDEN = tf.Variable(1.0, dtype=tf.float32, trainable=False)
        self.LR_A = tf.Variable(float(lr_a), dtype=tf.float32, trainable=False)
        self.LR_B = tf.Variable(float(lr_b), dtype=tf.float32, trainable=False)


def _best_recent_epoch(log_entries: list[dict[str, Any]], *, metric: str, lookback: int) -> dict[str, Any] | None:
    if not log_entries:
        return None
    recent = log_entries[-int(lookback):]
    valid = [e for e in recent if metric in e]
    if not valid:
        return None
    return max(valid, key=lambda e: float(e[metric]))



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
    if current_epoch < int(min_epoch):
        return {"action": "none", "reason": "min_epoch"}
    if last_rollback_epoch >= 0 and (current_epoch - last_rollback_epoch) < int(cooldown_epochs):
        return {"action": "none", "reason": "cooldown"}
    if len(epochs) < max(2, int(lookback)):
        return {"action": "none", "reason": "insufficient_history"}

    current = epochs[-1]
    if metric not in current:
        return {"action": "none", "reason": "metric_missing_current"}

    best_recent = _best_recent_epoch(epochs[:-1], metric=metric, lookback=lookback)
    if best_recent is None:
        return {"action": "none", "reason": "metric_missing_recent"}

    current_val = float(current[metric])
    best_val = float(best_recent[metric])
    drop = best_val - current_val
    if drop >= float(drop_abs):
        return {
            "action": "rollback_best",
            "reason": f"{metric}_drop",
            "metric": metric,
            "current": current_val,
            "best_recent": best_val,
            "best_recent_epoch": int(best_recent.get("epoch", -1)),
            "drop": drop,
        }
    return {"action": "none", "reason": "within_threshold"}

def evaluate_early_stop(epoch_metrics: dict, cfg: dict) -> bool:
    """
    Evaluate early stop rule for a single epoch.

    OR across rule dicts
    AND within each rule dict
    """

    rules = cfg.get("rules", [])

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

    y = add_logit_noise_tf(y, noise_std, training=True)
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

def make_train_steps(depth: int):
    depth = int(depth)

    def _core_forward(
        batch: Any,                      # raw batch
        model_a: tf.keras.Model,
        model_b: tf.keras.Model,

        # whether B runs in training mode
        b_training: tf.Tensor,           # bool scalar tensor

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
            meas_in_a_tgt_list,
            meas_out_a_tgt_list,
            comms_tgt_list,
        ) = _unpack_batch(batch)

        # A forward
        comm_logits, meas_list, out_list = model_a.compute_with_internal(
            field_logits=field_logits,
            replay_out_a_logits_list=meas_out_a_tgt_list,
            harden_between_levels=False,
            training=True,
        )

        # A -> B interface transforms
        comm_logits_for_b = _transform_tensor_for_b(
            comm_logits,
            stop_gradient=comm_stop_gradient,
            hard_mode=comm_hard_mode,
            beta=comm_beta,
            lam=comm_lambda,
            noise_std=comm_noise_std,
        )

        meas_list_for_b = _transform_list_for_b(
            list(meas_list),
            stop_gradient=meas_stop_gradient,
            hard_mode=meas_hard_mode,
            beta=meas_beta,
            lam=meas_lambda,
            noise_std=meas_noise_std,
        )

        out_list_for_b = _transform_list_for_b(
            list(out_list),
            stop_gradient=out_stop_gradient,
            hard_mode=out_hard_mode,
            beta=out_beta,
            lam=out_lambda,
            noise_std=out_noise_std,
        )

        # B forward
        shoot_logit, _, _, comms_logits_list, _ = model_b.compute_with_internal(
            gun_logits,
            comm_logits_for_b,
            list(meas_list_for_b),
            list(out_list_for_b),
            harden_between_levels=False,
            training=tf.cast(b_training, tf.bool),
        )

        return (
            comm_logits,          # raw A comm output
            comm_logits_for_b,    # transformed comm seen by B
            meas_list,            # raw A meas outputs
            out_list,             # raw A out outputs
            shoot_logit,
            comms_logits_list,
            shoot_tgt_logits,
            meas_in_a_tgt_list,
            comms_tgt_list,
        )

    def _losses(
        comm_logits: tf.Tensor,
        comm_logits_for_b: tf.Tensor,
        meas_list: list[tf.Tensor],
        out_list: list[tf.Tensor],
        shoot_logit: tf.Tensor,
        comms_logits_list: list[tf.Tensor],
        shoot_tgt_logits: tf.Tensor,
        meas_in_a_tgt_list: list[tf.Tensor],
        comms_tgt_list: list[tf.Tensor],
        w_meas_in_a: tf.Tensor,
        w_comms_a: tf.Tensor,
        w_comms_b_123: tf.Tensor,
        w_shoot: tf.Tensor,
        w_comm_mag: tf.Tensor,
        w_meas_mag: tf.Tensor,
    ) -> dict[str, tf.Tensor | list[tf.Tensor]]:
        del out_list  # intentionally unused; replay passthrough only

        comm_a_bits = tf.cast(comms_tgt_list[0] >= 0.0, tf.float32)
        comm_a_loss = w_comms_a * tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=comm_a_bits, logits=comm_logits)
        )

        meas_in_a_loss, meas_in_a_per = weighted_per_level_bce(
            list(meas_in_a_tgt_list),
            list(meas_list),
            w_meas_in_a,
        )

        shoot_tgt_bits = tf.cast(shoot_tgt_logits >= 0.0, tf.float32)
        shoot_pred = tf.cast(shoot_logit, tf.float32)
        shoot_loss = w_shoot * tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=shoot_tgt_bits, logits=shoot_pred)
        )
        shoot_acc = tf.reduce_mean(
            tf.cast(tf.equal(shoot_tgt_bits, tf.cast(shoot_pred >= 0.0, tf.float32)), tf.float32)
        )

        comms_pred_list_123 = list(comms_logits_list[1:depth])
        comms_tgt_list_123 = list(comms_tgt_list[1:depth])
        comms_b_loss, comms_b_per = weighted_per_level_bce(
            comms_tgt_list_123,
            comms_pred_list_123,
            w_comms_b_123,
        )

        comm_mag_loss_raw = magnitude_target_loss(comm_logits, beta_target=10.0)
        comm_mag_loss = w_comm_mag * comm_mag_loss_raw
        mean_abs_comm_for_b = tf.reduce_mean(tf.abs(tf.cast(comm_logits_for_b, tf.float32)))

        meas_mag_loss_terms = [magnitude_target_loss(t, beta_target=1.0) for t in meas_list]
        meas_mag_loss_raw = tf.add_n(meas_mag_loss_terms) / tf.cast(len(meas_mag_loss_terms), tf.float32)
        meas_mag_loss = w_meas_mag * meas_mag_loss_raw
        mean_abs_meas_in_a_terms = [tf.reduce_mean(tf.abs(tf.cast(t, tf.float32))) for t in meas_list]
        mean_abs_meas_in_a = tf.add_n(mean_abs_meas_in_a_terms) / tf.cast(len(mean_abs_meas_in_a_terms), tf.float32)

        total = shoot_loss + comm_a_loss + meas_in_a_loss + comms_b_loss + comm_mag_loss + meas_mag_loss

        return {
            "total": total,
            "shoot_acc": shoot_acc,
            "comm_a_loss": comm_a_loss,
            "meas_in_a_loss": meas_in_a_loss,
            "meas_in_a_per": meas_in_a_per,
            "comms_b_loss": comms_b_loss,
            "comms_b_per": comms_b_per,
            "shoot_loss": shoot_loss,
            "comm_mag_loss": comm_mag_loss,
            "comm_mag_loss_raw": comm_mag_loss_raw,
            "mean_abs_comm_for_b": mean_abs_comm_for_b,
            "meas_mag_loss": meas_mag_loss,
            "meas_mag_loss_raw": meas_mag_loss_raw,
            "mean_abs_meas_in_a": mean_abs_meas_in_a,
        }

    @tf.function(reduce_retracing=True)
    def train_step_a(
        batch: Any,

        # models
        model_a: tf.keras.Model,
        model_b: tf.keras.Model,

        # optimizers
        opt_a: tf.keras.optimizers.Optimizer,
        opt_b: tf.keras.optimizers.Optimizer,

        # which model(s) get updated
        update_a: tf.Tensor,   # bool scalar
        update_b: tf.Tensor,   # bool scalar

        # whether B should run in training mode
        b_training: tf.Tensor,

        # loss weights
        w_meas_in_a: tf.Tensor,
        w_comms_a: tf.Tensor,
        w_comms_b_123: tf.Tensor,
        w_shoot: tf.Tensor,
        w_comm_mag: tf.Tensor,
        w_meas_mag: tf.Tensor,

        # interface config: comm
        comm_stop_gradient: tf.Tensor,
        comm_hard_mode: tf.Tensor,
        comm_beta: tf.Tensor,
        comm_lambda: tf.Tensor,
        comm_noise_std: tf.Tensor,

        # interface config: meas
        meas_stop_gradient: tf.Tensor,
        meas_hard_mode: tf.Tensor,
        meas_beta: tf.Tensor,
        meas_lambda: tf.Tensor,
        meas_noise_std: tf.Tensor,

        # interface config: out
        out_stop_gradient: tf.Tensor,
        out_hard_mode: tf.Tensor,
        out_beta: tf.Tensor,
        out_lambda: tf.Tensor,
        out_noise_std: tf.Tensor,
    ) -> dict[str, tf.Tensor | list[tf.Tensor]]:

        with tf.GradientTape() as tape:

            vals = _core_forward(
                batch=batch,
                model_a=model_a,
                model_b=model_b,
                b_training=b_training,

                comm_stop_gradient=comm_stop_gradient,
                comm_hard_mode=comm_hard_mode,
                comm_beta=comm_beta,
                comm_lambda=comm_lambda,
                comm_noise_std=comm_noise_std,

                meas_stop_gradient=meas_stop_gradient,
                meas_hard_mode=meas_hard_mode,
                meas_beta=meas_beta,
                meas_lambda=meas_lambda,
                meas_noise_std=meas_noise_std,

                out_stop_gradient=out_stop_gradient,
                out_hard_mode=out_hard_mode,
                out_beta=out_beta,
                out_lambda=out_lambda,
                out_noise_std=out_noise_std,
            )

            out = _losses(
                *vals,
                w_meas_in_a,
                w_comms_a,
                w_comms_b_123,
                w_shoot,
                w_comm_mag,
                w_meas_mag,
            )

        total_loss = out["total"]

        # ---- A update ----
        def _apply_a():
            vars_a = model_a.trainable_variables
            grads = tape.gradient(total_loss, vars_a)
            gv = [(g, v) for g, v in zip(grads, vars_a) if g is not None]
            if gv:
                opt_a.apply_gradients(gv)
            return 0

        # ---- B update ----
        def _apply_b():
            vars_b = model_b.trainable_variables
            grads = tape.gradient(total_loss, vars_b)
            gv = [(g, v) for g, v in zip(grads, vars_b) if g is not None]
            if gv:
                opt_b.apply_gradients(gv)
            return 0

        tf.cond(update_a, _apply_a, lambda: 0)
        tf.cond(update_b, _apply_b, lambda: 0)

        return out

    @tf.function(reduce_retracing=True)
    def train_step_b(
        batch: Any,

        # models
        model_a: tf.keras.Model,
        model_b: tf.keras.Model,

        # optimizers
        opt_a: tf.keras.optimizers.Optimizer,
        opt_b: tf.keras.optimizers.Optimizer,

        # which model(s) get updated
        update_a: tf.Tensor,   # bool scalar
        update_b: tf.Tensor,   # bool scalar

        # whether B should run in training mode
        b_training: tf.Tensor,  # bool scalar

        # loss weights
        w_meas_in_a: tf.Tensor,     # shape [DEPTH]
        w_comms_a: tf.Tensor,       # scalar
        w_comms_b_123: tf.Tensor,   # shape [DEPTH-1]
        w_shoot: tf.Tensor,         # scalar
        w_comm_mag: tf.Tensor,      # scalar
        w_meas_mag: tf.Tensor,      # scalar

        # interface config: comm
        comm_stop_gradient: tf.Tensor,   # bool scalar
        comm_hard_mode: tf.Tensor,       # int scalar: 0=none, 1=hard, 2=interp
        comm_beta: tf.Tensor,            # scalar
        comm_lambda: tf.Tensor,          # scalar
        comm_noise_std: tf.Tensor,       # scalar

        # interface config: meas
        meas_stop_gradient: tf.Tensor,   # bool scalar
        meas_hard_mode: tf.Tensor,       # int scalar: 0=none, 1=hard, 2=interp
        meas_beta: tf.Tensor,            # scalar
        meas_lambda: tf.Tensor,          # scalar
        meas_noise_std: tf.Tensor,       # scalar

        # interface config: out
        out_stop_gradient: tf.Tensor,    # bool scalar
        out_hard_mode: tf.Tensor,        # int scalar: 0=none, 1=hard, 2=interp
        out_beta: tf.Tensor,             # scalar
        out_lambda: tf.Tensor,           # scalar
        out_noise_std: tf.Tensor,        # scalar
    ) -> dict[str, tf.Tensor | list[tf.Tensor]]:
        with tf.GradientTape() as tape:
            vals = _core_forward(batch, model_a, model_b, MODE_B, beta_harden, lambda_harden=lambda_harden)
            out = _losses(*vals, w_meas_in_a, w_comms_a, w_comms_b_123, w_shoot, w_comm_mag, w_meas_mag)
        vars_to_update = model_b.trainable_variables
        grads = tape.gradient(out["total"], vars_to_update)
        gv = [(g, v) for g, v in zip(grads, vars_to_update) if g is not None]
        if gv:
            opt_b.apply_gradients(gv)
        return out

    @tf.function(reduce_retracing=True)
    def train_step_ab(
        batch: Any,

        # models
        model_a: tf.keras.Model,
        model_b: tf.keras.Model,

        # optimizers
        opt_a: tf.keras.optimizers.Optimizer,
        opt_b: tf.keras.optimizers.Optimizer,

        # which model(s) get updated
        update_a: tf.Tensor,   # bool scalar
        update_b: tf.Tensor,   # bool scalar

        # whether B should run in training mode
        b_training: tf.Tensor,  # bool scalar

        # loss weights
        w_meas_in_a: tf.Tensor,     # shape [DEPTH]
        w_comms_a: tf.Tensor,       # scalar
        w_comms_b_123: tf.Tensor,   # shape [DEPTH-1]
        w_shoot: tf.Tensor,         # scalar
        w_comm_mag: tf.Tensor,      # scalar
        w_meas_mag: tf.Tensor,      # scalar

        # interface config: comm
        comm_stop_gradient: tf.Tensor,   # bool scalar
        comm_hard_mode: tf.Tensor,       # int scalar: 0=none, 1=hard, 2=interp
        comm_beta: tf.Tensor,            # scalar
        comm_lambda: tf.Tensor,          # scalar
        comm_noise_std: tf.Tensor,       # scalar

        # interface config: meas
        meas_stop_gradient: tf.Tensor,   # bool scalar
        meas_hard_mode: tf.Tensor,       # int scalar: 0=none, 1=hard, 2=interp
        meas_beta: tf.Tensor,            # scalar
        meas_lambda: tf.Tensor,          # scalar
        meas_noise_std: tf.Tensor,       # scalar

        # interface config: out
        out_stop_gradient: tf.Tensor,    # bool scalar
        out_hard_mode: tf.Tensor,        # int scalar: 0=none, 1=hard, 2=interp
        out_beta: tf.Tensor,             # scalar
        out_lambda: tf.Tensor,           # scalar
        out_noise_std: tf.Tensor,        # scalar
    ) -> dict[str, tf.Tensor | list[tf.Tensor]]:

        with tf.GradientTape(persistent=True) as tape:
            vals = _core_forward(
                batch=batch,
                model_a=model_a,
                model_b=model_b,
                beta_harden=beta_harden,
                noise_cfg=noise_cfg,
                lambda_harden=lambda_harden,
            )
            out = _losses(*vals, w_meas_in_a, w_comms_a, w_comms_b_123, w_shoot, w_comm_mag, w_meas_mag)

        vars_a = model_a.trainable_variables
        vars_b = model_b.trainable_variables
        grads_a = tape.gradient(out["total"], vars_a)
        grads_b = tape.gradient(out["total"], vars_b)
        del tape

        gv_a = [(g, v) for g, v in zip(grads_a, vars_a) if g is not None]
        gv_b = [(g, v) for g, v in zip(grads_b, vars_b) if g is not None]
        if gv_a:
            opt_a.apply_gradients(gv_a)
        if gv_b:
            opt_b.apply_gradients(gv_b)
        return out

    return train_step_a, train_step_b, train_step_a_shoot, train_step_ab

def train(config: dict[str, Any]) -> dict[str, Any]:
    ROOT = Path(config["ROOT"])
    WIP = Path(config["WIP"])
    DEPTH = int(config["DEPTH"])
    CHECKPOINT_DIR = Path(config["CHECKPOINT_DIR"])
    LOG_DIR = Path(config["LOG_DIR"])
    EPOCHS = int(config["EPOCHS"])
    LR = float(config["LR"])
    SAVE_WEIGHTS_EVERY = int(config["SAVE_WEIGHTS_EVERY"]) if config["SAVE_WEIGHTS_EVERY"] else 0
    SAVE_WEIGHTS_AT_END = bool(config["SAVE_WEIGHTS_AT_END"])
    LOAD_WEIGHTS_ON_START = bool(config["LOAD_WEIGHTS_ON_START"])
    MODEL_A_WEIGHTS_IN = config["MODEL_A_WEIGHTS_IN"]
    MODEL_B_WEIGHTS_IN = config["MODEL_B_WEIGHTS_IN"]
    LOG_FLUSH_EVERY_EPOCHS = int(config["LOG_FLUSH_EVERY_EPOCHS"]) if config["LOG_FLUSH_EVERY_EPOCHS"] else 0

    W_COMM_MAG = float(config.get("W_COMM_MAG", 0.01))
    W_MEAS_MAG = float(config.get("W_MEAS_MAG", 0.01))
    COMM_MAG_BETA_TARGET = float(config.get("COMM_MAG_BETA_TARGET", 1.0))
    PHASE_D_COMM_MAG_THRESHOLD = float(config.get("PHASE_D_COMM_MAG_THRESHOLD", 0.02))
    PHASE_D_SHOOT_ACC_FLOOR = float(config.get("PHASE_D_SHOOT_ACC_FLOOR", 0.995))
    PHASE_D_PATIENCE = int(config.get("PHASE_D_PATIENCE", 5))

    ROLLBACK_LOOKBACK = int(config.get("ROLLBACK_LOOKBACK", 20))
    ROLLBACK_DROP_ABS = float(config.get("ROLLBACK_DROP_ABS", 0.02))
    ROLLBACK_MIN_EPOCH = int(config.get("ROLLBACK_MIN_EPOCH", 50))
    ROLLBACK_COOLDOWN_EPOCHS = int(config.get("ROLLBACK_COOLDOWN_EPOCHS", 25))
    ENABLE_ROLLBACK = bool(config.get("ENABLE_ROLLBACK", True))

    tf.random.set_seed(int(config["SEED"]))

    raw_ds = load_dataset(config)
    tfds_train = build_train_pipeline(raw_ds, config)
    tfds_train = tfds_train.prefetch(tf.data.AUTOTUNE)

    model_a = build_model_a(config)
    model_b = build_model_b(config)
    opt_a = tf.keras.optimizers.Adam(learning_rate=LR)
    opt_b = tf.keras.optimizers.Adam(learning_rate=LR)

    first_batch = next(iter(tfds_train))
    _force_build_models(model_a, model_b, first_batch)
    opt_a.build(model_a.trainable_variables)
    opt_b.build(model_b.trainable_variables)

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

        loaded = load_ab_weights(model_a, model_b, a_in, b_in)
        used_a, used_b = (a_in, b_in) if loaded else (None, None)

        if not loaded:
            a_auto, b_auto = _latest_epoch_pair(CHECKPOINT_DIR)
            loaded = load_ab_weights(model_a, model_b, a_auto, b_auto)
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

    run_log, run_log_path = init_run_log(LOG_DIR, run_load_event)
    print(f"[log] run file: {run_log_path}")

    train_step_a, train_step_b, train_step_a_shoot, train_step_ab = make_train_steps(depth=DEPTH)

    epoch = 0
    interrupted = False
    best_shoot_acc = float("-inf")
    last_rollback_epoch = -10**9
    phase_d_success_streak = 0


    early_stop_cfg = {
            "consecutive": 3,
            "rules": [
                {
                    "shoot_acc": {"comparator": "gt", "value": 0.999},
                    "lambda_harden": {"comparator": "ge", "value": 1.0},
                }
            ]
        }
    interface_cfg = {
            "b_training": True,
            "comm": {"stop_gradient": False, "hardening": {"mode": "interp", "beta": 10.0, "lambda": 0.0}, "noise_std": 0.30},
            "meas": {"stop_gradient": False, "hardening": {"mode": "none", "beta": 10.0, "lambda": 0.0}, "noise_std": 0.30},
            "out":  {"stop_gradient": False, "hardening": {"mode": "none", "beta": 10.0, "lambda": 0.0}, "noise_std": 0.30},
        }
    stage_cfg = {
                "mode" : MODE_AB,
                "w_meas_in_a" : [0.02, 0.02, 0.02, 0.02],
                "w_comms_a" : 0.05,
                "w_comms_b_123" : [0.02, 0.02, 0.02],
                "w_shoot" : 1.0,
                "lr_a" : 1e-4,
                "lr_b" : 2e-4,
                "beta_harden" : 10.0,
                "w_comm_mag": float(0.001),
                "w_meas_mag": float(0.0),
                "lambda_harden" : 0.95
        }

    try:
        print("Enable rollback: ", ENABLE_ROLLBACK)
        while epoch < EPOCHS:
            opt_a.learning_rate.assign(stage_cfg["lr_a"])
            opt_b.learning_rate.assign(stage_cfg["lr_b"])

            m_total = tf.keras.metrics.Mean()
            m_shoot_acc = tf.keras.metrics.Mean()
            m_comm_a = tf.keras.metrics.Mean()
            m_meas_in_a = tf.keras.metrics.Mean()
            m_comms_b = tf.keras.metrics.Mean()
            m_shoot_loss = tf.keras.metrics.Mean()
            m_comm_mag = tf.keras.metrics.Mean()
            m_comm_mag_raw = tf.keras.metrics.Mean()
            m_mean_abs_comm_for_b = tf.keras.metrics.Mean()
            m_meas_mag = tf.keras.metrics.Mean()
            m_meas_mag_raw = tf.keras.metrics.Mean()
            m_mean_abs_meas_in_a = tf.keras.metrics.Mean()
            m_meas_in_a_per = [tf.keras.metrics.Mean() for _ in range(DEPTH)]
            m_comms_b_per = [tf.keras.metrics.Mean() for _ in range(max(DEPTH - 1, 0))]

            mode = int(stage_cfg["mode"])
            if mode == MODE_A:
                step_fn = train_step_a
            elif mode == MODE_B:
                step_fn = train_step_b
            elif mode == MODE_A_SHOOT:
                step_fn = train_step_ab
            else:
                print("Warning: Unknown training mode")

            for batch in tfds_train:
                out = step_fn(
                    batch = batch,
                    model_a = model_a,
                    model_b = model_b,
                    opt_a = opt_a,
                    opt_b = opt_b,
                    w_meas_in_a=stage_cfg["w_meas_in_a"],
                    w_comms_a=stage_cfg["w_comms_a"],
                    w_comms_b_123=stage_cfg["w_comms_b_123"],
                    w_shoot=stage_cfg["w_shoot"],
                    w_comm_mag=stage_cfg["w_comm_mag"],
                    w_meas_mag=stage_cfg["w_meas_mag"],
                    beta_harden=stage_cfg["beta_harden"],
                    lambda_harden=stage_cfg["lambda_harden"],
                    )
 
                m_total.update_state(out["total"])
                m_shoot_acc.update_state(out["shoot_acc"])
                m_comm_a.update_state(out["comm_a_loss"])
                m_meas_in_a.update_state(out["meas_in_a_loss"])
                m_comms_b.update_state(out["comms_b_loss"])
                m_shoot_loss.update_state(out["shoot_loss"])
                m_comm_mag.update_state(out["comm_mag_loss"])
                m_comm_mag_raw.update_state(out["comm_mag_loss_raw"])
                m_mean_abs_comm_for_b.update_state(out["mean_abs_comm_for_b"])
                m_meas_mag.update_state(out["meas_mag_loss"])
                m_meas_mag_raw.update_state(out["meas_mag_loss_raw"])
                m_mean_abs_meas_in_a.update_state(out["mean_abs_meas_in_a"])

                for i, t in enumerate(out["meas_in_a_per"]):
                    if i < len(m_meas_in_a_per):
                        m_meas_in_a_per[i].update_state(t)
                for i, t in enumerate(out["comms_b_per"]):
                    if i < len(m_comms_b_per):
                        m_comms_b_per[i].update_state(t)

            epoch_metrics: dict[str, Any] = {
                "training_mode": MODE_NAME.get(int(stage_cfg["mode"]), "unknown"),
                "epoch": float(epoch),
                "total": float(m_total.result().numpy()),
                "shoot_acc": float(m_shoot_acc.result().numpy()),
                "shoot_loss": float(m_shoot_loss.result().numpy()),
                "comm_a_loss": float(m_comm_a.result().numpy()),
                "meas_in_a_loss": float(m_meas_in_a.result().numpy()),
                "comms_b_loss": float(m_comms_b.result().numpy()),
                "comm_mag_loss": float(m_comm_mag.result().numpy()),
                "comm_mag_loss_raw": float(m_comm_mag_raw.result().numpy()),
                "mean_abs_comm_for_b": float(m_mean_abs_comm_for_b.result().numpy()),
                "meas_mag_loss": float(m_meas_mag.result().numpy()),
                "meas_mag_loss_raw": float(m_meas_mag_raw.result().numpy()),
                "mean_abs_meas_in_a": float(m_mean_abs_meas_in_a.result().numpy()),
                "comm_mag_beta_target": float(COMM_MAG_BETA_TARGET),
                "w_comm_mag": float(stage_cfg["w_comm_mag"]),
                "w_meas_mag": float(stage_cfg["w_meas_mag"]),
                "lambda_harden": float(stage_cfg["lambda_harden"]),
                "lr_a*1000": float(opt_a.learning_rate.numpy())*1000.0,
                "lr_b*1000": float(opt_b.learning_rate.numpy())*1000.0,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            for i, mm in enumerate(m_meas_in_a_per):
                epoch_metrics[f"meas_in_a_per_{i}"] = float(mm.result().numpy())
            for i, mm in enumerate(m_comms_b_per):
                epoch_metrics[f"comms_b_per_{i}"] = float(mm.result().numpy())

            phase_d_success = (
                epoch_metrics["shoot_acc"] >= PHASE_D_SHOOT_ACC_FLOOR
                and epoch_metrics["comm_mag_loss_raw"] <= PHASE_D_COMM_MAG_THRESHOLD
            )
            phase_d_success_streak = phase_d_success_streak + 1 if phase_d_success else 0
            epoch_metrics["phase_d_success"] = bool(phase_d_success)
            epoch_metrics["phase_d_success_streak"] = int(phase_d_success_streak)

            print(
                f"epoch {epoch:03d}  "
                + "  ".join([f"{k}={v:.4f}" for k, v in epoch_metrics.items() if isinstance(v, float)])
            )
            print()

            append_epoch_log(run_log, epoch_metrics)

            if epoch_metrics["shoot_acc"] > best_shoot_acc:
                best_shoot_acc = float(epoch_metrics["shoot_acc"])
                save_ab_weights(model_a, model_b, CHECKPOINT_DIR, tag="best")

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
                if load_ab_weights(model_a, model_b, best_a, best_b):
                    last_rollback_epoch = epoch
                    rollback_stage = stage_cfg
                    rollback_event["rollback_loaded"] = True
                    rollback_event["rollback_stage_mode"] = MODE_NAME[rollback_stage["mode"]]
                    rollback_event["rollback_lr_a"] = rollback_stage["lr_a"]
                    rollback_event["rollback_lr_b"] = rollback_stage["lr_b"]
                    print(
                        "[rollback] restored best checkpoint and reduced stage learning rates "
                        f"to lr_a={rollback_stage['lr_a']:.6g}, lr_b={rollback_stage['lr_b']:.6g}"
                    )
                else:
                    rollback_event["rollback_loaded"] = False
                    print("[rollback] requested but best checkpoint files were not available.")

            run_log.setdefault("events", []).append({
                "epoch": int(epoch),
                "type": "rollback_check",
                **rollback_event,
            })

            if LOG_FLUSH_EVERY_EPOCHS and ((epoch + 1) % LOG_FLUSH_EVERY_EPOCHS == 0):
                flush_run_log(run_log, run_log_path)

            if SAVE_WEIGHTS_EVERY and ((epoch + 1) % SAVE_WEIGHTS_EVERY == 0):
                save_ab_weights(model_a, model_b, CHECKPOINT_DIR, tag=f"epoch_{epoch + 1:04d}")
                save_ab_weights(model_a, model_b, CHECKPOINT_DIR, tag="latest")
                flush_run_log(run_log, run_log_path)

            if phase_d_success_streak >= PHASE_D_PATIENCE:
                print(
                    f"Early stopping criteria met at epoch {epoch}: "
                    f"shoot_acc >= {PHASE_D_SHOOT_ACC_FLOOR:.4f} and "
                    f"comm_mag_loss_raw <= {PHASE_D_COMM_MAG_THRESHOLD:.4f} "
                    f"for {PHASE_D_PATIENCE} consecutive epoch(s)."
                )
                run_log.setdefault("events", []).append({
                    "epoch": int(epoch),
                    "type": "phase_d_early_stop",
                    "shoot_acc_floor": float(PHASE_D_SHOOT_ACC_FLOOR),
                    "comm_mag_threshold": float(PHASE_D_COMM_MAG_THRESHOLD),
                    "patience": int(PHASE_D_PATIENCE),
                })
                break

            # Early stop functionality and counter increment
            if evaluate_early_stop(epoch_metrics, early_stop_cfg):
                early_stop_count += 1
            else:
                early_stop_count = 0

            if early_stop_count >= early_stop_cfg["consecutive"]:
                print(
                    f"Early stopping criteria met at epoch {epoch}: "
                    f"shoot_acc >= {PHASE_D_SHOOT_ACC_FLOOR:.4f} and "
                    f"comm_mag_loss_raw <= {PHASE_D_COMM_MAG_THRESHOLD:.4f} "
                    f"for {PHASE_D_PATIENCE} consecutive epoch(s)."
                )
                run_log.setdefault("events", []).append({
                    "epoch": int(epoch),
                    "type": "phase_d_early_stop",
                    "shoot_acc_floor": float(PHASE_D_SHOOT_ACC_FLOOR),
                    "comm_mag_threshold": float(PHASE_D_COMM_MAG_THRESHOLD),
                    "patience": int(PHASE_D_PATIENCE),
                })
                break
            
            # Epoch increment
            epoch += 1

            # ========== END OF EPOCH LOOP =========

    except KeyboardInterrupt:
        interrupted = True
        print("[train] interrupted by user.")

    finally:
        flush_run_log(run_log, run_log_path)

        if SAVE_WEIGHTS_AT_END:
            end_tag = f"epoch_{epoch:04d}_{'interrupted' if interrupted else 'final'}"
            save_ab_weights(model_a, model_b, CHECKPOINT_DIR, tag=end_tag)
            save_ab_weights(model_a, model_b, CHECKPOINT_DIR, tag="latest")

    return {
        "log_path": str(run_log_path),
        "epochs_recorded": len(run_log.get("epochs", [])),
        "interrupted": interrupted,
    }


if __name__ == "__main__":
    cfg = get_config()
    result = train(cfg)
    print(result)