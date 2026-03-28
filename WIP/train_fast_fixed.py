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
MODE_A_SHOOT = 2
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

    comm0 = tf.cast(comms_tgt_list[0], tf.float32)
    _ = model_b.compute_with_internal(
        gun_logits=gun_logits,
        comm_in_logits=comm0,
        prev_meas_list=meas_in_a_tgt_list,
        prev_out_list=meas_out_a_tgt_list,
        training=False,
    )


class TrainState:
    """Mutable training state stored as tf.Variables outside the graph."""

    def __init__(self, depth: int):
        self.depth = int(depth)
        self.W_MEAS_IN_A = tf.Variable([0.0, 0.0, 0.0, 0.0], dtype=tf.float32, trainable=False)
        self.W_COMMS_A = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.W_COMMS_B_123 = tf.Variable([1.0, 0.0, 0.0], dtype=tf.float32, trainable=False)
        self.W_SHOOT = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.TRAINING_MODE = tf.Variable(MODE_B, dtype=tf.int32, trainable=False)
        # Keep A_SHOOT hardening identical to your prior script.
        self.BETA_HARDEN = tf.Variable(1.0, dtype=tf.float32, trainable=False)



def make_train_steps(depth: int):
    depth = int(depth)

    def _core_forward(
        batch: Any,
        model_a: tf.keras.Model,
        model_b: tf.keras.Model,
        mode: tf.Tensor,
        beta_harden: tf.Tensor,
        noise_cfg: tuple[tf.Tensor, tf.Tensor | list[tf.Tensor], tf.Tensor | list[tf.Tensor]] = (None, None, None),
    ):
        """
        noise_cfg = (
            noise_std_comm,   # scalar tf.float32 Tensor
            noise_std_meas,   # scalar tf.float32 Tensor OR tuple/list of scalar tf.float32 Tensors
            noise_std_out,    # scalar tf.float32 Tensor OR tuple/list of scalar tf.float32 Tensors
        )
        """
        noise_std_comm, noise_std_meas, noise_std_out = noise_cfg

        (
            field_logits,
            gun_logits,
            shoot_tgt_logits,
            meas_in_a_tgt_list,
            meas_out_a_tgt_list,
            comms_tgt_list,
        ) = _unpack_batch(batch)

        comm_logits, meas_list, out_list = model_a.compute_with_internal(
            field_logits=field_logits,
            replay_out_a_logits_list=meas_out_a_tgt_list,
            harden_between_levels=False,
            training=True,
        )

        if mode == MODE_A_SHOOT:
            comm_logits_for_b = comm_logits
            meas_list_for_b = list(meas_list)
            out_list_for_b = list(out_list)
            b_training_flag = False

        elif mode == MODE_A:
            comm_logits_for_b = tf.stop_gradient(comm_logits)
            meas_list_for_b = [tf.stop_gradient(t) for t in meas_list]
            out_list_for_b = [tf.stop_gradient(t) for t in out_list]
            b_training_flag = True
        
        elif mode == MODE_B:
            comm_logits_for_b = tf.stop_gradient(comm_logits)
            meas_list_for_b = [tf.stop_gradient(t) for t in meas_list]
            out_list_for_b = [tf.stop_gradient(t) for t in out_list]
            b_training_flag = True

        else:  # MODE_AB
            comm_logits_for_b = comm_logits
            meas_list_for_b = list(meas_list)
            out_list_for_b = list(out_list)
            b_training_flag = True

            # Noise only in AB mode
            comm_logits_for_b = add_logit_noise_tf(
                comm_logits_for_b,
                noise_std_comm,
                training=True,
            )

            meas_list_for_b = add_logit_noise_tf(
                meas_list_for_b,
                noise_std_meas,
                training=True,
            )

            out_list_for_b = add_logit_noise_tf(
                out_list_for_b,
                noise_std_out,
                training=True,
            )

        shoot_logit, _, _, comms_logits_list, _ = model_b.compute_with_internal(
            gun_logits,
            comm_logits_for_b,
            list(meas_list_for_b),
            list(out_list_for_b),
            harden_between_levels=False,
            training=b_training_flag,
        )

        return (
            comm_logits,
            meas_list,
            out_list,
            shoot_logit,
            comms_logits_list,
            shoot_tgt_logits,
            meas_in_a_tgt_list,
            comms_tgt_list,
        )

    def _losses(
        comm_logits: tf.Tensor,
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

        total = shoot_loss + comm_a_loss + meas_in_a_loss + comms_b_loss

        return {
            "total": total,
            "shoot_acc": shoot_acc,
            "comm_a_loss": comm_a_loss,
            "meas_in_a_loss": meas_in_a_loss,
            "meas_in_a_per": meas_in_a_per,
            "comms_b_loss": comms_b_loss,
            "comms_b_per": comms_b_per,
            "shoot_loss": shoot_loss,
        }

    @tf.function(reduce_retracing=True)
    def train_step_a(
        batch: Any,
        model_a: tf.keras.Model,
        model_b: tf.keras.Model,
        opt: tf.keras.optimizers.Optimizer,
        w_meas_in_a: tf.Tensor,
        w_comms_a: tf.Tensor,
        w_comms_b_123: tf.Tensor,
        w_shoot: tf.Tensor,
        beta_harden: tf.Tensor,
    ) -> dict[str, tf.Tensor | list[tf.Tensor]]:
        with tf.GradientTape() as tape:
            vals = _core_forward(batch, model_a, model_b, MODE_A, beta_harden)
            out = _losses(*vals, w_meas_in_a, w_comms_a, w_comms_b_123, w_shoot)
        vars_to_update = model_a.trainable_variables
        grads = tape.gradient(out["total"], vars_to_update)
        gv = [(g, v) for g, v in zip(grads, vars_to_update) if g is not None]
        if gv:
            opt.apply_gradients(gv)
        return out

    @tf.function(reduce_retracing=True)
    def train_step_b(
        batch: Any,
        model_a: tf.keras.Model,
        model_b: tf.keras.Model,
        opt: tf.keras.optimizers.Optimizer,
        w_meas_in_a: tf.Tensor,
        w_comms_a: tf.Tensor,
        w_comms_b_123: tf.Tensor,
        w_shoot: tf.Tensor,
        beta_harden: tf.Tensor,
    ) -> dict[str, tf.Tensor | list[tf.Tensor]]:
        with tf.GradientTape() as tape:
            vals = _core_forward(batch, model_a, model_b, MODE_B, beta_harden)
            out = _losses(*vals, w_meas_in_a, w_comms_a, w_comms_b_123, w_shoot)
        vars_to_update = model_b.trainable_variables
        grads = tape.gradient(out["total"], vars_to_update)
        gv = [(g, v) for g, v in zip(grads, vars_to_update) if g is not None]
        if gv:
            opt.apply_gradients(gv)
        return out

    @tf.function(reduce_retracing=True)
    def train_step_a_shoot(
        batch: Any,
        model_a: tf.keras.Model,
        model_b: tf.keras.Model,
        opt: tf.keras.optimizers.Optimizer,
        w_meas_in_a: tf.Tensor,
        w_comms_a: tf.Tensor,
        w_comms_b_123: tf.Tensor,
        w_shoot: tf.Tensor,
        beta_harden: tf.Tensor,
    ) -> dict[str, tf.Tensor | list[tf.Tensor]]:
        with tf.GradientTape() as tape:
            vals = _core_forward(batch, model_a, model_b, MODE_A_SHOOT, beta_harden)
            out = _losses(*vals, w_meas_in_a, w_comms_a, w_comms_b_123, w_shoot)
        vars_to_update = model_a.trainable_variables
        grads = tape.gradient(out["total"], vars_to_update)
        gv = [(g, v) for g, v in zip(grads, vars_to_update) if g is not None]
        if gv:
            opt.apply_gradients(gv)
        return out

    @tf.function(reduce_retracing=True)
    def train_step_ab(
        batch: Any,
        model_a: tf.keras.Model,
        model_b: tf.keras.Model,
        opt: tf.keras.optimizers.Optimizer,
        w_meas_in_a: tf.Tensor,
        w_comms_a: tf.Tensor,
        w_comms_b_123: tf.Tensor,
        w_shoot: tf.Tensor,
        beta_harden: tf.Tensor,
    ) -> dict[str, tf.Tensor | list[tf.Tensor]]:
        with tf.GradientTape() as tape:
            noise_cfg = (
                tf.constant(0.30, dtype=tf.float32),  # comm
                tf.constant(0.30, dtype=tf.float32),  # meas
                tf.constant(0.30, dtype=tf.float32),  # out
            )

            vals = _core_forward(
                batch=batch,
                model_a=model_a,
                model_b=model_b,
                mode=tf.constant(MODE_AB, dtype=tf.int32),
                beta_harden=tf.constant(0.0, dtype=tf.float32),
                noise_cfg=noise_cfg,
            )
            #vals = _core_forward(batch, model_a, model_b, MODE_AB, beta_harden)
            out = _losses(*vals, w_meas_in_a, w_comms_a, w_comms_b_123, w_shoot)
        vars_to_update = model_a.trainable_variables + model_b.trainable_variables
        grads = tape.gradient(out["total"], vars_to_update)
        gv = [(g, v) for g, v in zip(grads, vars_to_update) if g is not None]
        if gv:
            opt.apply_gradients(gv)
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

    tf.random.set_seed(int(config["SEED"]))

    raw_ds = load_dataset(config)
    tfds_train = build_train_pipeline(raw_ds, config)
    tfds_train = tfds_train.prefetch(tf.data.AUTOTUNE)

    model_a = build_model_a(config)
    model_b = build_model_b(config)
    opt = tf.keras.optimizers.Adam(learning_rate=LR)

    first_batch = next(iter(tfds_train))
    _force_build_models(model_a, model_b, first_batch)
    opt.build(model_a.trainable_variables + model_b.trainable_variables)

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

    state = TrainState(depth=DEPTH)
    train_step_a, train_step_b, train_step_a_shoot, train_step_ab = make_train_steps(depth=DEPTH)

    epoch = 0
    interrupted = False
    full_system_optimization_started = False

    try:
        while epoch < EPOCHS:
            m_total = tf.keras.metrics.Mean()
            m_shoot_acc = tf.keras.metrics.Mean()
            m_comm_a = tf.keras.metrics.Mean()
            m_meas_in_a = tf.keras.metrics.Mean()
            m_comms_b = tf.keras.metrics.Mean()
            m_shoot_loss = tf.keras.metrics.Mean()
            m_meas_in_a_per = [tf.keras.metrics.Mean() for _ in range(DEPTH)]
            m_comms_b_per = [tf.keras.metrics.Mean() for _ in range(max(DEPTH - 1, 0))]

            mode = int(state.TRAINING_MODE.numpy())
            if mode == MODE_A:
                step_fn = train_step_a
            elif mode == MODE_B:
                step_fn = train_step_b
            elif mode == MODE_A_SHOOT:
                step_fn = train_step_a_shoot
            else:
                step_fn = train_step_ab

            for batch in tfds_train:
                out = step_fn(
                    batch,
                    model_a,
                    model_b,
                    opt,
                    state.W_MEAS_IN_A.read_value(),
                    state.W_COMMS_A.read_value(),
                    state.W_COMMS_B_123.read_value(),
                    state.W_SHOOT.read_value(),
                    state.BETA_HARDEN.read_value(),
                )

                m_total.update_state(out["total"])
                m_shoot_acc.update_state(out["shoot_acc"])
                m_comm_a.update_state(out["comm_a_loss"])
                m_meas_in_a.update_state(out["meas_in_a_loss"])
                m_comms_b.update_state(out["comms_b_loss"])
                m_shoot_loss.update_state(out["shoot_loss"])

                for i, t in enumerate(out["meas_in_a_per"]):
                    if i < len(m_meas_in_a_per):
                        m_meas_in_a_per[i].update_state(t)
                for i, t in enumerate(out["comms_b_per"]):
                    if i < len(m_comms_b_per):
                        m_comms_b_per[i].update_state(t)

            epoch_metrics: dict[str, Any] = {
                "total": float(m_total.result().numpy()),
                "shoot_acc": float(m_shoot_acc.result().numpy()),
                "comm_a_loss": float(m_comm_a.result().numpy()),
                "meas_in_a_loss": float(m_meas_in_a.result().numpy()),
                "comms_b_loss": float(m_comms_b.result().numpy()),
                "shoot_loss": float(m_shoot_loss.result().numpy()),
                "epoch": int(epoch),
                "training_mode": MODE_NAME.get(int(state.TRAINING_MODE.numpy()), "unknown"),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            for i, mm in enumerate(m_meas_in_a_per):
                epoch_metrics[f"meas_in_a_per_{i}"] = float(mm.result().numpy())
            for i, mm in enumerate(m_comms_b_per):
                epoch_metrics[f"comms_b_per_{i}"] = float(mm.result().numpy())

            print(
                f"epoch {epoch:03d}  "
                + "  ".join([f"{k}={v:.4f}" for k, v in epoch_metrics.items() if isinstance(v, float)])
            )
            print()

            append_epoch_log(run_log, epoch_metrics)

            w = state.W_COMMS_B_123.numpy().reshape(-1)
            is_100 = np.allclose(w, np.array([1.0, 0.0, 0.0], dtype=np.float32))
            is_110 = np.allclose(w, np.array([0.1, 1.0, 0.0], dtype=np.float32))
            is_111 = np.allclose(w, np.array([0.1, 0.1, 1.0], dtype=np.float32))

            if (
                not full_system_optimization_started
                and epoch_metrics["shoot_acc"] <= 0.9
                and epoch_metrics["comms_b_loss"] < 0.1
            ):
                if is_100:
                    print("Moving to next phase: enabling comms B levels 2 and 3 losses.")
                    state.W_COMMS_B_123.assign([0.1, 1.0, 0.0])
                elif is_110:
                    print("Moving to next phase: enabling comms B level 3 loss.")
                    state.W_COMMS_B_123.assign([0.1, 0.1, 1.0])
                elif is_111 and float(state.W_SHOOT.numpy()) < 0.5:
                    print("All comms B losses are already enabled. including the shoot loss.")
                    state.W_COMMS_B_123.assign([0.1, 0.1, 0.1])
                    state.W_SHOOT.assign(1.0)

            if (not full_system_optimization_started) and epoch_metrics["shoot_acc"] > 0.9:
                full_system_optimization_started = True
                print("\nEnabling full system optimization with shoot loss and all comms losses.")
                opt.learning_rate.assign(float(opt.learning_rate.numpy()) * 1.0)
                print(f"Reducing learning rate to {opt.learning_rate.numpy()} for shoot loss optimization.\n")

            if full_system_optimization_started:
                if epoch % 25 < 5:
                    print("Focusing model A with shoot loss for this epoch.")
                    state.W_MEAS_IN_A.assign([0.1, 0.1, 0.1, 0.1])
                    state.W_COMMS_A.assign(0.1)
                    state.W_COMMS_B_123.assign([0.0, 0.0, 0.0])
                    state.W_SHOOT.assign(1.0)
                    state.TRAINING_MODE.assign(MODE_A_SHOOT)
                elif epoch % 25 < 15:
                    print("Focusing model B with shoot loss for this epoch.")
                    state.W_MEAS_IN_A.assign([0.0, 0.0, 0.0, 0.0])
                    state.W_COMMS_A.assign(0.0)
                    state.W_COMMS_B_123.assign([0.1, 0.1, 0.1])
                    state.W_SHOOT.assign(1.0)
                    state.TRAINING_MODE.assign(MODE_B)
                else:
                    print("Optimizing full system with all losses for this epoch.")
                    state.W_MEAS_IN_A.assign([0.1, 0.1, 0.1, 0.1])
                    state.W_COMMS_A.assign(0.1)
                    state.W_COMMS_B_123.assign([0.1, 0.1, 0.1])
                    state.W_SHOOT.assign(1.0)
                    state.TRAINING_MODE.assign(MODE_AB)

            if LOG_FLUSH_EVERY_EPOCHS and ((epoch + 1) % LOG_FLUSH_EVERY_EPOCHS == 0):
                flush_run_log(run_log, run_log_path)

            if SAVE_WEIGHTS_EVERY and ((epoch + 1) % SAVE_WEIGHTS_EVERY == 0):
                save_ab_weights(model_a, model_b, CHECKPOINT_DIR, tag=f"epoch_{epoch + 1:04d}")
                save_ab_weights(model_a, model_b, CHECKPOINT_DIR, tag="latest")
                flush_run_log(run_log, run_log_path)

            if epoch_metrics["shoot_acc"] > 0.99:
                print(f"Early stopping criteria met at epoch {epoch}.")
                break

            epoch += 1

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
