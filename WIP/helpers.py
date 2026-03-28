from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pickle
import re
from typing import Any, Sequence

import tensorflow as tf


def magnitude_margin_loss(
    logits: tf.Tensor,
    beta_target: float | tf.Tensor = 1.0,
    *,
    reduction: str = "mean",
) -> tf.Tensor:
    """
    Sign-agnostic margin loss on logits.

    Encourages |logits| >= beta_target, but does not penalize larger magnitudes.
    Best low-risk first choice for gameplay preparation.

    Args:
        logits:
            Tensor of logits.
        beta_target:
            Desired minimum absolute magnitude.
        reduction:
            "mean", "sum", or "none".

    Returns:
        Scalar loss if reduction != "none", else elementwise tensor.
    """
    z = tf.cast(logits, tf.float32)
    beta_target = tf.cast(beta_target, tf.float32)

    per_elem = tf.square(tf.nn.relu(beta_target - tf.abs(z)))

    if reduction == "mean":
        return tf.reduce_mean(per_elem)
    if reduction == "sum":
        return tf.reduce_sum(per_elem)
    if reduction == "none":
        return per_elem
    raise ValueError(f"Unsupported reduction: {reduction!r}")


def magnitude_target_loss(
    logits: tf.Tensor,
    beta_target: float | tf.Tensor = 1.0,
    *,
    reduction: str = "mean",
) -> tf.Tensor:
    """
    Sign-agnostic target-magnitude loss on logits.

    Encourages |logits| ~= beta_target.
    Stricter than magnitude_margin_loss because it also penalizes too-large magnitudes.

    Args:
        logits:
            Tensor of logits.
        beta_target:
            Desired absolute magnitude.
        reduction:
            "mean", "sum", or "none".

    Returns:
        Scalar loss if reduction != "none", else elementwise tensor.
    """
    z = tf.cast(logits, tf.float32)
    beta_target = tf.cast(beta_target, tf.float32)

    per_elem = tf.square(tf.abs(z) - beta_target)

    if reduction == "mean":
        return tf.reduce_mean(per_elem)
    if reduction == "sum":
        return tf.reduce_sum(per_elem)
    if reduction == "none":
        return per_elem
    raise ValueError(f"Unsupported reduction: {reduction!r}")


def magnitude_target_huber_loss(
    logits: tf.Tensor,
    beta_target: float | tf.Tensor = 1.0,
    *,
    delta: float | tf.Tensor = 0.25,
    reduction: str = "mean",
) -> tf.Tensor:
    """
    Huber version of sign-agnostic target-magnitude loss.

    Encourages |logits| ~= beta_target, but is less harsh than pure MSE.
    Good second-step alternative if magnitude_margin_loss works and you want tighter calibration.

    Args:
        logits:
            Tensor of logits.
        beta_target:
            Desired absolute magnitude.
        delta:
            Huber transition point.
        reduction:
            "mean", "sum", or "none".

    Returns:
        Scalar loss if reduction != "none", else elementwise tensor.
    """
    z = tf.cast(logits, tf.float32)
    beta_target = tf.cast(beta_target, tf.float32)
    delta = tf.cast(delta, tf.float32)

    err = tf.abs(z) - beta_target
    abs_err = tf.abs(err)

    quadratic = 0.5 * tf.square(err)
    linear = delta * (abs_err - 0.5 * delta)
    per_elem = tf.where(abs_err <= delta, quadratic, linear)

    if reduction == "mean":
        return tf.reduce_mean(per_elem)
    if reduction == "sum":
        return tf.reduce_sum(per_elem)
    if reduction == "none":
        return per_elem
    raise ValueError(f"Unsupported reduction: {reduction!r}")


def per_level_magnitude_margin_loss(
    logits_list: Sequence[tf.Tensor],
    beta_target: float | tf.Tensor | Sequence[float | tf.Tensor] = 1.0,
    *,
    level_weights: Sequence[float | tf.Tensor] | None = None,
) -> tf.Tensor:
    """
    Apply magnitude_margin_loss to a per-depth list of logits and average/weight levels.

    Args:
        logits_list:
            Sequence of tensors, one per level.
        beta_target:
            Scalar applied to all levels, or per-level sequence.
        level_weights:
            Optional per-level weights. If None, all levels weighted equally.

    Returns:
        Scalar loss.
    """
    xs = list(logits_list)
    if len(xs) == 0:
        return tf.constant(0.0, dtype=tf.float32)

    if isinstance(beta_target, (list, tuple)):
        if len(beta_target) != len(xs):
            raise ValueError(f"beta_target length must match logits_list length: {len(beta_target)} != {len(xs)}")
        betas = [tf.cast(b, tf.float32) for b in beta_target]
    else:
        b = tf.cast(beta_target, tf.float32)
        betas = [b] * len(xs)

    if level_weights is None:
        ws = [tf.constant(1.0, dtype=tf.float32)] * len(xs)
    else:
        if len(level_weights) != len(xs):
            raise ValueError(f"level_weights length must match logits_list length: {len(level_weights)} != {len(xs)}")
        ws = [tf.cast(w, tf.float32) for w in level_weights]

    losses = []
    weight_sum = tf.constant(0.0, dtype=tf.float32)

    for x, b, w in zip(xs, betas, ws):
        losses.append(w * magnitude_margin_loss(x, beta_target=b, reduction="mean"))
        weight_sum += w

    total = tf.add_n(losses)
    return tf.math.divide_no_nan(total, weight_sum)


def per_level_magnitude_target_huber_loss(
    logits_list: Sequence[tf.Tensor],
    beta_target: float | tf.Tensor | Sequence[float | tf.Tensor] = 1.0,
    *,
    delta: float | tf.Tensor = 0.25,
    level_weights: Sequence[float | tf.Tensor] | None = None,
) -> tf.Tensor:
    """
    Apply magnitude_target_huber_loss to a per-depth list of logits and average/weight levels.

    Args:
        logits_list:
            Sequence of tensors, one per level.
        beta_target:
            Scalar applied to all levels, or per-level sequence.
        delta:
            Huber delta.
        level_weights:
            Optional per-level weights. If None, all levels weighted equally.

    Returns:
        Scalar loss.
    """
    xs = list(logits_list)
    if len(xs) == 0:
        return tf.constant(0.0, dtype=tf.float32)

    if isinstance(beta_target, (list, tuple)):
        if len(beta_target) != len(xs):
            raise ValueError(f"beta_target length must match logits_list length: {len(beta_target)} != {len(xs)}")
        betas = [tf.cast(b, tf.float32) for b in beta_target]
    else:
        b = tf.cast(beta_target, tf.float32)
        betas = [b] * len(xs)

    if level_weights is None:
        ws = [tf.constant(1.0, dtype=tf.float32)] * len(xs)
    else:
        if len(level_weights) != len(xs):
            raise ValueError(f"level_weights length must match logits_list length: {len(level_weights)} != {len(xs)}")
        ws = [tf.cast(w, tf.float32) for w in level_weights]

    losses = []
    weight_sum = tf.constant(0.0, dtype=tf.float32)

    for x, b, w in zip(xs, betas, ws):
        losses.append(w * magnitude_target_huber_loss(x, beta_target=b, delta=delta, reduction="mean"))
        weight_sum += w

    total = tf.add_n(losses)
    return tf.math.divide_no_nan(total, weight_sum)

# ----------------------------
# Generic loss/utility helpers
# ----------------------------
def bce_from_logits(tgt_logits: tf.Tensor, pred_logits: tf.Tensor) -> tf.Tensor:
    """
    BCE in logit space using hard targets from target logits:
    target_bit = 1.0 if tgt_logit >= 0 else 0.0
    """
    y_true = tf.cast(tgt_logits >= 0.0, tf.float32)
    per_elem = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=tf.cast(pred_logits, tf.float32))
    return tf.reduce_mean(per_elem)

def weighted_per_level_bce(tgt_logits_list, pred_logits_list, weights: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    tgt_logits_list/pred_logits_list: Python list length L, each (B, D_i) or (B,1)
    weights: shape (L,), float32

    Returns: (loss_scalar, per_level_losses shape (L,))
    """
    assert len(tgt_logits_list) == len(pred_logits_list)
    L = len(tgt_logits_list)
    weights = tf.cast(weights, tf.float32)
    if weights.shape.rank != 1 or int(weights.shape[0]) != L:
        raise ValueError(f"weights must be shape ({L},), got {weights.shape}")

    per_level = []
    for d in range(L):
        per_level.append(bce_from_logits(tgt_logits_list[d], pred_logits_list[d]))
    per_level = tf.stack(per_level, axis=0)  # (L,)
    loss = tf.reduce_sum(per_level * weights) / (tf.reduce_sum(weights) + 1e-8)
    return loss, per_level

def harden_ste(x: tf.Tensor, beta: float = 10.0) -> tf.Tensor:
    """
    Straight-through hardening:
    forward: +/- beta by sign(x), backward: identity gradient.
    """
    x = tf.cast(x, tf.float32)
    y_hard = tf.where(x >= 0.0, tf.cast(beta, tf.float32), tf.cast(-beta, tf.float32))
    return x + tf.stop_gradient(y_hard - x)

import tensorflow as tf
from typing import Sequence


def add_logit_noise_tf(
    x,
    stddev,
    *,
    training: bool = True,
):
    """
    Graph-safe additive Gaussian noise for logits.

    Supports:
      - x: single Tensor, stddev: scalar Tensor/float
      - x: list/tuple[Tensor], stddev:
            * scalar Tensor/float -> same stddev for all
            * list/tuple of scalar Tensor/float -> per-item stddev

    Notes:
      - If stddev <= 0, the input is returned unchanged.
      - Intended for use inside tf.function.
    """
    if not training:
        return list(x) if isinstance(x, (list, tuple)) else x

    def _add_one(t: tf.Tensor, sd) -> tf.Tensor:
        t = tf.cast(t, tf.float32)
        sd = tf.cast(sd, tf.float32)

        return tf.cond(
            sd > 0.0,
            lambda: t + tf.random.normal(
                shape=tf.shape(t),
                mean=0.0,
                stddev=sd,
                dtype=tf.float32,
            ),
            lambda: t,
        )

    if isinstance(x, (list, tuple)):
        xs = list(x)

        if isinstance(stddev, (list, tuple)):
            if len(stddev) != len(xs):
                raise ValueError(f"stddev length must match input length: {len(stddev)} != {len(xs)}")
            return [_add_one(t, sd) for t, sd in zip(xs, stddev)]

        return [_add_one(t, stddev) for t in xs]

    return _add_one(x, stddev) 

def logits_l2_reg(*tensors: tf.Tensor, weight: float = 1e-4) -> tf.Tensor:
    """
    weight * sum_i mean(square(tensor_i))
    """
    reg = tf.constant(0.0, tf.float32)
    for t in tensors:
        t = tf.cast(t, tf.float32)
        reg += tf.reduce_mean(tf.square(t))
    return tf.cast(weight, tf.float32) * reg


def _xor_logit_from_logits(a: tf.Tensor, b: tf.Tensor, eps: float = 1e-6) -> tf.Tensor:
    """
    Soft XOR probability from sigmoid(a), sigmoid(b), then mapped to logit.
    """
    p = tf.sigmoid(tf.cast(a, tf.float32))
    q = tf.sigmoid(tf.cast(b, tf.float32))
    xor_prob = p + q - 2.0 * p * q
    xor_prob = tf.clip_by_value(xor_prob, eps, 1.0 - eps)
    return tf.math.log(xor_prob) - tf.math.log(1.0 - xor_prob)


def flip_loss_from_logits_lists(
    tgt_logits_list: list[tf.Tensor],
    pred_logits_list: list[tf.Tensor],
    flip_weights: tf.Tensor | list[float],
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Transition flip BCE over adjacent levels.

    Returns:
      - weighted scalar loss
      - per-transition unweighted BCE tensor, shape [L-1]
    """
    if len(tgt_logits_list) != len(pred_logits_list):
        raise ValueError("tgt_logits_list and pred_logits_list must have same length.")
    if len(tgt_logits_list) < 2:
        raise ValueError("Need at least 2 levels for flip transitions.")

    L = len(tgt_logits_list)
    fw = tf.cast(flip_weights, tf.float32)
    fw = tf.reshape(fw, [-1])
    if fw.shape.rank != 1:
        raise ValueError("flip_weights must be 1D.")
    if fw.shape[0] is not None and int(fw.shape[0]) != (L - 1):
        raise ValueError(f"flip_weights length must be L-1={L-1}, got {int(fw.shape[0])}.")

    per = []
    for d in range(L - 1):
        tgt_a = tf.cast(tgt_logits_list[d], tf.float32)
        tgt_b = tf.cast(tgt_logits_list[d + 1], tf.float32)
        pred_a = tf.cast(pred_logits_list[d], tf.float32)
        pred_b = tf.cast(pred_logits_list[d + 1], tf.float32)

        tgt_flip = tf.cast(tf.not_equal(tgt_a >= 0.0, tgt_b >= 0.0), tf.float32)
        pred_flip_logit = _xor_logit_from_logits(pred_a, pred_b)

        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=tgt_flip, logits=pred_flip_logit)
        per.append(tf.reduce_mean(bce))

    per_t = tf.stack(per, axis=0)  # [L-1]
    weighted = tf.reduce_sum(per_t * fw) / tf.maximum(tf.reduce_sum(fw), 1e-8)
    return weighted, per_t


# ----------------------------
# Checkpoint helpers
# ----------------------------
def _wfile(base_dir: Path, model_name: str, tag: str) -> Path:
    return Path(base_dir) / f"{model_name}_{tag}.weights.h5"


def save_ab_weights(model_a: Any, model_b: Any, base_dir: Path, tag: str) -> tuple[Path | None, Path | None]:
    # If a model explicitly reports built=False, skip saving to avoid partial/invalid checkpoints.
    if getattr(model_a, "built", True) is False or getattr(model_b, "built", True) is False:
        return None, None

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    a_path = _wfile(base_dir, "model_a", tag)
    b_path = _wfile(base_dir, "model_b", tag)

    try:
        model_a.save_weights_to(str(a_path))
        model_b.save_weights_to(str(b_path))
        return a_path, b_path
    except Exception as e:
        return None, None


def load_ab_weights(model_a: Any, model_b: Any, a_path: Path | None, b_path: Path | None) -> bool:
    if a_path is None or b_path is None:
        return False

    a = Path(a_path)
    b = Path(b_path)
    if not a.exists() or not b.exists():
        return False

    model_a.load_weights_from(str(a))
    model_b.load_weights_from(str(b))
    return True


def _latest_epoch_pair(base_dir: Path) -> tuple[Path | None, Path | None]:
    base_dir = Path(base_dir)
    a_files = sorted(base_dir.glob("model_a_epoch_*.weights.h5"))
    b_files = sorted(base_dir.glob("model_b_epoch_*.weights.h5"))
    if not a_files or not b_files:
        return None, None

    def _ep(p: Path) -> int:
        m = re.search(r"_epoch_(\d+)\.weights\.h5$", p.name)
        return int(m.group(1)) if m else -1

    a_by = {_ep(p): p for p in a_files}
    b_by = {_ep(p): p for p in b_files}
    common = sorted(set(a_by.keys()) & set(b_by.keys()))
    if not common:
        return None, None
    e = common[-1]
    return a_by[e], b_by[e]


# ----------------------------
# Run logging helpers (.pkl)
# ----------------------------
def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _iso_mtime(path: Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")


def init_run_log(log_dir: Path, load_event: dict[str, Any], log_name: str | None = None) -> tuple[dict[str, Any], Path]:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    run_id = _now_tag()
    if log_name is None:
        log_path = log_dir / f"train_run_{run_id}.pkl"
    else:
        log_path = log_dir / log_name

    run_log: dict[str, Any] = {
        "meta": {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "start_mode": load_event.get("start_mode", "fresh"),
            "loaded_from": load_event.get("loaded_from", None),
        },
        "epochs": [],
    }
    return run_log, log_path


def append_epoch_log(run_log: dict[str, Any], epoch_row: dict[str, Any]) -> None:
    run_log["epochs"].append(epoch_row)


def flush_run_log(run_log: dict[str, Any], log_path: Path) -> None:
    """ Atomically write run_log to log_path as a pickle file."""
    log_path = Path(log_path)
    tmp = log_path.with_suffix(log_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(run_log, f)
    tmp.replace(log_path)
    


# ----------------------------
# Compatibility aliases
# ----------------------------
latest_epoch_pair = _latest_epoch_pair