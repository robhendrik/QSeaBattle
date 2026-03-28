# %%
from __future__ import annotations

import os, sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# Optional: reduce oneDNN variability on some Windows installs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def change_to_repo_root(marker: str = "WIP") -> None:
    """Change CWD to repository root (where marker directory exists)."""
    here = Path.cwd()
    for parent in [here] + list(here.parents):
        if (parent / marker).is_dir():
            os.chdir(parent)
            return
    raise RuntimeError(f"Could not find repo root containing '{marker}/' starting from {here}")

# Match your existing notebooks
change_to_repo_root("WIP")
ROOT = Path.cwd()
WIP_SRC  = ROOT / "WIP" / "src"
CORE_SRC = ROOT / "src"
WIP      = ROOT / "WIP"

sys.path.insert(0, str(WIP_SRC))
sys.path.insert(0, str(CORE_SRC))
sys.path.insert(0, str(WIP))

from Q_Sea_Battle_New.pyr_dataset_generation_utilities import generate_pyr_dataset
from Q_Sea_Battle_New.pyr_dataset_conversion_utilities import convert_all_traces
from Q_Sea_Battle_New.pyr_internal_model_a import PyrInternalModelA
from Q_Sea_Battle_New.pyr_internal_model_b import PyrInternalModelB
from Q_Sea_Battle.game_layout import GameLayout

print("Repo root:", ROOT)

# %%
# -----------------------------
# Settings
# -----------------------------
FIELD_SIZE = 4
N2 = FIELD_SIZE * FIELD_SIZE
COMMS_SIZE = 1
P_HIGH = 1.0
NUM_GAMES_DATASET = 150_000
SEED = 1234

# Bits -> logits conversion hardness used at the *training boundary*
BETA_INPUT = 10.0

BATCH = 256
EPOCHS = 250
LR = 1e-3

# Exploration is NOT used during supervised training.
# (If you do curriculum with noise later, add it explicitly in the forward pass.)

LAYOUT = GameLayout(
    field_size=FIELD_SIZE,
    comms_size=COMMS_SIZE,
    number_of_games_in_tournament=1000,
    channel_noise=0.0,
    enemy_probability=0.5,
)
SAVE_PATH = Path("WIP") / "dataset" / "tfds_train_raw"   # current working dir / WIP / dataset
DEPTH = int(np.log2(N2))
assert 2**DEPTH == N2, "N2 must be a power of 2"

# ---- Run logging ----
LOG_DIR = ROOT / "WIP" / "logs"
LOG_FLUSH_EVERY_EPOCHS = 0   # 0 => flush only at checkpoint moments + final/interrupt; set 1 for max safety

# ---- Weights I/O options ----
CHECKPOINT_DIR = ROOT / "WIP" / "checkpoints" / "weights_pyr_models"
LOAD_WEIGHTS_ON_START = False          # True => try loading at model init
MODEL_A_WEIGHTS_IN = None              # e.g. CHECKPOINT_DIR / "model_a_latest.weights.h5"
MODEL_B_WEIGHTS_IN = None              # e.g. CHECKPOINT_DIR / "model_b_latest.weights.h5"

SAVE_WEIGHTS_EVERY = 100               # 0/None disables periodic saves
SAVE_WEIGHTS_AT_END = True

print("N2:", N2, "DEPTH:", DEPTH)

# %%
# -----------------------------
# Create canonical dataset (bits) and logit training views
# -----------------------------
ds_bits = generate_pyr_dataset(n2=N2, num_games=NUM_GAMES_DATASET, seed=SEED, validate=True)

# Convert ALL traces to logits views (hard_logit) so targets are in logit-space.
# (Values are ±BETA_INPUT; semantic bit = 1[logit >= 0])
tr = convert_all_traces(
    ds_bits,
    rep_field="hard_logit",
    rep_gun="hard_logit",
    rep_comms="hard_logit",
    rep_meas_in_a="hard_logit",
    rep_meas_out_a="hard_logit",
    rep_meas_in_b="hard_logit",
    rep_meas_out_b="hard_logit",
    rep_shoot="hard_logit",   # gives shoot target as logits; we'll use bits via sign below
    beta=BETA_INPUT,
)

# Convenience handles:
field0 = tr["field"][0]            # (N, n2)
gun0   = tr["gun"][0]              # (N, n2)
shoot_tgt_logits = tr["shoot"]     # (N, 1) logits view

# A targets (logits)
meas_in_a_tgt_list  = [tr["meas_in_a"][d]  for d in range(DEPTH)]  # list of (N, k_d)
meas_out_a_tgt_list = [tr["meas_out_a"][d] for d in range(DEPTH)]  # list of (N, k_d)

# B comm targets per level (logits), length DEPTH+1, each (N,1)
comms_tgt_list = [tr["comms"][d] for d in range(DEPTH+1)]

print("field0", field0.shape, "gun0", gun0.shape, "shoot", shoot_tgt_logits.shape)
print("meas_in_a[0]", meas_in_a_tgt_list[0].shape, "comms[0]", comms_tgt_list[0].shape)

# %%
# -----------------------------
# Build tf.data dataset
# -----------------------------
# Keep everything as float32 for TF.
N = field0.shape[0]

def make_ds():
    ds = tf.data.Dataset.from_tensor_slices((
        tf.convert_to_tensor(field0, tf.float32),
        tf.convert_to_tensor(gun0, tf.float32),
        tf.convert_to_tensor(shoot_tgt_logits, tf.float32),
        # Pack lists as tuple-of-tensors so Dataset can carry them
        tuple(tf.convert_to_tensor(x, tf.float32) for x in meas_in_a_tgt_list),
        tuple(tf.convert_to_tensor(x, tf.float32) for x in meas_out_a_tgt_list),
        tuple(tf.convert_to_tensor(x, tf.float32) for x in comms_tgt_list),
    ))
    #ds = ds.shuffle(min(N, 50_000), seed=SEED, reshuffle_each_iteration=True)
    #ds = ds.batch(BATCH, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds

tfds_train = make_ds()


tfds_train.save(str(SAVE_PATH))
print("Saved to:", SAVE_PATH.resolve())

# %%
from pathlib import Path
import re

def _wfile(base_dir: Path, model_name: str, tag: str) -> Path:
    return Path(base_dir) / f"{model_name}_{tag}.weights.h5"

def save_ab_weights(model_a, model_b, base_dir: Path, tag: str) -> tuple[Path, Path]:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    a_path = _wfile(base_dir, "model_a", tag)
    b_path = _wfile(base_dir, "model_b", tag)

    # Guard: do not crash training/logging if model is not built
    if not getattr(model_a, "built", False) or not getattr(model_b, "built", False):
        print(f"[weights] skip save ({tag}): model_a.built={getattr(model_a,'built',None)}, model_b.built={getattr(model_b,'built',None)}")
        return None, None
    
    try:
        model_a.save_weights(str(a_path))
        model_b.save_weights(str(b_path))
        print(f"[weights] saved: {a_path.name}, {b_path.name}")
        return a_path, b_path
    except Exception as e:
        print(f"[weights] save failed ({tag}): {e}")
        return None, None

def _latest_epoch_pair(base_dir: Path) -> tuple[Path | None, Path | None]:
    base_dir = Path(base_dir)
    a_files = sorted(base_dir.glob("model_a_epoch_*.weights.h5"))
    b_files = sorted(base_dir.glob("model_b_epoch_*.weights.h5"))

    if not a_files or not b_files:
        return None, None

    def ep(p: Path) -> int:
        m = re.search(r"_epoch_(\d+)\.weights\.h5$", p.name)
        return int(m.group(1)) if m else -1

    a_by_ep = {ep(p): p for p in a_files}
    b_by_ep = {ep(p): p for p in b_files}
    common = sorted(set(a_by_ep.keys()) & set(b_by_ep.keys()))
    if not common:
        return None, None
    e = common[-1]
    return a_by_ep[e], b_by_ep[e]

def load_ab_weights(model_a, model_b, a_path: Path | None, b_path: Path | None) -> bool:
    if a_path is None or b_path is None:
        print("[weights] skip load: both A and B paths are required.")
        return False
    a_path = Path(a_path)
    b_path = Path(b_path)
    if not a_path.exists() or not b_path.exists():
        print(f"[weights] skip load: missing file(s): {a_path}, {b_path}")
        return False

    model_a.load_weights(str(a_path))
    model_b.load_weights(str(b_path))
    print(f"[weights] loaded: {a_path.name}, {b_path.name}")
    return True

# %%
import pickle
from datetime import datetime

def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _iso_mtime(path: Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")

def init_run_log(log_dir: Path, load_event: dict) -> tuple[dict, Path]:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    run_id = _now_tag()
    log_path = log_dir / f"train_run_{run_id}.pkl"

    run_log = {
        "meta": {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "start_mode": load_event.get("start_mode", "fresh"),   # "fresh" | "loaded"
            "loaded_from": load_event.get("loaded_from", None),    # dict or None
        },
        "epochs": []
    }
    return run_log, log_path

def append_epoch_log(run_log: dict, epoch_row: dict) -> None:
    run_log["epochs"].append(epoch_row)

def flush_run_log(run_log: dict, log_path: Path) -> None:
    log_path = Path(log_path)
    tmp = log_path.with_suffix(log_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(run_log, f)
    tmp.replace(log_path)  # atomic on same filesystem
    print(f"[log] flushed -> {log_path}")

# %%
# -----------------------------
# Instantiate internal models A and B
# -----------------------------
# These constructors should match your existing FINAL notebooks.
# If your internal models require additional args (depth, n2, hidden_units, etc.),
# set them here accordingly.



print("Composing internal models ...")
model_a = PyrInternalModelA(
    LAYOUT,
    sr_mode="replay",
    p_high=P_HIGH,
    beta=BETA_INPUT,
    alpha=5.0,
    seed=SEED+10
)
opt_a = tf.keras.optimizers.Adam(learning_rate=LR)

model_b = PyrInternalModelB(
    LAYOUT,
    sr_mode="replay",
    p_high=P_HIGH,
    beta=BETA_INPUT,
    alpha=5.0
)
opt_b = tf.keras.optimizers.Adam(learning_rate=LR)

# Force variable creation (build) with one dummy forward pass.
_dummy_field = tf.zeros((1, N2), tf.float32)
_dummy_gun   = tf.zeros((1, N2), tf.float32)
_dummy_out_list0 = [tf.zeros((1, kd), tf.float32) for kd in (N2 // (2 ** d) // 2 for d in range(DEPTH))]
comm_logits0, meas_list0, out_list0 = model_a.compute_with_internal(field_logits = _dummy_field, 
                                                                    replay_out_a_logits_list = _dummy_out_list0,
                                                                    training=False)
# ModelB signature varies across versions; we assume compute_with_internal returns tuple with shoot_logit first.
b_out = model_b.compute_with_internal(gun_logits=_dummy_gun, 
                                      comm_in_logits=comm_logits0, 
                                      prev_meas_list=meas_list0, 
                                      prev_out_list=out_list0, 
                                      training=False)
shoot_logit0 = b_out[0] if isinstance(b_out, (tuple, list)) else b_out

# 1) Ensure model vars exist (dummy forward already done)
vars_a = model_a.trainable_variables
vars_b = model_b.trainable_variables

# 2) Force optimizers to create slot variables OUTSIDE tf.function
opt_a.build(vars_a)
opt_b.build(vars_b)

print("Built. A comm:", comm_logits0.shape, "B shoot:", shoot_logit0.shape)

RUN_LOAD_EVENT = {"start_mode": "fresh", "loaded_from": None}

# Optional restore (weights only)
if LOAD_WEIGHTS_ON_START:
    a_in = Path(MODEL_A_WEIGHTS_IN) if MODEL_A_WEIGHTS_IN else (_wfile(CHECKPOINT_DIR, "model_a", "latest"))
    b_in = Path(MODEL_B_WEIGHTS_IN) if MODEL_B_WEIGHTS_IN else (_wfile(CHECKPOINT_DIR, "model_b", "latest"))

    loaded = load_ab_weights(model_a, model_b, a_in, b_in)
    used_a, used_b = (a_in, b_in) if loaded else (None, None)

    if not loaded:
        a_auto, b_auto = _latest_epoch_pair(CHECKPOINT_DIR)
        loaded = load_ab_weights(model_a, model_b, a_auto, b_auto)
        if loaded:
            used_a, used_b = a_auto, b_auto

    if loaded:
        RUN_LOAD_EVENT = {
            "start_mode": "loaded",
            "loaded_from": {
                "model_a_path": str(used_a),
                "model_b_path": str(used_b),
                "model_a_mtime": _iso_mtime(used_a),
                "model_b_mtime": _iso_mtime(used_b),
            },
        }



# %%
# -----------------------------
# Loss helpers (logit-space)
# -----------------------------
def logits_l2_reg(*tensors, weight: float = 1e-4) -> tf.Tensor:
    """
    L2 regularizer for logits.

    Computes:
        weight * sum_i mean(square(tensors[i]))

    Notes:
      - Each tensor contributes equally after its own mean reduction.
      - Total regularization scale increases with the number of tensors provided.
      - Returns a scalar float32 tensor.
    """
    reg = 0.0
    for t in tensors:
        t = tf.cast(t, tf.float32)
        reg += tf.reduce_mean(tf.square(t))
    return tf.cast(weight, tf.float32) * reg

def bce_from_logits(tgt_logits: tf.Tensor, pred_logits: tf.Tensor) -> tf.Tensor:
    """
    Binary cross-entropy in logit space.

    Target semantics:
      - Convert target logits to hard bits by sign:
            target_bit = 1.0 if tgt_logit >= 0.0 else 0.0
      - Note: exactly 0.0 maps to class 1.

    Args:
        tgt_logits: Tensor of target logits, any shape broadcast-compatible with pred_logits.
        pred_logits: Tensor of predicted logits, same shape as tgt_logits (or broadcast-compatible).

    Returns:
        Scalar mean BCE over all elements.
    """
    tgt_bits = tf.cast(tgt_logits >= 0.0, tf.float32)
    per_ex = tf.nn.sigmoid_cross_entropy_with_logits(labels=tgt_bits, logits=pred_logits)
    return tf.reduce_mean(per_ex)

def margin_loss_from_logits(tgt_logits: tf.Tensor,
                            pred_logits: tf.Tensor,
                            margin: float = 1.0) -> tf.Tensor:
    """
    Encourages pred_logits to have:
      - correct sign
      - magnitude >= margin
    """
    tgt_sign = tf.where(tgt_logits >= 0.0, 1.0, -1.0)
    loss = tf.maximum(0.0, margin - tgt_sign * pred_logits)
    return tf.reduce_mean(loss)

def weighted_per_level_bce(tgt_list, pred_list, weights: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    tgt_list/pred_list: Python list length L, each (B, D_i) or (B,1)
    weights: shape (L,), float32

    Returns: (loss_scalar, per_level_losses shape (L,))
    """
    assert len(tgt_list) == len(pred_list)
    L = len(tgt_list)
    weights = tf.cast(weights, tf.float32)
    if weights.shape.rank != 1 or int(weights.shape[0]) != L:
        raise ValueError(f"weights must be shape ({L},), got {weights.shape}")

    per_level = []
    for d in range(L):
        per_level.append(bce_from_logits(tgt_list[d], pred_list[d]))
        #per_level.append(margin_loss_from_logits(tgt_list[d], pred_list[d]))
    per_level = tf.stack(per_level, axis=0)  # (L,)
    loss = tf.reduce_sum(per_level * weights) / (tf.reduce_sum(weights) + 1e-8)
    return loss, per_level

def harden_ste(x: tf.Tensor, beta: float = 10.0) -> tf.Tensor:
    """
    Straight-through hardening to +/- beta.

    Forward pass:
        y = +beta where x >= 0, else -beta
    Backward pass (STE):
        dy/dx is approximated as identity (same gradient as y=x).

    This keeps discrete-like activations at interfaces while preserving gradient flow.
    """
    hard = tf.where(x >= 0.0, beta, -beta)
    return x + tf.stop_gradient(hard - x)

def flip_loss_from_logits_lists(
    tgt_logits_list,         # list/tuple length (DEPTH+1), each (B,1) logits
    pred_logits_list,        # list/tuple length (DEPTH+1), each (B,1) logits
    flip_weights: tf.Tensor, # shape (DEPTH,), float32
    eps: float = 1e-6,
    ) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Differentiable XOR-transition loss on ordered level logits.

    Let L = len(tgt_logits_list) = len(pred_logits_list), with L >= 2.
    There are (L - 1) transitions:
        transition d compares level d and d+1, for d=0..L-2.

    Target flip bit at transition d:
        tgt_flip[d] = XOR(bit_d, bit_{d+1}),
        bit_k = 1 if tgt_logit_k >= 0 else 0.

    Predicted flip logit at transition d:
        p = sigmoid(pred_logit_d), q = sigmoid(pred_logit_{d+1})
        P(XOR=1) = p + q - 2pq
        pred_flip_logit = logit(P(XOR=1)) after clipping to [eps, 1-eps].

    Args:
        flip_weights: shape (L-1,), one weight per transition.

    Returns:
        flip_loss: weighted mean scalar over transitions.
        flip_per:  shape (L-1,), unweighted per-transition BCE.
    """
    if len(tgt_logits_list) != len(pred_logits_list):
        raise ValueError(f"tgt/pred length mismatch: {len(tgt_logits_list)} vs {len(pred_logits_list)}")

    L = len(tgt_logits_list)
    if L < 2:
        raise ValueError("Need at least 2 levels to define flips.")
    DEPTH = L - 1

    flip_weights = tf.cast(flip_weights, tf.float32)
    if flip_weights.shape.rank != 1 or int(flip_weights.shape[0]) != DEPTH:
        raise ValueError(f"flip_weights must be shape ({DEPTH},), got {flip_weights.shape}")

    # ---- target flip bits (hard XOR from sign of target logits) ----
    tgt_bits = [tf.cast(t >= 0.0, tf.float32) for t in tgt_logits_list]
    tgt_flip_bits = [
        tf.math.floormod(tgt_bits[d] + tgt_bits[d + 1], 2.0)  # XOR in {0,1}
        for d in range(DEPTH)
    ]

    # ---- predicted flip logits (soft XOR computed from predicted logits) ----
    def xor_logit(a_logits, b_logits):
        p = tf.sigmoid(a_logits)
        q = tf.sigmoid(b_logits)
        pxor = p + q - 2.0 * p * q
        pxor = tf.clip_by_value(pxor, eps, 1.0 - eps)
        return tf.math.log(pxor) - tf.math.log1p(-pxor)

    per = []
    for d in range(DEPTH):
        flip_logits = xor_logit(pred_logits_list[d], pred_logits_list[d + 1])
        per_ex = tf.nn.sigmoid_cross_entropy_with_logits(labels=tgt_flip_bits[d], logits=flip_logits)
        per.append(tf.reduce_mean(per_ex))

    flip_per = tf.stack(per, axis=0)  # (DEPTH,)

    flip_loss = tf.reduce_sum(flip_per * flip_weights) / (tf.reduce_sum(flip_weights) + 1e-8)
    return flip_loss, flip_per



# %%
# -----------------------------
# Training step (joint A+B)
# -----------------------------
@tf.function
def train_step(field_logits, gun_logits, shoot_tgt_logits, meas_in_a_tgt, meas_out_a_tgt, comms_tgt, weights):
    """
    meas_in_a_tgt: tuple length DEPTH of (B,k_d) logits
    meas_out_a_tgt: tuple length DEPTH of (B,k_d) logits
    comms_tgt: tuple length DEPTH+1 of (B,1) logits
    """
    W_MEAS_IN_A = weights["W_MEAS_IN_A"]
    W_COMMS_B = weights["W_COMMS_B"]
    W_SHOOT = weights["W_SHOOT"]
    W_COMMS_A = weights["W_COMMS_A"]
    TRAINING_MODE = weights["TRAINING_MODE"]
    if not TRAINING_MODE in ("A", "A_SHOOT", "B", "AB"):
        raise ValueError(f"Invalid TRAINING_MODE {TRAINING_MODE}, expected 'A', 'A_SHOOT', 'B', or 'AB'")
    if len(W_COMMS_B) != DEPTH-1:
        raise ValueError(f"W_COMMS_B must be length DEPTH-1, got {len(W_COMMS_B)}")
    
    with tf.GradientTape() as tape:
        # ---- Forward A ----
        comm_logits, meas_list, out_list = model_a.compute_with_internal(
            field_logits = field_logits, 
            replay_out_a_logits_list = meas_out_a_tgt,
            harden_between_levels= False,
            training=True,
        )

        # ---- A auxiliary losses (meas) ----
        meas_in_a_loss, meas_in_a_per = weighted_per_level_bce(list(meas_in_a_tgt), list(meas_list), W_MEAS_IN_A)
        #meas_out_a_loss, meas_out_a_per = weighted_per_level_bce(list(meas_out_a_tgt), list(out_list), W_MEAS_OUT_A)
        
        # ---- A comm out loss (comm_logits) ----
        comm_A_bits = tf.cast(comms_tgt[0]  >= 0.0, tf.float32)  # from sign
        comm_A_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=comm_A_bits, logits=comm_logits)
        )

        BETA = BETA_INPUT  # match your target logit magnitude (often 10 in your datasets)

        if TRAINING_MODE == "A":
            comm_logits_for_b = tf.stop_gradient(comm_logits)
            meas_list_for_b = [tf.stop_gradient(t) for t in meas_list]
            out_list_for_b  = [tf.stop_gradient(t) for t in out_list]
        elif TRAINING_MODE == "B":
            # comm_logits_for_b = tf.stop_gradient(comm_logits)
            # meas_list_for_b = [tf.stop_gradient(t) for t in meas_list]
            # out_list_for_b  = [tf.stop_gradient(t) for t in out_list]
            comm_logits_for_b = harden_ste(comm_logits, beta=BETA)                        
            meas_list_for_b = [tf.stop_gradient(t) for t in meas_list]
            out_list_for_b  = [tf.stop_gradient(t) for t in out_list]
            # meas_list_for_b   = [harden_ste(t, beta=BETA) for t in meas_list]
            # out_list_for_b    = [harden_ste(t, beta=BETA) for t in out_list]
        elif TRAINING_MODE == "A_SHOOT":
            # B frozen by var selection, but allow gradients into A through B
            # Use STE to keep the interface distribution stable for B
            comm_logits_for_b = harden_ste(comm_logits, beta=BETA)
            meas_list_for_b   = [harden_ste(t, beta=BETA) for t in meas_list]
            out_list_for_b    = [harden_ste(t, beta=BETA) for t in out_list]
        else:  # "AB"
            # End-to-end, but keep interface distribution stable
            comm_logits_for_b = harden_ste(comm_logits, beta=BETA)
            meas_list_for_b   = [harden_ste(t, beta=BETA) for t in meas_list]
            out_list_for_b    = [harden_ste(t, beta=BETA) for t in out_list]

        # ---- Forward B ----
        b_out = model_b.compute_with_internal(
            gun_logits,
            comm_logits_for_b,
            list(meas_list_for_b),
            list(out_list_for_b),
            harden_between_levels= False,
            training=True,
        )
        shoot_logit, meas_b_logits_list, out_b_logits_list, comms_logits_list, gun_logits_list = b_out

        # ---- Main shoot loss ----
        shoot_bits = tf.cast(shoot_tgt_logits >= 0.0, tf.float32)  # from sign
        shoot_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=shoot_bits, logits=shoot_logit)
        )
        
        # ---- B comm loss ----
        # Comms as returned from model_b.compute_with_internal is a list of logits length DEPTH+1.
        # The first element corresponds to the comms output of model_a, and is not trainable by model_b.
        # The last element of comms is the final shoot logit.
        # For losses, we already have a loss on comms from A (comm_A_loss), and a loss on shoot (shoot_loss), so
        # the only additional comms losses we add here are on the internal transitions of B, i.e. levels 1..DEPTH-1.
        # This is conscious and not a bug.
        comms_pred_list_123 = comms_logits_list[1:DEPTH]
        comms_tgt_list_123 = comms_tgt[1:DEPTH]
        W_COMMS_B_123 = W_COMMS_B

        comms_b_loss, comms_b_per = weighted_per_level_bce(list(comms_tgt_list_123), list(comms_pred_list_123), W_COMMS_B_123)

        # ---- B flip loss on comms ----
        W_FLIP = tf.constant([1.0, 0.0, 0.0, 0.0], tf.float32)
        flip_loss, flip_per = flip_loss_from_logits_lists(
            comms_tgt,
            comms_logits_list,
            flip_weights=W_FLIP,  # only DEPTH-1 transitions
        )
        # # Example: regularize B outputs + A outputs
        # reg = logits_l2_reg(
        #     comm_logits,                          # A comm0
        #     *meas_list, *out_list,                # A meas/out lists
        #     shoot_logit,                          # B shoot
        #     *meas_b_logits_list, *out_b_logits_list,
        #     *comms_logits_list, *gun_logits_list, # B internal traces if you want
        #     weight=1e-4,
        # )
        total = (W_SHOOT * shoot_loss) + (W_COMMS_A * comm_A_loss) + meas_in_a_loss + comms_b_loss

        # --- select vars once ---
        vars_a = model_a.trainable_variables
        vars_b = model_b.trainable_variables
        vars_all = vars_a + vars_b

        # --- select total based on mode (still inside the tape!) ---
        if TRAINING_MODE == "A":
            total = (W_COMMS_A * comm_A_loss) + meas_in_a_loss
        elif TRAINING_MODE == "A_SHOOT":
            total = (W_SHOOT * shoot_loss) + (W_COMMS_A * comm_A_loss) + meas_in_a_loss
        elif TRAINING_MODE == "B":
            total = (W_SHOOT * shoot_loss) + comms_b_loss
        else:  # "AB"
            total = (W_SHOOT * shoot_loss) + (W_COMMS_A * comm_A_loss) + meas_in_a_loss + comms_b_loss
        #total = (W_SHOOT * shoot_loss) + (W_COMMS_A * comm_A_loss) + meas_in_a_loss + comms_b_loss

        # --- ONE gradient call for all vars ---
        grads_all = tape.gradient(total, vars_all)

        # --- split grads back to A/B ---
        na = len(vars_a)
        grads_a = grads_all[:na]
        grads_b = grads_all[na:]

        # --- apply depending on mode ---
        pairs_a = [(g, v) for g, v in zip(grads_a, vars_a) if g is not None]
        pairs_b = [(g, v) for g, v in zip(grads_b, vars_b) if g is not None]

        if TRAINING_MODE in ("A", "A_SHOOT", "AB"):
            if pairs_a:
                opt_a.apply_gradients(pairs_a)

        if TRAINING_MODE in ("B", "AB"):
            if pairs_b:
                opt_b.apply_gradients(pairs_b)
        # if TRAINING_MODE == "A":
        #     total = (W_COMMS_A * comm_A_loss) + meas_in_a_loss 
        #     vars_all = model_a.trainable_variables
        #     opt_used = opt_a
        #     grads = tape.gradient(total, vars_all)
        #     pairs = [(g, v) for g, v in zip(grads, vars_all) if g is not None]
        #     opt_used.apply_gradients(pairs)
        # elif TRAINING_MODE == "B":
        #     total = (W_SHOOT * shoot_loss) + comms_b_loss
        #     vars_all = model_b.trainable_variables
        #     opt_used = opt_b
        #     grads = tape.gradient(total, vars_all)
        #     pairs = [(g, v) for g, v in zip(grads, vars_all) if g is not None]
        #     opt_used.apply_gradients(pairs)
        # else:
        #     total = (W_SHOOT * shoot_loss) + (W_COMMS_A * comm_A_loss) + meas_in_a_loss + comms_b_loss
        #     vars_all = model_a.trainable_variables + model_b.trainable_variables
        #     vars_a = model_a.trainable_variables
        #     vars_b = model_b.trainable_variables

        #     grads_a = tape.gradient(total, vars_a)
        #     grads_b = tape.gradient(total, vars_b)

        #     pairs_a = [(g, v) for g, v in zip(grads_a, vars_a) if g is not None]
        #     pairs_b = [(g, v) for g, v in zip(grads_b, vars_b) if g is not None]

        #     opt_a.apply_gradients(pairs_a)
        #     opt_b.apply_gradients(pairs_b)
        

    # simple metrics
    shoot_pred_bits = tf.cast(shoot_logit >= 0.0, tf.float32)
    shoot_acc = tf.reduce_mean(tf.cast(tf.equal(shoot_pred_bits, shoot_bits), tf.float32))

    return {
        "total": total,
        "shoot_loss": shoot_loss,
        "shoot_acc": shoot_acc,
        "meas_in_a_loss": meas_in_a_loss,
        "meas_in_a_per": meas_in_a_per,
        "comm_a_loss": comm_A_loss,
        "comms_b_loss": comms_b_loss,
        "comms_b_per": comms_b_per,
        "flip_loss": flip_loss,
        "flip_per": flip_per,
        "reg": 0.0
    }

# %%
# -----------------------------
# Load dataset
# -----------------------------
loaded_raw = tf.data.Dataset.load(str(SAVE_PATH))  # element_spec restored from metadata
tfds_train = (
    loaded_raw
    .shuffle(min(N, 50_000), seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

# Create one log file for this training run
RUN_LOG, RUN_LOG_PATH = init_run_log(LOG_DIR, RUN_LOAD_EVENT)
print(f"[log] run file: {RUN_LOG_PATH}")



# -----------------------------
# Training loop
# -----------------------------
EPOCHS = 500
m_cb1 = tf.keras.metrics.Mean(name="cb_l1")
m_cb2 = tf.keras.metrics.Mean(name="cb_l2")
m_cb3 = tf.keras.metrics.Mean(name="cb_l3")
m_flip1 = tf.keras.metrics.Mean(name="flip_l1")
m_flip2 = tf.keras.metrics.Mean(name="flip_l2")
m_flip3 = tf.keras.metrics.Mean(name="flip_l3")
m_flip4 = tf.keras.metrics.Mean(name="flip_l4")
# ------- PHASES -------
train_a_alone_epochs = 100
train_b_alone_epochs_l1 = 50
train_b_alone_epochs_l2 = 50
train_b_alone_epochs_l3 = 50
train_b_alone_shoot = 50
epoch = 0

interrupted = False
try:
    while epoch < EPOCHS:
        m_total = tf.keras.metrics.Mean()
        m_shoot = tf.keras.metrics.Mean()
        m_comm_a = tf.keras.metrics.Mean()
        m_acc   = tf.keras.metrics.Mean()
        m_comms_b     = tf.keras.metrics.Mean()
        m_mi    = tf.keras.metrics.Mean()
        m_reg   = tf.keras.metrics.Mean()
        m_cb1.reset_state(); m_cb2.reset_state(); m_cb3.reset_state()
        m_flip1.reset_state(); m_flip2.reset_state(); m_flip3.reset_state(); m_flip4.reset_state()
        
        if epoch < train_a_alone_epochs:
            weights = {
                "W_MEAS_IN_A": tf.constant([1.0, 1.0, 1.0, 1.0], tf.float32),
                "W_MEAS_OUT_A": tf.constant([0.0, 0.0, 0.0, 0.0], tf.float32),
                "W_COMMS_B": tf.constant([0.0, 0.0, 0.0], tf.float32),
                "W_SHOOT": 0.0,
                "W_COMMS_A": 1.0, 
                "TRAINING_MODE": "A" # "A", "B", or "AB" for your internal use if you want to condition behavior on it
                }
        elif epoch < train_a_alone_epochs + train_b_alone_epochs_l1:
            weights = {
                "W_MEAS_IN_A": tf.constant([0.0, 1.0, 1.0, 1.0], tf.float32),
                "W_MEAS_OUT_A": tf.constant([0.0, 0.0, 0.0, 0.0], tf.float32),
                "W_COMMS_B": tf.constant([1.0, 0.0, 0.0], tf.float32),
                "W_SHOOT": 0.0,
                "W_COMMS_A": 0.0, 
                "TRAINING_MODE": "B" # "A", "B", or "AB" for your internal use if you want to condition behavior on it
                }
        elif epoch < train_a_alone_epochs + train_b_alone_epochs_l1 + train_b_alone_epochs_l2:
            weights = {
                "W_MEAS_IN_A": tf.constant([0.0, 0.0, 1.0, 1.0], tf.float32),
                "W_MEAS_OUT_A": tf.constant([0.0, 0.0, 0.0, 0.0], tf.float32),
                "W_COMMS_B": tf.constant([1.0, 1.0, 0.0], tf.float32),
                "W_SHOOT": 0.0,
                "W_COMMS_A": 0.0, 
                "TRAINING_MODE": "B" # "A", "B", or "AB" for your internal use if you want to condition behavior on it
                }
        elif epoch < train_a_alone_epochs + train_b_alone_epochs_l1 + train_b_alone_epochs_l2 + train_b_alone_epochs_l3:
            weights = {
                "W_MEAS_IN_A": tf.constant([0.0, 0.0, 0.0, 1.0], tf.float32),
                "W_MEAS_OUT_A": tf.constant([0.0, 0.0, 0.0, 0.0], tf.float32),
                "W_COMMS_B": tf.constant([1.0, 1.0, 1.0], tf.float32),
                "W_SHOOT": 0.0,
                "W_COMMS_A": 0.0, 
                "TRAINING_MODE": "B" # "A", "B", or "AB" for your internal use if you want to condition behavior on it
                }
        elif epoch < train_a_alone_epochs + train_b_alone_epochs_l1 + train_b_alone_epochs_l2 + train_b_alone_epochs_l3 + train_b_alone_shoot:
            weights = {
                "W_MEAS_IN_A": tf.constant([0.0, 0.0, 0.0, 0.0], tf.float32),
                "W_MEAS_OUT_A": tf.constant([0.0, 0.0, 0.0, 0.0], tf.float32),
                "W_COMMS_B": tf.constant([1.0, 1.0, 1.0], tf.float32),
                "W_SHOOT": 1.0,
                "W_COMMS_A": 0.0, 
                "TRAINING_MODE": "B" # "A", "B", or "AB" for your internal use if you want to condition behavior on it
                }
        else:
            if epoch % 50 < 5:
                weights = {
                    "W_MEAS_IN_A": tf.constant([0.1, 0.1, 0.1, 0.1], tf.float32),
                    "W_MEAS_OUT_A": tf.constant([0.0, 0.0, 0.0, 0.0], tf.float32),
                    "W_COMMS_B": tf.constant([0.0, 0.0, 0.0], tf.float32),
                    "W_SHOOT": 1.0,
                    "W_COMMS_A": 0.1, 
                    "TRAINING_MODE": "A_SHOOT" # "A", "A_SHOOT", "B", or "AB" for your internal use if you want to condition behavior on it
                    }
            else:
                weights = {
                    "W_MEAS_IN_A": tf.constant([0.0, 0.0, 0.0, 0.0], tf.float32),
                    "W_MEAS_OUT_A": tf.constant([0.0, 0.0, 0.0, 0.0], tf.float32),
                    "W_COMMS_B": tf.constant([0.0, 0.0, 0.0], tf.float32),
                    "W_SHOOT": 1.0,
                    "W_COMMS_A": 0.0, 
                    "TRAINING_MODE": "B" # "A", "A_SHOOT", "B", or "AB" for your internal use if you want to condition behavior on it
                    }

        for (field_logits_b, gun_logits_b, shoot_tgt_logits_b, meas_in_a_tgt_b, meas_out_a_tgt_b, comms_tgt_b) in tfds_train:
            out = train_step(field_logits_b, gun_logits_b, shoot_tgt_logits_b, meas_in_a_tgt_b, meas_out_a_tgt_b, comms_tgt_b, weights)
            m_total.update_state(out["total"])
            m_shoot.update_state(out["shoot_loss"])
            m_acc.update_state(out["shoot_acc"])
            m_comm_a.update_state(out["comm_a_loss"])
            m_comms_b.update_state(out["comms_b_loss"])
            m_mi.update_state(out["meas_in_a_loss"])
            m_reg.update_state(out["reg"])
            m_cb1.update_state(out["comms_b_per"][0])
            m_cb2.update_state(out["comms_b_per"][1])
            m_cb3.update_state(out["comms_b_per"][2])
            m_flip1.update_state(out["flip_per"][0])
            m_flip2.update_state(out["flip_per"][1])
            m_flip3.update_state(out["flip_per"][2])
            m_flip4.update_state(out["flip_per"][3])

    
        print(
            f"epoch {epoch:03d}  total={m_total.result().numpy():.4f}  "
            f"shoot_loss={m_shoot.result().numpy():.4f}  shoot_acc={m_acc.result().numpy():.4f}  "
            f"comm_a={m_comm_a.result().numpy():.4f}  "
            f"comms_b={m_comms_b.result().numpy():.4f}  "
            f"cb_l1={m_cb1.result().numpy():.4f} cb_l2={m_cb2.result().numpy():.4f} cb_l3={m_cb3.result().numpy():.4f}"
            f"meas_in_a={m_mi.result().numpy():.4f}  "
            f"flip_l1={m_flip1.result().numpy():.4f} flip_l2={m_flip2.result().numpy():.4f} flip_l3={m_flip3.result().numpy():.4f} flip_l4={m_flip4.result().numpy():.4f}  "
            f"reg={m_reg.result().numpy():.4f}"
        )

# after metrics are computed/printed for this epoch:
        epoch_row = {
            "epoch": int(epoch),
            "total": float(m_total.result().numpy()),
            "shoot_loss": float(m_shoot.result().numpy()),
            "shoot_acc": float(m_acc.result().numpy()),
            "comm_a": float(m_comm_a.result().numpy()),
            "comms_b": float(m_comms_b.result().numpy()),
            "meas_in_a": float(m_mi.result().numpy()),
            "cb_l1": float(m_cb1.result().numpy()),
            "cb_l2": float(m_cb2.result().numpy()),
            "cb_l3": float(m_cb3.result().numpy()),
            "flip_l1": float(m_flip1.result().numpy()),
            "flip_l2": float(m_flip2.result().numpy()),
            "flip_l3": float(m_flip3.result().numpy()),
            "flip_l4": float(m_flip4.result().numpy()),
            "reg": float(m_reg.result().numpy()),
            "training_mode": str(weights["TRAINING_MODE"]),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        append_epoch_log(RUN_LOG, epoch_row)

        # optional periodic flush
        if LOG_FLUSH_EVERY_EPOCHS and ((epoch + 1) % int(LOG_FLUSH_EVERY_EPOCHS) == 0):
            flush_run_log(RUN_LOG, RUN_LOG_PATH)

        # periodic checkpoint (+ flush log at same moment)
        if SAVE_WEIGHTS_EVERY and ((epoch + 1) % int(SAVE_WEIGHTS_EVERY) == 0):
            save_ab_weights(model_a, model_b, CHECKPOINT_DIR, tag=f"epoch_{epoch+1:04d}")
            save_ab_weights(model_a, model_b, CHECKPOINT_DIR, tag="latest")
            flush_run_log(RUN_LOG, RUN_LOG_PATH)

        if epoch < train_a_alone_epochs and m_comm_a.result().numpy() < 0.0001 and m_mi.result().numpy() < 0.0001:
            print("Early stopping condition met for A initial training.")
            epoch = train_a_alone_epochs

        epoch += 1
        
except KeyboardInterrupt:
    interrupted = True
    print("[train] interrupted by user.")

finally:
    # always persist log (also covers manual interrupt)
    flush_run_log(RUN_LOG, RUN_LOG_PATH)

    if SAVE_WEIGHTS_AT_END:
        end_tag = f"epoch_{epoch:04d}_{'interrupted' if interrupted else 'final'}"
        try:
            save_ab_weights(model_a, model_b, CHECKPOINT_DIR, tag=end_tag)
            save_ab_weights(model_a, model_b, CHECKPOINT_DIR, tag="latest")
        except Exception as e:
            print(f"[weights] end-save skipped due to error: {e}")

# %% [markdown]
# epoch 001  total=1.2897  shoot_loss=0.9067  shoot_acc=0.4996  comm_a=0.6719  comms_b=0.0000  meas_in_a=0.6178  reg=0.0000
# epoch 002  total=0.4907  shoot_loss=0.9113  shoot_acc=0.4995  comm_a=0.0019  comms_b=0.0000  meas_in_a=0.4888  reg=0.0000
# epoch 003  total=0.3595  shoot_loss=0.9312  shoot_acc=0.4997  comm_a=0.0004  comms_b=0.0000  meas_in_a=0.3592  reg=0.0000
# epoch 004  total=0.2637  shoot_loss=0.9352  shoot_acc=0.4996  comm_a=0.0002  comms_b=0.0000  meas_in_a=0.2636  reg=0.0000
# epoch 005  total=0.2608  shoot_loss=0.9364  shoot_acc=0.4996  comm_a=0.0001  comms_b=0.0000  meas_in_a=0.2607  reg=0.0000
# epoch 006  total=0.2606  shoot_loss=0.9365  shoot_acc=0.4995  comm_a=0.0001  comms_b=0.0000  meas_in_a=0.2605  reg=0.0000
# epoch 007  total=0.1278  shoot_loss=0.9577  shoot_acc=0.5000  comm_a=0.0001  comms_b=0.0000  meas_in_a=0.1278  reg=0.0000
# epoch 008  total=0.0174  shoot_loss=0.9712  shoot_acc=0.4998  comm_a=0.0002  comms_b=0.0000  meas_in_a=0.0172  reg=0.0000
# epoch 009  total=0.0004  shoot_loss=0.9715  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0004  reg=0.0000
# epoch 010  total=0.0001  shoot_loss=0.9717  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0001  reg=0.0000
# epoch 011  total=0.0001  shoot_loss=0.9717  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0001  reg=0.0000
# epoch 012  total=0.0000  shoot_loss=0.9716  shoot_acc=0.5001  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 013  total=0.0000  shoot_loss=0.9716  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 014  total=0.0000  shoot_loss=0.9718  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 015  total=0.0000  shoot_loss=0.9718  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 016  total=0.0000  shoot_loss=0.9719  shoot_acc=0.4998  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 017  total=0.0000  shoot_loss=0.9719  shoot_acc=0.4997  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 018  total=0.0000  shoot_loss=0.9719  shoot_acc=0.4998  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 019  total=0.0000  shoot_loss=0.9720  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 020  total=0.0000  shoot_loss=0.9721  shoot_acc=0.4997  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 021  total=0.0000  shoot_loss=0.9719  shoot_acc=0.4998  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 022  total=0.0000  shoot_loss=0.9722  shoot_acc=0.4998  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 023  total=0.0000  shoot_loss=0.9724  shoot_acc=0.4998  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 024  total=0.0000  shoot_loss=0.9724  shoot_acc=0.4998  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 025  total=0.0000  shoot_loss=0.9723  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 026  total=0.0000  shoot_loss=0.9724  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 027  total=0.0000  shoot_loss=0.9726  shoot_acc=0.4998  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 028  total=0.0000  shoot_loss=0.9725  shoot_acc=0.4998  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 029  total=0.0000  shoot_loss=0.9723  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 030  total=0.0000  shoot_loss=0.9723  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 031  total=0.0000  shoot_loss=0.9722  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 032  total=0.0000  shoot_loss=0.9723  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 033  total=0.0000  shoot_loss=0.9721  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 034  total=0.0000  shoot_loss=0.9722  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 035  total=0.0000  shoot_loss=0.9724  shoot_acc=0.4997  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 036  total=0.0000  shoot_loss=0.9720  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 037  total=0.0000  shoot_loss=0.9721  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 038  total=0.0000  shoot_loss=0.9720  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 039  total=0.0000  shoot_loss=0.9721  shoot_acc=0.4998  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 040  total=0.0000  shoot_loss=0.9720  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 041  total=0.0000  shoot_loss=0.9720  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 042  total=0.0000  shoot_loss=0.9719  shoot_acc=0.5001  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 043  total=0.0000  shoot_loss=0.9719  shoot_acc=0.5001  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 044  total=0.0000  shoot_loss=0.9717  shoot_acc=0.5002  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 045  total=0.0000  shoot_loss=0.9717  shoot_acc=0.5002  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 046  total=0.0000  shoot_loss=0.9719  shoot_acc=0.5001  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 047  total=0.0000  shoot_loss=0.9719  shoot_acc=0.5001  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 048  total=0.0000  shoot_loss=0.9717  shoot_acc=0.5001  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 049  total=0.0000  shoot_loss=0.9717  shoot_acc=0.5001  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 050  total=0.0000  shoot_loss=0.9718  shoot_acc=0.5001  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 051  total=0.0000  shoot_loss=0.9718  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 052  total=0.0000  shoot_loss=0.9717  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 053  total=0.0000  shoot_loss=0.9718  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 054  total=0.0000  shoot_loss=0.9719  shoot_acc=0.4998  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 055  total=0.0000  shoot_loss=0.9719  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 056  total=0.0000  shoot_loss=0.9718  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 057  total=0.0000  shoot_loss=0.9717  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 058  total=0.0000  shoot_loss=0.9717  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 059  total=0.0000  shoot_loss=0.9717  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 060  total=0.0000  shoot_loss=0.9715  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 061  total=0.0000  shoot_loss=0.9717  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 062  total=0.0000  shoot_loss=0.9716  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 063  total=0.0000  shoot_loss=0.9717  shoot_acc=0.4998  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 064  total=0.0000  shoot_loss=0.9716  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 065  total=0.0000  shoot_loss=0.9716  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 066  total=0.0000  shoot_loss=0.9716  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 067  total=0.0000  shoot_loss=0.9717  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 068  total=0.0000  shoot_loss=0.9716  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 069  total=0.0000  shoot_loss=0.9715  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 070  total=0.0000  shoot_loss=0.9715  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 071  total=0.0000  shoot_loss=0.9716  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 072  total=0.0000  shoot_loss=0.9715  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 073  total=0.0000  shoot_loss=0.9716  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 074  total=0.0000  shoot_loss=0.9716  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 075  total=0.0000  shoot_loss=0.9714  shoot_acc=0.5001  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 076  total=0.0000  shoot_loss=0.9715  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 077  total=0.0000  shoot_loss=0.9714  shoot_acc=0.5001  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 078  total=0.0000  shoot_loss=0.9717  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 079  total=0.0000  shoot_loss=0.9714  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 080  total=0.0000  shoot_loss=0.9715  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 081  total=0.0000  shoot_loss=0.9713  shoot_acc=0.5001  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 082  total=0.0000  shoot_loss=0.9716  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 083  total=0.0000  shoot_loss=0.9715  shoot_acc=0.5001  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 084  total=0.0000  shoot_loss=0.9714  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 085  total=0.0000  shoot_loss=0.9716  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 086  total=0.0000  shoot_loss=0.9714  shoot_acc=0.5001  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 087  total=0.0000  shoot_loss=0.9714  shoot_acc=0.5001  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 088  total=0.0000  shoot_loss=0.9715  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 089  total=0.0000  shoot_loss=0.9714  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 090  total=0.0000  shoot_loss=0.9714  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 091  total=0.0000  shoot_loss=0.9715  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 092  total=0.0000  shoot_loss=0.9714  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 093  total=0.0000  shoot_loss=0.9715  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 094  total=0.0000  shoot_loss=0.9715  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 095  total=0.0000  shoot_loss=0.9715  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 096  total=0.0000  shoot_loss=0.9715  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 097  total=0.0000  shoot_loss=0.9714  shoot_acc=0.5001  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 098  total=0.0000  shoot_loss=0.9715  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 099  total=0.0000  shoot_loss=0.9715  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 100  total=0.6677  shoot_loss=0.9524  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.6677  meas_in_a=0.0000  reg=0.0000
# epoch 101  total=0.4118  shoot_loss=0.9477  shoot_acc=0.5007  comm_a=0.0000  comms_b=0.4118  meas_in_a=0.0000  reg=0.0000
# epoch 102  total=0.3180  shoot_loss=0.9487  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.3180  meas_in_a=0.0000  reg=0.0000
# epoch 103  total=0.2794  shoot_loss=0.9488  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.2794  meas_in_a=0.0000  reg=0.0000
# epoch 104  total=0.2617  shoot_loss=0.9498  shoot_acc=0.4999  comm_a=0.0000  comms_b=0.2617  meas_in_a=0.0000  reg=0.0000
# epoch 105  total=0.2533  shoot_loss=0.9507  shoot_acc=0.4998  comm_a=0.0000  comms_b=0.2533  meas_in_a=0.0000  reg=0.0000
# epoch 106  total=0.2411  shoot_loss=0.9510  shoot_acc=0.5000  comm_a=0.0000  comms_b=0.2411  meas_in_a=0.0000  reg=0.0000
# epoch 107  total=0.1709  shoot_loss=0.9505  shoot_acc=0.4994  comm_a=0.0000  comms_b=0.1709  meas_in_a=0.0000  reg=0.0000
# epoch 108  total=0.1571  shoot_loss=0.9518  shoot_acc=0.4991  comm_a=0.0000  comms_b=0.1571  meas_in_a=0.0000  reg=0.0000
# epoch 109  total=0.1506  shoot_loss=0.9525  shoot_acc=0.4985  comm_a=0.0000  comms_b=0.1506  meas_in_a=0.0000  reg=0.0000
# epoch 110  total=0.1482  shoot_loss=0.9519  shoot_acc=0.4988  comm_a=0.0000  comms_b=0.1482  meas_in_a=0.0000  reg=0.0000
# epoch 111  total=0.1164  shoot_loss=0.9514  shoot_acc=0.4990  comm_a=0.0000  comms_b=0.1164  meas_in_a=0.0000  reg=0.0000
# epoch 112  total=0.1025  shoot_loss=0.9507  shoot_acc=0.4995  comm_a=0.0000  comms_b=0.1025  meas_in_a=0.0000  reg=0.0000
# epoch 113  total=0.0984  shoot_loss=0.9511  shoot_acc=0.4996  comm_a=0.0000  comms_b=0.0984  meas_in_a=0.0000  reg=0.0000
# epoch 114  total=0.0973  shoot_loss=0.9522  shoot_acc=0.4990  comm_a=0.0000  comms_b=0.0973  meas_in_a=0.0000  reg=0.0000
# epoch 115  total=0.0965  shoot_loss=0.9529  shoot_acc=0.4985  comm_a=0.0000  comms_b=0.0965  meas_in_a=0.0000  reg=0.0000
# epoch 116  total=0.0961  shoot_loss=0.9522  shoot_acc=0.4989  comm_a=0.0000  comms_b=0.0961  meas_in_a=0.0000  reg=0.0000
# epoch 117  total=0.0955  shoot_loss=0.9520  shoot_acc=0.4986  comm_a=0.0000  comms_b=0.0955  meas_in_a=0.0000  reg=0.0000
# epoch 118  total=0.0836  shoot_loss=0.9530  shoot_acc=0.4983  comm_a=0.0000  comms_b=0.0836  meas_in_a=0.0000  reg=0.0000
# epoch 119  total=0.0508  shoot_loss=0.9553  shoot_acc=0.4983  comm_a=0.0000  comms_b=0.0508  meas_in_a=0.0000  reg=0.0000
# epoch 120  total=0.0487  shoot_loss=0.9559  shoot_acc=0.4985  comm_a=0.0000  comms_b=0.0487  meas_in_a=0.0000  reg=0.0000
# epoch 121  total=0.0479  shoot_loss=0.9551  shoot_acc=0.4980  comm_a=0.0000  comms_b=0.0479  meas_in_a=0.0000  reg=0.0000
# epoch 122  total=0.0476  shoot_loss=0.9553  shoot_acc=0.4982  comm_a=0.0000  comms_b=0.0476  meas_in_a=0.0000  reg=0.0000
# epoch 123  total=0.0476  shoot_loss=0.9552  shoot_acc=0.4980  comm_a=0.0000  comms_b=0.0476  meas_in_a=0.0000  reg=0.0000
# epoch 124  total=0.0467  shoot_loss=0.9552  shoot_acc=0.4979  comm_a=0.0000  comms_b=0.0467  meas_in_a=0.0000  reg=0.0000
# epoch 125  total=0.0465  shoot_loss=0.9546  shoot_acc=0.4983  comm_a=0.0000  comms_b=0.0465  meas_in_a=0.0000  reg=0.0000
# epoch 126  total=0.0463  shoot_loss=0.9545  shoot_acc=0.4982  comm_a=0.0000  comms_b=0.0463  meas_in_a=0.0000  reg=0.0000
# epoch 127  total=0.0463  shoot_loss=0.9546  shoot_acc=0.4981  comm_a=0.0000  comms_b=0.0463  meas_in_a=0.0000  reg=0.0000
# epoch 128  total=0.0473  shoot_loss=0.9538  shoot_acc=0.4982  comm_a=0.0000  comms_b=0.0473  meas_in_a=0.0000  reg=0.0000
# epoch 129  total=0.0461  shoot_loss=0.9533  shoot_acc=0.4978  comm_a=0.0000  comms_b=0.0461  meas_in_a=0.0000  reg=0.0000
# epoch 130  total=0.0458  shoot_loss=0.9526  shoot_acc=0.4982  comm_a=0.0000  comms_b=0.0458  meas_in_a=0.0000  reg=0.0000
# epoch 131  total=0.0462  shoot_loss=0.9524  shoot_acc=0.4983  comm_a=0.0000  comms_b=0.0462  meas_in_a=0.0000  reg=0.0000
# epoch 132  total=0.0461  shoot_loss=0.9515  shoot_acc=0.4985  comm_a=0.0000  comms_b=0.0461  meas_in_a=0.0000  reg=0.0000
# epoch 133  total=0.0461  shoot_loss=0.9511  shoot_acc=0.4983  comm_a=0.0000  comms_b=0.0461  meas_in_a=0.0000  reg=0.0000
# epoch 134  total=0.0453  shoot_loss=0.9515  shoot_acc=0.4980  comm_a=0.0000  comms_b=0.0453  meas_in_a=0.0000  reg=0.0000
# epoch 135  total=0.0456  shoot_loss=0.9519  shoot_acc=0.4984  comm_a=0.0000  comms_b=0.0456  meas_in_a=0.0000  reg=0.0000
# epoch 136  total=0.0455  shoot_loss=0.9522  shoot_acc=0.4978  comm_a=0.0000  comms_b=0.0455  meas_in_a=0.0000  reg=0.0000
# epoch 137  total=0.0457  shoot_loss=0.9516  shoot_acc=0.4979  comm_a=0.0000  comms_b=0.0457  meas_in_a=0.0000  reg=0.0000
# epoch 138  total=0.0455  shoot_loss=0.9516  shoot_acc=0.4980  comm_a=0.0000  comms_b=0.0455  meas_in_a=0.0000  reg=0.0000
# epoch 139  total=0.0456  shoot_loss=0.9521  shoot_acc=0.4982  comm_a=0.0000  comms_b=0.0456  meas_in_a=0.0000  reg=0.0000
# epoch 140  total=0.0451  shoot_loss=0.9521  shoot_acc=0.4982  comm_a=0.0000  comms_b=0.0451  meas_in_a=0.0000  reg=0.0000
# epoch 141  total=0.0452  shoot_loss=0.9526  shoot_acc=0.4981  comm_a=0.0000  comms_b=0.0452  meas_in_a=0.0000  reg=0.0000
# epoch 142  total=0.0447  shoot_loss=0.9524  shoot_acc=0.4982  comm_a=0.0000  comms_b=0.0447  meas_in_a=0.0000  reg=0.0000
# epoch 143  total=0.0448  shoot_loss=0.9534  shoot_acc=0.4981  comm_a=0.0000  comms_b=0.0448  meas_in_a=0.0000  reg=0.0000
# epoch 144  total=0.0447  shoot_loss=0.9535  shoot_acc=0.4979  comm_a=0.0000  comms_b=0.0447  meas_in_a=0.0000  reg=0.0000
# epoch 145  total=0.0450  shoot_loss=0.9525  shoot_acc=0.4981  comm_a=0.0000  comms_b=0.0450  meas_in_a=0.0000  reg=0.0000
# epoch 146  total=0.0446  shoot_loss=0.9521  shoot_acc=0.4982  comm_a=0.0000  comms_b=0.0446  meas_in_a=0.0000  reg=0.0000
# epoch 147  total=0.0446  shoot_loss=0.9511  shoot_acc=0.4979  comm_a=0.0000  comms_b=0.0446  meas_in_a=0.0000  reg=0.0000
# epoch 148  total=0.0445  shoot_loss=0.9501  shoot_acc=0.4984  comm_a=0.0000  comms_b=0.0445  meas_in_a=0.0000  reg=0.0000
# epoch 149  total=0.0443  shoot_loss=0.9500  shoot_acc=0.4987  comm_a=0.0000  comms_b=0.0443  meas_in_a=0.0000  reg=0.0000
# epoch 150  total=0.1929  shoot_loss=0.9597  shoot_acc=0.4996  comm_a=0.0000  comms_b=0.1929  meas_in_a=0.0000  reg=0.0000
# epoch 151  total=0.0760  shoot_loss=0.9953  shoot_acc=0.5021  comm_a=0.0000  comms_b=0.0760  meas_in_a=0.0000  reg=0.0000
# epoch 152  total=0.0484  shoot_loss=1.0140  shoot_acc=0.5039  comm_a=0.0000  comms_b=0.0484  meas_in_a=0.0000  reg=0.0000
# epoch 153  total=0.0458  shoot_loss=1.0235  shoot_acc=0.5058  comm_a=0.0000  comms_b=0.0458  meas_in_a=0.0000  reg=0.0000
# epoch 154  total=0.0453  shoot_loss=1.0305  shoot_acc=0.5063  comm_a=0.0000  comms_b=0.0453  meas_in_a=0.0000  reg=0.0000
# epoch 155  total=0.0452  shoot_loss=1.0340  shoot_acc=0.5067  comm_a=0.0000  comms_b=0.0452  meas_in_a=0.0000  reg=0.0000
# epoch 156  total=0.0450  shoot_loss=1.0376  shoot_acc=0.5065  comm_a=0.0000  comms_b=0.0450  meas_in_a=0.0000  reg=0.0000
# epoch 157  total=0.0445  shoot_loss=1.0397  shoot_acc=0.5066  comm_a=0.0000  comms_b=0.0445  meas_in_a=0.0000  reg=0.0000
# epoch 158  total=0.0445  shoot_loss=1.0430  shoot_acc=0.5073  comm_a=0.0000  comms_b=0.0445  meas_in_a=0.0000  reg=0.0000
# epoch 159  total=0.0452  shoot_loss=1.0417  shoot_acc=0.5075  comm_a=0.0000  comms_b=0.0452  meas_in_a=0.0000  reg=0.0000
# epoch 160  total=0.0444  shoot_loss=1.0436  shoot_acc=0.5083  comm_a=0.0000  comms_b=0.0444  meas_in_a=0.0000  reg=0.0000
# epoch 161  total=0.0441  shoot_loss=1.0456  shoot_acc=0.5082  comm_a=0.0000  comms_b=0.0441  meas_in_a=0.0000  reg=0.0000
# epoch 162  total=0.0449  shoot_loss=1.0437  shoot_acc=0.5085  comm_a=0.0000  comms_b=0.0449  meas_in_a=0.0000  reg=0.0000
# epoch 163  total=0.0441  shoot_loss=1.0489  shoot_acc=0.5078  comm_a=0.0000  comms_b=0.0441  meas_in_a=0.0000  reg=0.0000
# epoch 164  total=0.0439  shoot_loss=1.0517  shoot_acc=0.5080  comm_a=0.0000  comms_b=0.0439  meas_in_a=0.0000  reg=0.0000
# epoch 165  total=0.0439  shoot_loss=1.0491  shoot_acc=0.5081  comm_a=0.0000  comms_b=0.0439  meas_in_a=0.0000  reg=0.0000
# epoch 166  total=0.0438  shoot_loss=1.0550  shoot_acc=0.5082  comm_a=0.0000  comms_b=0.0438  meas_in_a=0.0000  reg=0.0000
# epoch 167  total=0.0440  shoot_loss=1.0573  shoot_acc=0.5078  comm_a=0.0000  comms_b=0.0440  meas_in_a=0.0000  reg=0.0000
# epoch 168  total=0.0439  shoot_loss=1.0564  shoot_acc=0.5077  comm_a=0.0000  comms_b=0.0439  meas_in_a=0.0000  reg=0.0000
# epoch 169  total=0.0437  shoot_loss=1.0601  shoot_acc=0.5075  comm_a=0.0000  comms_b=0.0437  meas_in_a=0.0000  reg=0.0000
# epoch 170  total=0.0436  shoot_loss=1.0621  shoot_acc=0.5080  comm_a=0.0000  comms_b=0.0436  meas_in_a=0.0000  reg=0.0000
# epoch 171  total=0.0467  shoot_loss=1.0520  shoot_acc=0.5079  comm_a=0.0000  comms_b=0.0467  meas_in_a=0.0000  reg=0.0000
# epoch 172  total=0.0437  shoot_loss=1.0530  shoot_acc=0.5083  comm_a=0.0000  comms_b=0.0437  meas_in_a=0.0000  reg=0.0000
# epoch 173  total=0.0436  shoot_loss=1.0565  shoot_acc=0.5076  comm_a=0.0000  comms_b=0.0436  meas_in_a=0.0000  reg=0.0000
# epoch 174  total=0.0435  shoot_loss=1.0593  shoot_acc=0.5079  comm_a=0.0000  comms_b=0.0435  meas_in_a=0.0000  reg=0.0000
# epoch 175  total=0.0435  shoot_loss=1.0598  shoot_acc=0.5077  comm_a=0.0000  comms_b=0.0435  meas_in_a=0.0000  reg=0.0000
# epoch 176  total=0.0435  shoot_loss=1.0630  shoot_acc=0.5071  comm_a=0.0000  comms_b=0.0435  meas_in_a=0.0000  reg=0.0000
# epoch 177  total=0.0434  shoot_loss=1.0652  shoot_acc=0.5064  comm_a=0.0000  comms_b=0.0434  meas_in_a=0.0000  reg=0.0000
# epoch 178  total=0.0433  shoot_loss=1.0659  shoot_acc=0.5063  comm_a=0.0000  comms_b=0.0433  meas_in_a=0.0000  reg=0.0000
# epoch 179  total=0.0434  shoot_loss=1.0689  shoot_acc=0.5056  comm_a=0.0000  comms_b=0.0434  meas_in_a=0.0000  reg=0.0000
# epoch 180  total=0.0438  shoot_loss=1.0606  shoot_acc=0.5060  comm_a=0.0000  comms_b=0.0438  meas_in_a=0.0000  reg=0.0000
# epoch 181  total=0.0433  shoot_loss=1.0616  shoot_acc=0.5061  comm_a=0.0000  comms_b=0.0433  meas_in_a=0.0000  reg=0.0000
# epoch 182  total=0.0431  shoot_loss=1.0637  shoot_acc=0.5062  comm_a=0.0000  comms_b=0.0431  meas_in_a=0.0000  reg=0.0000
# epoch 183  total=0.0432  shoot_loss=1.0648  shoot_acc=0.5063  comm_a=0.0000  comms_b=0.0432  meas_in_a=0.0000  reg=0.0000
# epoch 184  total=0.0431  shoot_loss=1.0679  shoot_acc=0.5061  comm_a=0.0000  comms_b=0.0431  meas_in_a=0.0000  reg=0.0000
# epoch 185  total=0.0431  shoot_loss=1.0689  shoot_acc=0.5058  comm_a=0.0000  comms_b=0.0431  meas_in_a=0.0000  reg=0.0000
# epoch 186  total=0.0431  shoot_loss=1.0720  shoot_acc=0.5059  comm_a=0.0000  comms_b=0.0431  meas_in_a=0.0000  reg=0.0000
# epoch 187  total=0.0430  shoot_loss=1.0711  shoot_acc=0.5056  comm_a=0.0000  comms_b=0.0430  meas_in_a=0.0000  reg=0.0000
# epoch 188  total=0.0429  shoot_loss=1.0724  shoot_acc=0.5050  comm_a=0.0000  comms_b=0.0429  meas_in_a=0.0000  reg=0.0000
# epoch 189  total=0.0439  shoot_loss=1.0589  shoot_acc=0.5052  comm_a=0.0000  comms_b=0.0439  meas_in_a=0.0000  reg=0.0000
# epoch 190  total=0.0429  shoot_loss=1.0635  shoot_acc=0.5051  comm_a=0.0000  comms_b=0.0429  meas_in_a=0.0000  reg=0.0000
# epoch 191  total=0.0429  shoot_loss=1.0663  shoot_acc=0.5053  comm_a=0.0000  comms_b=0.0429  meas_in_a=0.0000  reg=0.0000
# epoch 192  total=0.0428  shoot_loss=1.0708  shoot_acc=0.5049  comm_a=0.0000  comms_b=0.0428  meas_in_a=0.0000  reg=0.0000
# epoch 193  total=0.0429  shoot_loss=1.0702  shoot_acc=0.5054  comm_a=0.0000  comms_b=0.0429  meas_in_a=0.0000  reg=0.0000
# epoch 194  total=0.0428  shoot_loss=1.0741  shoot_acc=0.5057  comm_a=0.0000  comms_b=0.0428  meas_in_a=0.0000  reg=0.0000
# epoch 195  total=0.0428  shoot_loss=1.0715  shoot_acc=0.5050  comm_a=0.0000  comms_b=0.0428  meas_in_a=0.0000  reg=0.0000
# epoch 196  total=0.0428  shoot_loss=1.0742  shoot_acc=0.5051  comm_a=0.0000  comms_b=0.0428  meas_in_a=0.0000  reg=0.0000
# epoch 197  total=0.0428  shoot_loss=1.0764  shoot_acc=0.5052  comm_a=0.0000  comms_b=0.0428  meas_in_a=0.0000  reg=0.0000
# epoch 198  total=0.0428  shoot_loss=1.0749  shoot_acc=0.5050  comm_a=0.0000  comms_b=0.0428  meas_in_a=0.0000  reg=0.0000
# epoch 199  total=0.0427  shoot_loss=1.0800  shoot_acc=0.5055  comm_a=0.0000  comms_b=0.0427  meas_in_a=0.0000  reg=0.0000
# epoch 200  total=0.0674  shoot_loss=1.0966  shoot_acc=0.4437  comm_a=0.0000  comms_b=0.0674  meas_in_a=0.0000  reg=0.0000
# epoch 201  total=0.0437  shoot_loss=1.2091  shoot_acc=0.4122  comm_a=0.0000  comms_b=0.0437  meas_in_a=0.0000  reg=0.0000
# epoch 202  total=0.0434  shoot_loss=1.2391  shoot_acc=0.4079  comm_a=0.0000  comms_b=0.0434  meas_in_a=0.0000  reg=0.0000
# epoch 203  total=0.0440  shoot_loss=1.2520  shoot_acc=0.4025  comm_a=0.0000  comms_b=0.0440  meas_in_a=0.0000  reg=0.0000
# epoch 204  total=0.0433  shoot_loss=1.2725  shoot_acc=0.3962  comm_a=0.0000  comms_b=0.0433  meas_in_a=0.0000  reg=0.0000
# epoch 205  total=0.0431  shoot_loss=1.2800  shoot_acc=0.3976  comm_a=0.0000  comms_b=0.0431  meas_in_a=0.0000  reg=0.0000
# epoch 206  total=0.0431  shoot_loss=1.2879  shoot_acc=0.3964  comm_a=0.0000  comms_b=0.0431  meas_in_a=0.0000  reg=0.0000
# epoch 207  total=0.0431  shoot_loss=1.2918  shoot_acc=0.3956  comm_a=0.0000  comms_b=0.0431  meas_in_a=0.0000  reg=0.0000
# epoch 208  total=0.0431  shoot_loss=1.2950  shoot_acc=0.3968  comm_a=0.0000  comms_b=0.0431  meas_in_a=0.0000  reg=0.0000
# epoch 209  total=0.0430  shoot_loss=1.3005  shoot_acc=0.3942  comm_a=0.0000  comms_b=0.0430  meas_in_a=0.0000  reg=0.0000
# epoch 210  total=0.0430  shoot_loss=1.2995  shoot_acc=0.3947  comm_a=0.0000  comms_b=0.0430  meas_in_a=0.0000  reg=0.0000
# epoch 211  total=0.0435  shoot_loss=1.3110  shoot_acc=0.3828  comm_a=0.0000  comms_b=0.0435  meas_in_a=0.0000  reg=0.0000
# epoch 212  total=0.0432  shoot_loss=1.3206  shoot_acc=0.3839  comm_a=0.0000  comms_b=0.0432  meas_in_a=0.0000  reg=0.0000
# epoch 213  total=0.0430  shoot_loss=1.3251  shoot_acc=0.3818  comm_a=0.0000  comms_b=0.0430  meas_in_a=0.0000  reg=0.0000
# epoch 214  total=0.0432  shoot_loss=1.3239  shoot_acc=0.3811  comm_a=0.0000  comms_b=0.0432  meas_in_a=0.0000  reg=0.0000
# epoch 215  total=0.0435  shoot_loss=1.3098  shoot_acc=0.3917  comm_a=0.0000  comms_b=0.0435  meas_in_a=0.0000  reg=0.0000
# epoch 216  total=0.0431  shoot_loss=1.3155  shoot_acc=0.4011  comm_a=0.0000  comms_b=0.0431  meas_in_a=0.0000  reg=0.0000
# epoch 217  total=0.0431  shoot_loss=1.3145  shoot_acc=0.3984  comm_a=0.0000  comms_b=0.0431  meas_in_a=0.0000  reg=0.0000
# epoch 218  total=0.0431  shoot_loss=1.3162  shoot_acc=0.3966  comm_a=0.0000  comms_b=0.0431  meas_in_a=0.0000  reg=0.0000
# epoch 219  total=0.0431  shoot_loss=1.3201  shoot_acc=0.3988  comm_a=0.0000  comms_b=0.0431  meas_in_a=0.0000  reg=0.0000
# epoch 220  total=0.0434  shoot_loss=1.3401  shoot_acc=0.3893  comm_a=0.0000  comms_b=0.0434  meas_in_a=0.0000  reg=0.0000
# epoch 221  total=0.0430  shoot_loss=1.3513  shoot_acc=0.3865  comm_a=0.0000  comms_b=0.0430  meas_in_a=0.0000  reg=0.0000
# epoch 222  total=0.0430  shoot_loss=1.3594  shoot_acc=0.3857  comm_a=0.0000  comms_b=0.0430  meas_in_a=0.0000  reg=0.0000
# epoch 223  total=0.0430  shoot_loss=1.3669  shoot_acc=0.3814  comm_a=0.0000  comms_b=0.0430  meas_in_a=0.0000  reg=0.0000
# epoch 224  total=0.0430  shoot_loss=1.3659  shoot_acc=0.3780  comm_a=0.0000  comms_b=0.0430  meas_in_a=0.0000  reg=0.0000
# epoch 225  total=0.0429  shoot_loss=1.3675  shoot_acc=0.3804  comm_a=0.0000  comms_b=0.0429  meas_in_a=0.0000  reg=0.0000
# epoch 226  total=0.0430  shoot_loss=1.3662  shoot_acc=0.3798  comm_a=0.0000  comms_b=0.0430  meas_in_a=0.0000  reg=0.0000
# epoch 227  total=0.0429  shoot_loss=1.3671  shoot_acc=0.3770  comm_a=0.0000  comms_b=0.0429  meas_in_a=0.0000  reg=0.0000
# epoch 228  total=0.0438  shoot_loss=1.3505  shoot_acc=0.3820  comm_a=0.0000  comms_b=0.0438  meas_in_a=0.0000  reg=0.0000
# epoch 229  total=0.0432  shoot_loss=1.3439  shoot_acc=0.3941  comm_a=0.0000  comms_b=0.0432  meas_in_a=0.0000  reg=0.0000
# epoch 230  total=0.0429  shoot_loss=1.3512  shoot_acc=0.3923  comm_a=0.0000  comms_b=0.0429  meas_in_a=0.0000  reg=0.0000
# epoch 231  total=0.0429  shoot_loss=1.3595  shoot_acc=0.3853  comm_a=0.0000  comms_b=0.0429  meas_in_a=0.0000  reg=0.0000
# epoch 232  total=0.0428  shoot_loss=1.3544  shoot_acc=0.3904  comm_a=0.0000  comms_b=0.0428  meas_in_a=0.0000  reg=0.0000
# epoch 233  total=0.0434  shoot_loss=1.3446  shoot_acc=0.3887  comm_a=0.0000  comms_b=0.0434  meas_in_a=0.0000  reg=0.0000
# epoch 234  total=0.0450  shoot_loss=1.3568  shoot_acc=0.3740  comm_a=0.0000  comms_b=0.0450  meas_in_a=0.0000  reg=0.0000
# epoch 235  total=0.0430  shoot_loss=1.3848  shoot_acc=0.3716  comm_a=0.0000  comms_b=0.0430  meas_in_a=0.0000  reg=0.0000
# epoch 236  total=0.0429  shoot_loss=1.3832  shoot_acc=0.3715  comm_a=0.0000  comms_b=0.0429  meas_in_a=0.0000  reg=0.0000
# epoch 237  total=0.0429  shoot_loss=1.3836  shoot_acc=0.3756  comm_a=0.0000  comms_b=0.0429  meas_in_a=0.0000  reg=0.0000
# epoch 238  total=0.0429  shoot_loss=1.3846  shoot_acc=0.3768  comm_a=0.0000  comms_b=0.0429  meas_in_a=0.0000  reg=0.0000
# epoch 239  total=0.0429  shoot_loss=1.3704  shoot_acc=0.3765  comm_a=0.0000  comms_b=0.0429  meas_in_a=0.0000  reg=0.0000
# epoch 240  total=0.0429  shoot_loss=1.3670  shoot_acc=0.3832  comm_a=0.0000  comms_b=0.0429  meas_in_a=0.0000  reg=0.0000
# epoch 241  total=0.0429  shoot_loss=1.3666  shoot_acc=0.3875  comm_a=0.0000  comms_b=0.0429  meas_in_a=0.0000  reg=0.0000
# epoch 242  total=0.0441  shoot_loss=1.3426  shoot_acc=0.3919  comm_a=0.0000  comms_b=0.0441  meas_in_a=0.0000  reg=0.0000
# epoch 243  total=0.0429  shoot_loss=1.3611  shoot_acc=0.3868  comm_a=0.0000  comms_b=0.0429  meas_in_a=0.0000  reg=0.0000
# epoch 244  total=0.0429  shoot_loss=1.3674  shoot_acc=0.3905  comm_a=0.0000  comms_b=0.0429  meas_in_a=0.0000  reg=0.0000
# epoch 245  total=0.0429  shoot_loss=1.3711  shoot_acc=0.3909  comm_a=0.0000  comms_b=0.0429  meas_in_a=0.0000  reg=0.0000
# epoch 246  total=0.0428  shoot_loss=1.3766  shoot_acc=0.3978  comm_a=0.0000  comms_b=0.0428  meas_in_a=0.0000  reg=0.0000
# epoch 247  total=0.0428  shoot_loss=1.3764  shoot_acc=0.3982  comm_a=0.0000  comms_b=0.0428  meas_in_a=0.0000  reg=0.0000
# epoch 248  total=0.0429  shoot_loss=1.3602  shoot_acc=0.3982  comm_a=0.0000  comms_b=0.0429  meas_in_a=0.0000  reg=0.0000
# epoch 249  total=0.0429  shoot_loss=1.3731  shoot_acc=0.4114  comm_a=0.0000  comms_b=0.0429  meas_in_a=0.0000  reg=0.0000
# epoch 250  total=0.1297  shoot_loss=0.1297  shoot_acc=0.9216  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 251  total=0.0882  shoot_loss=0.0882  shoot_acc=0.9384  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 252  total=0.0879  shoot_loss=0.0879  shoot_acc=0.9384  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 253  total=0.0873  shoot_loss=0.0873  shoot_acc=0.9392  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 254  total=0.0873  shoot_loss=0.0873  shoot_acc=0.9395  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 255  total=0.0891  shoot_loss=0.0891  shoot_acc=0.9376  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 256  total=0.0872  shoot_loss=0.0872  shoot_acc=0.9390  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 257  total=0.0871  shoot_loss=0.0871  shoot_acc=0.9391  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 258  total=0.0879  shoot_loss=0.0879  shoot_acc=0.9395  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 259  total=0.0796  shoot_loss=0.0796  shoot_acc=0.9458  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 260  total=0.0663  shoot_loss=0.0663  shoot_acc=0.9540  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 261  total=0.0656  shoot_loss=0.0656  shoot_acc=0.9539  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 262  total=0.0654  shoot_loss=0.0654  shoot_acc=0.9546  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 263  total=0.0540  shoot_loss=0.0540  shoot_acc=0.9622  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 264  total=0.0435  shoot_loss=0.0435  shoot_acc=0.9696  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 265  total=0.0468  shoot_loss=0.0468  shoot_acc=0.9693  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 266  total=0.0433  shoot_loss=0.0433  shoot_acc=0.9691  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 267  total=0.0435  shoot_loss=0.0435  shoot_acc=0.9689  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 268  total=0.0432  shoot_loss=0.0432  shoot_acc=0.9694  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 269  total=0.0433  shoot_loss=0.0433  shoot_acc=0.9693  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 270  total=0.0431  shoot_loss=0.0431  shoot_acc=0.9691  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 271  total=0.0432  shoot_loss=0.0432  shoot_acc=0.9689  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 272  total=0.0431  shoot_loss=0.0431  shoot_acc=0.9697  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 273  total=0.0432  shoot_loss=0.0432  shoot_acc=0.9689  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 274  total=0.0431  shoot_loss=0.0431  shoot_acc=0.9693  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 275  total=0.0433  shoot_loss=0.0433  shoot_acc=0.9693  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 276  total=0.0432  shoot_loss=0.0432  shoot_acc=0.9690  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 277  total=0.0431  shoot_loss=0.0431  shoot_acc=0.9696  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 278  total=0.0431  shoot_loss=0.0431  shoot_acc=0.9687  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 279  total=0.0434  shoot_loss=0.0434  shoot_acc=0.9690  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 280  total=0.0508  shoot_loss=0.0508  shoot_acc=0.9674  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 281  total=0.0430  shoot_loss=0.0430  shoot_acc=0.9698  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 282  total=0.0431  shoot_loss=0.0431  shoot_acc=0.9691  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 283  total=0.0436  shoot_loss=0.0436  shoot_acc=0.9694  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 284  total=0.0431  shoot_loss=0.0431  shoot_acc=0.9693  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 285  total=0.0433  shoot_loss=0.0433  shoot_acc=0.9691  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 286  total=0.0432  shoot_loss=0.0432  shoot_acc=0.9693  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 287  total=0.0431  shoot_loss=0.0431  shoot_acc=0.9692  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 288  total=0.0431  shoot_loss=0.0431  shoot_acc=0.9687  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 289  total=0.0431  shoot_loss=0.0431  shoot_acc=0.9694  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 290  total=0.0431  shoot_loss=0.0431  shoot_acc=0.9688  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 291  total=0.0430  shoot_loss=0.0430  shoot_acc=0.9690  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 292  total=0.0430  shoot_loss=0.0430  shoot_acc=0.9694  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 293  total=0.0437  shoot_loss=0.0437  shoot_acc=0.9692  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 294  total=0.0433  shoot_loss=0.0433  shoot_acc=0.9690  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 295  total=0.0434  shoot_loss=0.0434  shoot_acc=0.9693  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 296  total=0.0431  shoot_loss=0.0431  shoot_acc=0.9695  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 297  total=0.0430  shoot_loss=0.0430  shoot_acc=0.9692  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 298  total=0.0430  shoot_loss=0.0430  shoot_acc=0.9694  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 299  total=0.0430  shoot_loss=0.0430  shoot_acc=0.9698  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 300  total=0.3389  shoot_loss=0.3389  shoot_acc=0.8576  comm_a=0.1091  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 301  total=0.1330  shoot_loss=0.1330  shoot_acc=0.9475  comm_a=0.0001  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 302  total=0.0932  shoot_loss=0.0932  shoot_acc=0.9593  comm_a=0.0002  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 303  total=0.0780  shoot_loss=0.0780  shoot_acc=0.9627  comm_a=0.0001  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 304  total=0.0795  shoot_loss=0.0795  shoot_acc=0.9623  comm_a=0.0003  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 305  total=0.0694  shoot_loss=0.0694  shoot_acc=0.9639  comm_a=0.0079  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 306  total=0.0854  shoot_loss=0.0854  shoot_acc=0.9609  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 307  total=0.0692  shoot_loss=0.0692  shoot_acc=0.9635  comm_a=0.0001  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 308  total=0.1327  shoot_loss=0.1327  shoot_acc=0.9429  comm_a=1.2951  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 309  total=0.1067  shoot_loss=0.1067  shoot_acc=0.9528  comm_a=0.7612  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 310  total=0.0529  shoot_loss=0.0529  shoot_acc=0.9671  comm_a=0.0101  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 311  total=0.0720  shoot_loss=0.0720  shoot_acc=0.9641  comm_a=0.0086  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 312  total=0.1279  shoot_loss=0.1279  shoot_acc=0.9483  comm_a=0.0006  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 313  total=0.0970  shoot_loss=0.0970  shoot_acc=0.9576  comm_a=0.0003  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 314  total=0.0773  shoot_loss=0.0773  shoot_acc=0.9632  comm_a=0.0100  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 315  total=0.0800  shoot_loss=0.0800  shoot_acc=0.9615  comm_a=0.0260  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 316  total=0.0911  shoot_loss=0.0911  shoot_acc=0.9594  comm_a=0.0033  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 317  total=0.0907  shoot_loss=0.0907  shoot_acc=0.9596  comm_a=0.0017  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 318  total=0.0704  shoot_loss=0.0704  shoot_acc=0.9645  comm_a=0.0004  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 319  total=0.0688  shoot_loss=0.0688  shoot_acc=0.9640  comm_a=0.0002  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 320  total=0.0629  shoot_loss=0.0629  shoot_acc=0.9656  comm_a=0.0139  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 321  total=0.0511  shoot_loss=0.0511  shoot_acc=0.9679  comm_a=0.0001  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 322  total=0.0562  shoot_loss=0.0562  shoot_acc=0.9676  comm_a=0.0001  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 323  total=0.0644  shoot_loss=0.0644  shoot_acc=0.9654  comm_a=0.0001  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 324  total=0.0630  shoot_loss=0.0630  shoot_acc=0.9658  comm_a=0.0001  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 325  total=0.0554  shoot_loss=0.0554  shoot_acc=0.9669  comm_a=0.0000  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 326  total=0.1116  shoot_loss=0.1116  shoot_acc=0.9523  comm_a=0.8846  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 327  total=0.1580  shoot_loss=0.1580  shoot_acc=0.9309  comm_a=3.6203  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 328  total=0.1586  shoot_loss=0.1586  shoot_acc=0.9288  comm_a=4.3151  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 329  total=0.1622  shoot_loss=0.1622  shoot_acc=0.9249  comm_a=4.8738  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 330  total=0.1750  shoot_loss=0.1750  shoot_acc=0.9201  comm_a=5.0928  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 331  total=0.1588  shoot_loss=0.1588  shoot_acc=0.9257  comm_a=5.0059  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 332  total=0.1549  shoot_loss=0.1549  shoot_acc=0.9263  comm_a=5.4302  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 333  total=0.1642  shoot_loss=0.1642  shoot_acc=0.9262  comm_a=4.9898  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 334  total=0.1741  shoot_loss=0.1741  shoot_acc=0.9243  comm_a=4.8399  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 335  total=0.1497  shoot_loss=0.1497  shoot_acc=0.9297  comm_a=4.9812  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 336  total=0.1405  shoot_loss=0.1405  shoot_acc=0.9301  comm_a=5.5127  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 337  total=0.1608  shoot_loss=0.1608  shoot_acc=0.9202  comm_a=6.3705  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 338  total=0.1670  shoot_loss=0.1670  shoot_acc=0.9177  comm_a=7.1139  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 339  total=0.1597  shoot_loss=0.1597  shoot_acc=0.9177  comm_a=7.4382  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 340  total=0.2156  shoot_loss=0.2156  shoot_acc=0.8931  comm_a=7.6138  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 341  total=0.1944  shoot_loss=0.1944  shoot_acc=0.9092  comm_a=6.1871  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 342  total=0.1530  shoot_loss=0.1530  shoot_acc=0.9261  comm_a=5.4256  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 343  total=0.1668  shoot_loss=0.1668  shoot_acc=0.9214  comm_a=5.7640  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 344  total=0.1899  shoot_loss=0.1899  shoot_acc=0.9173  comm_a=5.6846  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 345  total=0.1855  shoot_loss=0.1855  shoot_acc=0.9209  comm_a=5.3355  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 346  total=0.1752  shoot_loss=0.1752  shoot_acc=0.9213  comm_a=5.8823  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 347  total=0.1416  shoot_loss=0.1416  shoot_acc=0.9328  comm_a=5.1556  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 348  total=0.1566  shoot_loss=0.1566  shoot_acc=0.9280  comm_a=5.5888  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 349  total=0.1596  shoot_loss=0.1596  shoot_acc=0.9241  comm_a=6.2306  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 350  total=0.1404  shoot_loss=0.1404  shoot_acc=0.9291  comm_a=6.4115  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 351  total=0.1484  shoot_loss=0.1484  shoot_acc=0.9270  comm_a=6.5337  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 352  total=0.1375  shoot_loss=0.1375  shoot_acc=0.9282  comm_a=6.8817  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 353  total=0.1573  shoot_loss=0.1573  shoot_acc=0.9236  comm_a=7.3033  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 354  total=0.1480  shoot_loss=0.1480  shoot_acc=0.9257  comm_a=7.4431  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 355  total=0.1679  shoot_loss=0.1679  shoot_acc=0.9205  comm_a=7.6180  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 356  total=0.1488  shoot_loss=0.1488  shoot_acc=0.9253  comm_a=7.4292  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 357  total=0.1519  shoot_loss=0.1519  shoot_acc=0.9258  comm_a=6.9711  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 358  total=0.1490  shoot_loss=0.1490  shoot_acc=0.9265  comm_a=6.8630  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 359  total=0.1563  shoot_loss=0.1563  shoot_acc=0.9268  comm_a=6.5342  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 360  total=0.1632  shoot_loss=0.1632  shoot_acc=0.9242  comm_a=6.5396  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 361  total=0.2013  shoot_loss=0.2013  shoot_acc=0.9132  comm_a=6.7207  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 362  total=0.1621  shoot_loss=0.1621  shoot_acc=0.9238  comm_a=6.7223  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 363  total=0.1652  shoot_loss=0.1652  shoot_acc=0.9239  comm_a=6.6559  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 364  total=0.1890  shoot_loss=0.1890  shoot_acc=0.9163  comm_a=6.6669  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 365  total=0.2633  shoot_loss=0.2633  shoot_acc=0.8827  comm_a=6.4578  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 366  total=0.2145  shoot_loss=0.2145  shoot_acc=0.9074  comm_a=6.2797  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 367  total=0.2142  shoot_loss=0.2142  shoot_acc=0.9103  comm_a=6.5920  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 368  total=0.2203  shoot_loss=0.2203  shoot_acc=0.9050  comm_a=6.4729  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 369  total=0.1737  shoot_loss=0.1737  shoot_acc=0.9230  comm_a=6.4182  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 370  total=0.1666  shoot_loss=0.1666  shoot_acc=0.9250  comm_a=6.0242  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 371  total=0.1421  shoot_loss=0.1421  shoot_acc=0.9293  comm_a=6.5067  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 372  total=0.1412  shoot_loss=0.1412  shoot_acc=0.9256  comm_a=7.8009  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 373  total=0.1439  shoot_loss=0.1439  shoot_acc=0.9223  comm_a=8.3699  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 374  total=0.1323  shoot_loss=0.1323  shoot_acc=0.9270  comm_a=7.5187  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 375  total=0.1592  shoot_loss=0.1592  shoot_acc=0.9185  comm_a=7.0533  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 376  total=0.1473  shoot_loss=0.1473  shoot_acc=0.9232  comm_a=6.5618  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 377  total=0.1323  shoot_loss=0.1323  shoot_acc=0.9279  comm_a=6.7281  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 378  total=0.1473  shoot_loss=0.1473  shoot_acc=0.9227  comm_a=7.5862  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 379  total=0.1659  shoot_loss=0.1659  shoot_acc=0.9078  comm_a=9.3839  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 380  total=0.1511  shoot_loss=0.1511  shoot_acc=0.9129  comm_a=10.0715  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000
# epoch 381  total=0.1636  shoot_loss=0.1636  shoot_acc=0.9075  comm_a=10.9554  comms_b=0.0000  meas_in_a=0.0000  reg=0.0000

# %%
batch   = tfds_train.take(1)
import random
for (field_logits_b, gun_logits_b, shoot_tgt_logits_b, meas_in_a_tgt_b, meas_out_a_tgt_b, comms_tgt_b) in tfds_train:
    i = random.randint(0, field_logits_b.shape[0]-1)
    field_logits = field_logits_b[i:i+1]
    gun_logits   = gun_logits_b[i:i+1]
    meas_out_a_tgt = [t[i:i+1] for t in meas_out_a_tgt_b]

    comm_logits, meas_list, out_list = model_a.compute_with_internal(
        field_logits = field_logits, 
        replay_out_a_logits_list = meas_out_a_tgt,
        training=True,
        )
    #return comm_logits, meas_list, out_list
    shoot_logit, meas_b_logits_list, out_b_logits_list, comms_logits_list, gun_logits_list = model_b.compute_with_internal(
            gun_logits,
            comm_logits,
            list(meas_list),
            list(out_list),
            training=True,
        )
    #return shoot_logit, meas_b_logits_list, out_b_logits_list, comms_logits_list, gun_logits_list
    break
print("field_logits\t\t\t", field_logits.numpy())
print("gun_logits\t\t\t", gun_logits.numpy())
print("shoot_logit\t\t\t", shoot_logit.numpy())
print("comm_logits\t\t\t", comm_logits.numpy())
for d in range(DEPTH):
    print(f"meas_list[{d}]\t\t\t", meas_list[d].numpy())
    print(f"out_list[{d}]\t\t\t", out_list[d].numpy())
    print(f"meas_b_logits_list[{d}]\t\t", meas_b_logits_list[d].numpy())
    print(f"out_b_logits_list[{d}]\t\t", out_b_logits_list[d].numpy())
    print(f"comms_logits_list[{d}]\t\t", comms_logits_list[d].numpy())
    print(f"gun_logits_list[{d}]\t\t", gun_logits_list[d].numpy())


# %%
# --------------------------
# Generate canonical dataset (bits)
# --------------------------
ds_bits = generate_pyr_dataset(n2=N2, num_games=100, seed=SEED, validate=True)

tr = convert_all_traces(
    ds_bits,
    rep_field="hard_logit",        
    rep_gun="hard_logit",
    rep_comms="hard_logit",
    rep_meas_in_a="hard_logit",
    rep_meas_out_a="hard_logit",
    rep_meas_in_b="hard_logit",
    rep_meas_out_b="hard_logit",
    rep_shoot="hard_logit",
    beta=BETA_INPUT,
)

# Model-B inputs (level-0 gun + comm0, plus A's previous lists)
gun_logits_np   = tr["gun"][0]      # (N, n2)
field_logits_np = tr["field"][0]    # (N, n2)
comm0_logits_np = tr["comms"][0]    # (N, 1)
shoot_logits_np = tr["shoot"]       # (N, 1)

prev_meas_list_np = [tr["meas_in_a"][d]  for d in range(DEPTH)]   # list of (N, k_d)
prev_out_list_np  = [tr["meas_out_a"][d] for d in range(DEPTH)]   # list of (N, k_d)

# Model-B targets (what you compare meas_b/out_b against)
meas_b_list_np = [tr["meas_in_b"][d]  for d in range(DEPTH)]      # list of (N, k_d)
out_b_list_np  = [tr["meas_out_b"][d] for d in range(DEPTH)]      # list of (N, k_d)

# Gun and comms targets are not used for training, but we convert them anyway for potential diagnostics
field_logits_list_np = [tr["field"][d] for d in range(DEPTH+1)]    # list of (N, n2)
gun_logits_list_np = [tr["gun"][d] for d in range(DEPTH+1)]      # list of (N, n2)
comm_logits_list_np = [tr["comms"][d] for d in range(DEPTH+1)]

# Shoot target (what BETA_TEACHEyou compare shoot_logit against)
shoot_target_np = tr["shoot"]          # (N, 1) bits


# numpy -> tf
gun_logits   = tf.constant(gun_logits_np, tf.float32)         # (N,n2) logits
field_logits = tf.constant(field_logits_np, tf.float32)       # (N,n2) logits
comm0_logits = tf.constant(comm0_logits_np, tf.float32)       # (N,1)  logits

prev_meas_t  = tuple(tf.constant(a, tf.float32) for a in prev_meas_list_np)  # tuple of (N,k_d) logits
prev_out_t   = tuple(tf.constant(a, tf.float32) for a in prev_out_list_np)   # tuple of (N,k_d) logits

meas_tgt_t   = tuple(tf.constant(a, tf.float32) for a in meas_b_list_np)     # tuple of (N,k_d) logits (TARGET)
out_tgt_t    = tuple(tf.constant(a, tf.float32) for a in out_b_list_np)      # tuple of (N,k_d) logits (TARGET)

gun_tgt_list_t = tuple(tf.constant(a, tf.float32) for a in gun_logits_list_np)   # tuple of (N,n2) logits (TARGET)
field_tgt_list_t = tuple(tf.constant(a, tf.float32) for a in field_logits_list_np)   # tuple of (N,n2) logits (TARGET)  
comm_tgt_list_t = tuple(tf.constant(a, tf.float32) for a in comm_logits_list_np)   # tuple of (N,1) logits (TARGET)

shoot_bits   = tf.constant(shoot_target_np, tf.float32)       # (N,1) bits

X_train = (gun_logits, field_logits, prev_out_t, )
Y_train = (comm_tgt_list_t, field_tgt_list_t, gun_tgt_list_t, prev_meas_t, meas_tgt_t, out_tgt_t)

tfds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
tfds = tfds.shuffle(200_000, seed=SEED, reshuffle_each_iteration=True)
tfds = tfds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

for X,Y in tfds:
    gun_logits, field_logits, prev_out_t  = X
    comm_tgt_list_t, field_tgt_list_t, gun_tgt_list_t, prev_meas_t, meas_tgt_t, out_tgt_t = Y
    i = random.randint(0, 90)

    # inputs
    field_logits = field_logits_b[i:i+1]
    gun_logits   = gun_logits_b[i:i+1]
    prev_out_logits_list = [t[i:i+1] for t in prev_out_t]

    # outputs
    comm_logits_tgt = [t[i:i+1] for t in comm_tgt_list_t]
    field_logits_tgt = [t[i:i+1] for t in field_tgt_list_t]
    gun_logits_tgt = [t[i:i+1] for t in gun_tgt_list_t]
    prev_meas_tgt = [t[i:i+1] for t in prev_meas_t]
    meas_tgt = [t[i:i+1] for t in meas_tgt_t]
    out_tgt = [t[i:i+1] for t in out_tgt_t]

    comm_logits, meas_list, out_list = model_a.compute_with_internal(
        field_logits = field_logits, 
        replay_out_a_logits_list = prev_out_logits_list,
        training=True,
        )
    #return comm_logits, meas_list, out_list
    shoot_logit, meas_b_logits_list, out_b_logits_list, comms_logits_list, gun_logits_list = model_b.compute_with_internal(
            gun_logits,
            comm_logits,
            list(meas_list),
            list(out_list),
            training=True,
        )
    #return shoot_logit, meas_b_logits_list, out_b_logits_list, comms_logits_list, gun_logits_list
    break

print("Inputs:")
print("field_logits\t", field_logits.numpy())
print("gun_logits\t", gun_logits.numpy())
print("prev_out_logits_list:")
for d in range(DEPTH):
    print(f"\tprev_out[{d}]\t\t\t", prev_out_logits_list[d].numpy())
print("\nTargets:")
for d in range(DEPTH):
    print(f"\nDepth: {d}")
    print(f"comms_tgt_list[{d}]\t\t", comm_logits_tgt[d].numpy())
    print(f"field_tgt_list[{d}]\t\t", field_logits_tgt[d].numpy())
    print(f"gun_tgt_list[{d}]\t\t\t", gun_logits_tgt[d].numpy())
    print(f"prev_meas[{d}]\t\t\t", prev_meas_tgt[d].numpy())
    print(f"meas_b_logits_list[{d}]\t\t", meas_tgt[d].numpy())
    print(f"out_b_logits_list[{d}]\t\t", out_tgt[d].numpy())
print(f"\nDepth: {d+1}")
print(f"comms_tgt_list[{d+1}]\t\t", comm_logits_tgt[d+1].numpy())
print(f"field_tgt_list[{d+1}]\t\t", field_logits_tgt[d+1].numpy())
print(f"gun_tgt_list[{d+1}]\t\t\t", gun_logits_tgt[d+1].numpy())
print("\nTargets:")
for d in range(DEPTH):
    print(f"\nDepth: {d}")
    print(f"comms_tgt_list[{d}]\t\t", comms_logits_list[d].numpy())
    #print(f"field_tgt_list[{d}]\t\t", field_logits_tgt[d].numpy())
    print(f"gun_tgt_list[{d}]\t\t\t", gun_logits_list[d].numpy())
    print(f"prev_meas[{d}]\t\t\t", meas_list[d].numpy())
    print(f"meas_b_logits_list[{d}]\t\t", meas_b_logits_list[d].numpy())
    print(f"out_b_logits_list[{d}]\t\t", out_b_logits_list[d].numpy())
print(f"\nDepth: {d+1}")
print(f"comms_tgt_list[{d+1}]\t\t", comms_logits_list[d+1].numpy())
print(f"field_tgt_list[{d+1}]\t\t", field_logits_tgt[d+1].numpy())
print(f"gun_tgt_list[{d+1}]\t\t\t", gun_logits_list[d+1].numpy())
    

# %%
import tensorflow as tf
import numpy as np
import random

# --------------------------
# Helpers
# --------------------------
def bit_from_logits(x: tf.Tensor) -> tf.Tensor:
    """Semantic bit from logits: 1[logit>=0]."""
    x = tf.convert_to_tensor(x)
    return tf.cast(x >= 0.0, tf.float32)

def sign_match_pct(pred: tf.Tensor, tgt: tf.Tensor) -> tf.Tensor:
    """Return scalar % of elements where sign-bit matches."""
    pb = bit_from_logits(pred)
    tb = bit_from_logits(tgt)
    eq = tf.cast(tf.equal(pb, tb), tf.float32)
    return tf.reduce_mean(eq) * 100.0

def mean_abs(x: tf.Tensor) -> tf.Tensor:
    x = tf.cast(x, tf.float32)
    return tf.reduce_mean(tf.abs(x))

def mae(pred: tf.Tensor, tgt: tf.Tensor) -> tf.Tensor:
    pred = tf.cast(pred, tf.float32)
    tgt  = tf.cast(tgt, tf.float32)
    return tf.reduce_mean(tf.abs(pred - tgt))

def report_line(name: str, pred: tf.Tensor, tgt: tf.Tensor):
    sm  = float(sign_match_pct(pred, tgt).numpy())
    m   = float(mae(pred, tgt).numpy())
    ap  = float(mean_abs(pred).numpy())
    at  = float(mean_abs(tgt).numpy())
    print(f"{name:22s}  sign_match={sm:7.2f}%   mae={m:9.4f}   |pred|={ap:9.4f}   |tgt|={at:9.4f}")

def crop_to(t: tf.Tensor, width: int) -> tf.Tensor:
    """Crop last dim to width (safe if already <= width)."""
    width = int(width)
    return t[..., :width]

def level_sizes(n2: int, d: int):
    """(L_d, k_d) per spec: L_d = n2/2^d, k_d = L_d/2."""  # see converters spec
    Ld = n2 // (2 ** d)
    kd = Ld // 2
    return Ld, kd


# --------------------------
# Pull exactly one batch, run models, print one trace + batch analytics
# --------------------------
for X, Y in tfds.take(1):
    # X: (gun_logits, field_logits, prev_out_t)
    # Y: (comm_tgt_list_t, field_tgt_list_t, gun_tgt_list_t, prev_meas_t, meas_tgt_t, out_tgt_t)
    gun_batch, field_batch, prev_out_t = X
    (comm_tgt_list_t,
     field_tgt_list_t,
     gun_tgt_list_t,
     prev_meas_t,
     meas_tgt_t,
     out_tgt_t) = Y

    # Basic geometry
    B = int(gun_batch.shape[0])
    n2 = int(gun_batch.shape[1])
    assert field_batch.shape[0] == B and field_batch.shape[1] == n2

    # Choose a valid index INSIDE this batch
    i = random.randrange(B)

    # -------- Single-trace slice (shape (1, ...))
    gun_1   = gun_batch[i:i+1]
    field_1 = field_batch[i:i+1]
    prev_out_1_list = [t[i:i+1] for t in prev_out_t]          # list length DEPTH
    # Targets (single-trace)
    comm_tgt_1_list  = [t[i:i+1] for t in comm_tgt_list_t]    # length DEPTH+1
    field_tgt_1_list = [t[i:i+1] for t in field_tgt_list_t]   # length DEPTH+1
    gun_tgt_1_list   = [t[i:i+1] for t in gun_tgt_list_t]     # length DEPTH+1
    prev_meas_tgt_1  = [t[i:i+1] for t in prev_meas_t]        # length DEPTH
    meas_tgt_1_list  = [t[i:i+1] for t in meas_tgt_t]         # length DEPTH
    out_tgt_1_list   = [t[i:i+1] for t in out_tgt_t]          # length DEPTH

    # -------- Run A then B on the SINGLE TRACE
    comm0_1, meas_a_1_list, out_a_1_list = model_a.compute_with_internal(
        field_logits=field_1,
        replay_out_a_logits_list=prev_out_1_list,
        training=True,
    )

    shoot_1, meas_b_1_list, out_b_1_list, comms_b_1_list, gun_b_1_list = model_b.compute_with_internal(
        gun_1,
        comm0_1,
        list(meas_a_1_list),
        list(out_a_1_list),
        training=True,
    )

    # --------------------------
    # Single-trace printout (compact + meaningful)
    # --------------------------
    print("\n====================")
    print(f"SINGLE TRACE (i={i} in batch of {B})")
    print("====================")
    print("Inputs:")
    print(" field_logits:", field_1.numpy())
    print(" gun_logits:  ", gun_1.numpy())
    print(" prev_out_logits_list:")
    for d in range(DEPTH):
        print(f"  prev_out[{d}]:", prev_out_1_list[d].numpy())

    print("\nA outputs vs targets:")
    # A comm is level-0 comm in this wiring
    report_line("A comm[d=0]", comm0_1, comm_tgt_1_list[0])
    for d in range(DEPTH):
        Ld, kd = level_sizes(n2, d)
        # meas/out are k_d wide; ensure consistent cropping
        report_line(f"A meas[d={d}]", crop_to(meas_a_1_list[d], kd), crop_to(prev_meas_tgt_1[d], kd))
        # out_a is "replay out"; compare to provided replay-out input for sanity, and to out_tgt if desired
        report_line(f"A out[d={d}] (vs prev_out)", crop_to(out_a_1_list[d], kd), crop_to(prev_out_1_list[d], kd))

    print("\nB outputs vs targets:")
    # shoot target is in your dataset as bit; convert to +/-1 logits target for sign-compare
    # (Any positive logit means bit=1)
    # If you have shoot bits in this tfds, add it to Y; otherwise skip.
    # Here we only print shoot magnitude:
    print("B shoot_logit:", shoot_1.numpy(), "  |shoot| mean:", float(mean_abs(shoot_1).numpy()))

    for d in range(DEPTH):
        Ld, kd = level_sizes(n2, d)
        report_line(f"B meas[d={d}]", crop_to(meas_b_1_list[d], kd), crop_to(meas_tgt_1_list[d], kd))
        report_line(f"B out[d={d}]",  crop_to(out_b_1_list[d], kd),  crop_to(out_tgt_1_list[d], kd))

    print("\nB internal comms/gun traces vs targets (where available):")
    for d in range(DEPTH + 1):
        Ld, _ = level_sizes(n2, d)
        report_line(f"B comm[d={d}]", comms_b_1_list[d], comm_tgt_1_list[d])
        report_line(f"B gun[d={d}]",  crop_to(gun_b_1_list[d], Ld), crop_to(gun_tgt_1_list[d], Ld))

    # --------------------------
    # Batch-wide analytics (first batch)
    # --------------------------
    print("\n====================")
    print("BATCH-WIDE SIGN-MATCH % (first batch)")
    print("====================")

    # Run A and B on the WHOLE BATCH (vectorized)
    # A: returns comm0 (B,1), meas/out lists of length DEPTH
    comm0_B, meas_a_B_list, out_a_B_list = model_a.compute_with_internal(
        field_logits=field_batch,
        replay_out_a_logits_list=list(prev_out_t),
        training=False,
    )

    # B: returns shoot (B,1) and per-level lists
    shoot_B, meas_b_B_list, out_b_B_list, comms_b_B_list, gun_b_B_list = model_b.compute_with_internal(
        gun_batch,
        comm0_B,
        list(meas_a_B_list),
        list(out_a_B_list),
        training=False,
    )

    # A comm level-0
    report_line("A comm[d=0] (batch)", comm0_B, comm_tgt_list_t[0])

    # A per-level
    for d in range(DEPTH):
        Ld, kd = level_sizes(n2, d)
        report_line(f"A meas[d={d}] (batch)", crop_to(meas_a_B_list[d], kd), crop_to(prev_meas_t[d], kd))
        # compare A out to prev_out input (replay consistency)
        report_line(f"A out[d={d}] vs prev_out (batch)", crop_to(out_a_B_list[d], kd), crop_to(prev_out_t[d], kd))

    # B per-level
    for d in range(DEPTH):
        Ld, kd = level_sizes(n2, d)
        report_line(f"B meas[d={d}] (batch)", crop_to(meas_b_B_list[d], kd), crop_to(meas_tgt_t[d], kd))
        report_line(f"B out[d={d}] (batch)",  crop_to(out_b_B_list[d], kd),  crop_to(out_tgt_t[d], kd))

    # B comm/gun traces per level
    for d in range(DEPTH + 1):
        Ld, _ = level_sizes(n2, d)
        report_line(f"B comm[d={d}] (batch)", comms_b_B_list[d], comm_tgt_list_t[d])
        report_line(f"B gun[d={d}] (batch)",  crop_to(gun_b_B_list[d], Ld), crop_to(gun_tgt_list_t[d], Ld))

    break

# %% [markdown]
# ## Next steps / experiments
# 
# - Sweep `W_COMMS_B` (e.g., supervise only deeper levels, or emphasize early levels).
# - Sweep `W_MEAS_IN_A`, `W_MEAS_OUT_A` per depth.
# - Try different `BETA_INPUT` (or a beta mixture view using your converter’s mixed-beta mode).
# - If you want *soft* guidance, replace sign-derived `tgt_bits` with `tf.sigmoid(tgt_logits)` for a “soft label” variant (still logit space),
#   but your stated preference was sign semantics.


