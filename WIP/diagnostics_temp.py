# %% [markdown]
# 
# # Pyramid diagnostics notebook (weights-loaded, no training)
# 
# This notebook was derived from `Pyr_train_models_diagnostics.ipynb` with the training workflow removed.
# 
# ## Analysis of the source notebook
# 
# **Diagnostics kept**
# - Random single-trace printout from the main regenerated dataset
# - The separate 100-game regenerated diagnostic dataset block
# - The detailed single-trace comparison printout
# - The batch-wide sign-match analytics
# 
# **Training-related content removed**
# - Optimizer setup
# - Loss / regularization helpers used only for training
# - `train_step`
# - Epoch loop / metrics accumulation
# - Checkpoint saving during training
# - Run-log persistence
# 
# **Helpers retained**
# - Repo-path / import setup
# - Settings and layout creation
# - Weight-file helpers and weight-load logic
# - Model build and force-build helpers
# - Dataset regeneration / conversion helpers needed by the diagnostics
# - Diagnostic helper functions for sign-match, MAE, shape cropping, and reporting
# 
# The notebook now:
# 1. builds model A and model B,
# 2. loads weights using the same pattern as `diagnostics.py`,
# 3. regenerates the datasets in-notebook,
# 4. runs only the diagnostics.
# 

# %%

from __future__ import annotations

import os
import sys
import random
from pathlib import Path
from typing import Any

# Optional: reduce oneDNN variability on some Windows installs.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def change_to_repo_root(marker: str = "WIP") -> Path:
    """Change CWD to repository root (where marker directory exists)."""
    here = Path.cwd()
    for parent in [here] + list(here.parents):
        if (parent / marker).is_dir():
            os.chdir(parent)
            return parent
    raise RuntimeError(f"Could not find repo root containing '{marker}/' starting from {here}")

ROOT = change_to_repo_root("WIP")
WIP_SRC  = ROOT / "WIP" / "src"
CORE_SRC = ROOT / "src"
WIP      = ROOT / "WIP"

for p in (WIP_SRC, CORE_SRC, WIP):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import numpy as np
import tensorflow as tf

try:
    from config import get_config
    CONFIG = dict(get_config())
    print("[config] loaded from config.py")
except Exception as e:
    print(f"[config] could not import config.py; using fallback defaults. Error: {e}")
    CONFIG = {
        "ROOT": str(ROOT),
        "FIELD_SIZE": 4,
        "N2": 16,
        "COMMS_SIZE": 1,
        "P_HIGH": 1.0,
        "NUM_GAMES_DATASET": 150_000,
        "SEED": 1234,
        "BETA_INPUT": 10.0,
        "BATCH": 256,
        "CHECKPOINT_DIR": str(ROOT / "WIP" / "checkpoints" / "weights_pyr_models"),
        "LOAD_WEIGHTS_ON_START": False,
        "MODEL_A_WEIGHTS_IN": None,
        "MODEL_B_WEIGHTS_IN": None,
    }

from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle_New.pyr_dataset_generation_utilities import generate_pyr_dataset
from Q_Sea_Battle_New.pyr_dataset_conversion_utilities import convert_all_traces
from Q_Sea_Battle_New.pyr_internal_model_a import PyrInternalModelA
from Q_Sea_Battle_New.pyr_internal_model_b import PyrInternalModelB


# %%

# -----------------------------
# Settings
# -----------------------------
FIELD_SIZE = int(CONFIG.get("FIELD_SIZE", int(np.sqrt(int(CONFIG.get("N2", 16))))))
N2 = int(CONFIG.get("N2", FIELD_SIZE * FIELD_SIZE))
COMMS_SIZE = int(CONFIG.get("COMMS_SIZE", 1))
P_HIGH = float(CONFIG.get("P_HIGH", 1.0))
NUM_GAMES_DATASET = 1000 # only for diagnostics, so keep small for speed; CONFIG.get("NUM_GAMES_DATASET", 150_000)
SEED = int(CONFIG.get("SEED", 1234))
BETA_INPUT = float(CONFIG.get("BETA_INPUT", 10.0))
BATCH = int(CONFIG.get("BATCH", 256))

CHECKPOINT_DIR = Path(CONFIG.get("CHECKPOINT_DIR", ROOT / "WIP" / "checkpoints" / "weights_pyr_models"))
LOAD_WEIGHTS_ON_START = bool(CONFIG.get("LOAD_WEIGHTS_ON_START", False))
MODEL_A_WEIGHTS_IN = CONFIG.get("MODEL_A_WEIGHTS_IN", None)
MODEL_B_WEIGHTS_IN = CONFIG.get("MODEL_B_WEIGHTS_IN", None)

LAYOUT = GameLayout(
    field_size=FIELD_SIZE,
    comms_size=COMMS_SIZE,
    number_of_games_in_tournament=1000,
    channel_noise=0.0,
    enemy_probability=0.5,
)

DEPTH = int(np.log2(N2))
assert 2 ** DEPTH == N2, "N2 must be a power of 2"

print("ROOT               :", ROOT)
print("FIELD_SIZE / N2    :", FIELD_SIZE, "/", N2)
print("COMMS_SIZE         :", COMMS_SIZE)
print("DEPTH              :", DEPTH)
print("NUM_GAMES_DATASET  :", NUM_GAMES_DATASET)
print("BETA_INPUT         :", BETA_INPUT)
print("CHECKPOINT_DIR     :", CHECKPOINT_DIR)
print("LOAD_WEIGHTS_START :", LOAD_WEIGHTS_ON_START)
print("MODEL_A_WEIGHTS_IN :", MODEL_A_WEIGHTS_IN)
print("MODEL_B_WEIGHTS_IN :", MODEL_B_WEIGHTS_IN)


# %%

# -----------------------------
# Weight helpers
# -----------------------------
import re

def _wfile(base_dir: Path, model_name: str, tag: str) -> Path:
    return Path(base_dir) / f"{model_name}_{tag}.weights.h5"

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

# -----------------------------
# Model builders + force-build helper
# -----------------------------
def build_model_a(config: dict[str, Any]) -> tf.keras.Model:
    return PyrInternalModelA(
        LAYOUT,
        sr_mode="replay",
        p_high=float(config.get("P_HIGH", P_HIGH)),
        beta=float(config.get("BETA_INPUT", BETA_INPUT)),
        alpha=5.0,
        seed=int(config.get("SEED", SEED)) + 10,
    )

def build_model_b(config: dict[str, Any]) -> tf.keras.Model:
    return PyrInternalModelB(
        LAYOUT,
        sr_mode="replay",
        p_high=float(config.get("P_HIGH", P_HIGH)),
        beta=float(config.get("BETA_INPUT", BETA_INPUT)),
        alpha=5.0,
    )

def _force_build_models_if_needed(model_a: tf.keras.Model, model_b: tf.keras.Model) -> None:
    # Force variable creation with one dummy forward pass.
    dummy_field = tf.zeros((1, N2), tf.float32)
    dummy_gun   = tf.zeros((1, N2), tf.float32)

    dummy_prev_out = [
        tf.zeros((1, N2 // (2 ** (d + 1))), tf.float32)
        for d in range(DEPTH)
    ]

    comm_logits0, meas_list0, out_list0 = model_a.compute_with_internal(
        field_logits=dummy_field,
        replay_out_a_logits_list=dummy_prev_out,
        harden_between_levels=False,
        training=False,
    )

    _ = model_b.compute_with_internal(
        dummy_gun,
        comm_logits0,
        list(meas_list0),
        list(out_list0),
        harden_between_levels=False,
        training=False,
    )

    print("[build] model A and B variables created.")


# %%

# -----------------------------
# Instantiate models and load weights
# -----------------------------
print("Composing internal models ...")
model_a = build_model_a(CONFIG)
model_b = build_model_b(CONFIG)
_force_build_models_if_needed(model_a, model_b)

model_a._ensure_built()
model_b._ensure_built()

loaded = False
if LOAD_WEIGHTS_ON_START:
    a_in = Path(MODEL_A_WEIGHTS_IN) if MODEL_A_WEIGHTS_IN else _wfile(CHECKPOINT_DIR, "model_a", "latest")
    b_in = Path(MODEL_B_WEIGHTS_IN) if MODEL_B_WEIGHTS_IN else _wfile(CHECKPOINT_DIR, "model_b", "latest")
    loaded = load_ab_weights(model_a, model_b, a_in, b_in)
    if not loaded:
        # Add WIP folder prefix and try again
        a_in = Path(ROOT / "WIP" / MODEL_A_WEIGHTS_IN) if MODEL_A_WEIGHTS_IN else _wfile(CHECKPOINT_DIR, "model_a", "latest")
        b_in = Path(ROOT /"WIP" / MODEL_B_WEIGHTS_IN) if MODEL_B_WEIGHTS_IN else _wfile(CHECKPOINT_DIR, "model_b", "latest")
        loaded = load_ab_weights(model_a, model_b, a_in, b_in)

if not loaded:
    a_auto, b_auto = _latest_epoch_pair(CHECKPOINT_DIR)
    loaded = load_ab_weights(model_a, model_b, a_auto, b_auto)

if not loaded:
    raise FileNotFoundError(
        "Could not load model A/B weights from configured or latest checkpoint files."
    )

print("[weights] models ready for diagnostics.")
# print current workin directory
print("[diagnostics] current working directory:", Path.cwd())

# %%

# -----------------------------
# Regenerate the main canonical dataset (bits) and logit views
# -----------------------------
ds_bits = generate_pyr_dataset(
    n2=N2,
    num_games=NUM_GAMES_DATASET,
    seed=SEED,
    validate=True,
)

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

field0 = tr["field"][0]
gun0 = tr["gun"][0]
shoot_tgt_logits = tr["shoot"]

meas_in_a_tgt_list = [tr["meas_in_a"][d] for d in range(DEPTH)]
meas_out_a_tgt_list = [tr["meas_out_a"][d] for d in range(DEPTH)]
comms_tgt_list = [tr["comms"][d] for d in range(DEPTH + 1)]

N = field0.shape[0]

tfds_train = tf.data.Dataset.from_tensor_slices((
    tf.convert_to_tensor(field0, tf.float32),
    tf.convert_to_tensor(gun0, tf.float32),
    tf.convert_to_tensor(shoot_tgt_logits, tf.float32),
    tuple(tf.convert_to_tensor(x, tf.float32) for x in meas_in_a_tgt_list),
    tuple(tf.convert_to_tensor(x, tf.float32) for x in meas_out_a_tgt_list),
    tuple(tf.convert_to_tensor(x, tf.float32) for x in comms_tgt_list),
))
tfds_train = (
    tfds_train
    .shuffle(min(N, 50_000), seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

print("[data] main regenerated dataset ready.")
print(" field0 shape:", field0.shape)
print(" gun0 shape  :", gun0.shape)
print(" shoot shape :", shoot_tgt_logits.shape)


# %%

# -----------------------------
# Diagnostic 1: random single trace from the main regenerated dataset
# -----------------------------
for (field_logits_b, gun_logits_b, shoot_tgt_logits_b, meas_in_a_tgt_b, meas_out_a_tgt_b, comms_tgt_b) in tfds_train:
    i = random.randint(0, int(field_logits_b.shape[0]) - 1)

    field_logits = field_logits_b[i:i+1]
    gun_logits = gun_logits_b[i:i+1]
    meas_out_a_tgt = [t[i:i+1] for t in meas_out_a_tgt_b]

    comm_logits, meas_list, out_list = model_a.compute_with_internal(
        field_logits=field_logits,
        replay_out_a_logits_list=meas_out_a_tgt,
        training=False,
    )

    shoot_logit, meas_b_logits_list, out_b_logits_list, comms_logits_list, gun_logits_list = model_b.compute_with_internal(
        gun_logits,
        comm_logits,
        list(meas_list),
        list(out_list),
        training=False,
    )
    break

print("field_logits\t\t\t", field_logits.numpy())
print("gun_logits\t\t\t", gun_logits.numpy())
print("shoot_logit\t\t\t", shoot_logit.numpy())
print("comm_logits\t\t\t", comm_logits.numpy())
for d in range(DEPTH):
    print(f"meas_list[{d}]\t\t\t", meas_list[d].numpy())
for d in range(DEPTH):
    print(f"out_list[{d}]\t\t\t", out_list[d].numpy())
for d in range(DEPTH):
    print(f"meas_b_logits_list[{d}]\t", meas_b_logits_list[d].numpy())
for d in range(DEPTH):
    print(f"out_b_logits_list[{d}]\t", out_b_logits_list[d].numpy())
for d in range(DEPTH + 1):
    print(f"comms_logits_list[{d}]\t\t", comms_logits_list[d].numpy())
for d in range(DEPTH + 1):
    print(f"gun_logits_list[{d}]\t\t", gun_logits_list[d].numpy())


# %%

# --------------------------
# Regenerate the dedicated 100-game diagnostic dataset
# --------------------------
ds_bits_diag = generate_pyr_dataset(n2=N2, num_games=100, seed=SEED, validate=True)

tr_diag = convert_all_traces(
    ds_bits_diag,
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

gun_logits_np = tr_diag["gun"][0]
field_logits_np = tr_diag["field"][0]
comm0_logits_np = tr_diag["comms"][0]
shoot_logits_np = tr_diag["shoot"]

prev_meas_list_np = [tr_diag["meas_in_a"][d] for d in range(DEPTH)]
prev_out_list_np = [tr_diag["meas_out_a"][d] for d in range(DEPTH)]

meas_b_list_np = [tr_diag["meas_in_b"][d] for d in range(DEPTH)]
out_b_list_np = [tr_diag["meas_out_b"][d] for d in range(DEPTH)]

field_logits_list_np = [tr_diag["field"][d] for d in range(DEPTH + 1)]
gun_logits_list_np = [tr_diag["gun"][d] for d in range(DEPTH + 1)]
comm_logits_list_np = [tr_diag["comms"][d] for d in range(DEPTH + 1)]

shoot_target_np = tr_diag["shoot"]

gun_logits_all = tf.constant(gun_logits_np, tf.float32)
field_logits_all = tf.constant(field_logits_np, tf.float32)
comm0_logits_all = tf.constant(comm0_logits_np, tf.float32)

prev_meas_t = tuple(tf.constant(a, tf.float32) for a in prev_meas_list_np)
prev_out_t = tuple(tf.constant(a, tf.float32) for a in prev_out_list_np)

meas_tgt_t = tuple(tf.constant(a, tf.float32) for a in meas_b_list_np)
out_tgt_t = tuple(tf.constant(a, tf.float32) for a in out_b_list_np)

gun_tgt_list_t = tuple(tf.constant(a, tf.float32) for a in gun_logits_list_np)
field_tgt_list_t = tuple(tf.constant(a, tf.float32) for a in field_logits_list_np)
comm_tgt_list_t = tuple(tf.constant(a, tf.float32) for a in comm_logits_list_np)

shoot_bits = tf.constant(shoot_target_np, tf.float32)

X_train = (gun_logits_all, field_logits_all, prev_out_t)
Y_train = (comm_tgt_list_t, field_tgt_list_t, gun_tgt_list_t, prev_meas_t, meas_tgt_t, out_tgt_t, shoot_bits)

tfds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
tfds = tfds.shuffle(200_000, seed=SEED, reshuffle_each_iteration=True)
tfds = tfds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

print("[data] 100-game diagnostic dataset ready.")


# %%

# --------------------------
# Diagnostic 2: detailed single-trace printout on the 100-game diagnostic dataset
# --------------------------
for X, Y in tfds:
    gun_logits_b, field_logits_b, prev_out_t_b = X
    comm_tgt_list_t_b, field_tgt_list_t_b, gun_tgt_list_t_b, prev_meas_t_b, meas_tgt_t_b, out_tgt_t_b, shoot_tgt_t_b = Y

    upper = int(gun_logits_b.shape[0]) - 1
    i = random.randint(0, max(0, min(90, upper)))

    field_logits = field_logits_b[i:i+1]
    gun_logits = gun_logits_b[i:i+1]
    prev_out_logits_list = [t[i:i+1] for t in prev_out_t_b]

    comm_logits_tgt = [t[i:i+1] for t in comm_tgt_list_t_b]
    field_logits_tgt = [t[i:i+1] for t in field_tgt_list_t_b]
    gun_logits_tgt = [t[i:i+1] for t in gun_tgt_list_t_b]
    prev_meas_tgt = [t[i:i+1] for t in prev_meas_t_b]
    meas_tgt = [t[i:i+1] for t in meas_tgt_t_b]
    out_tgt = [t[i:i+1] for t in out_tgt_t_b]

    comm_logits, meas_list, out_list = model_a.compute_with_internal(
        field_logits=field_logits,
        replay_out_a_logits_list=prev_out_logits_list,
        training=False,
    )

    shoot_logit, meas_b_logits_list, out_b_logits_list, comms_logits_list, gun_logits_list = model_b.compute_with_internal(
        gun_logits,
        comm_logits,
        list(meas_list),
        list(out_list),
        training=False,
    )
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
print(f"\nDepth: {DEPTH}")
print(f"comms_tgt_list[{DEPTH}]\t\t", comm_logits_tgt[DEPTH].numpy())
print(f"field_tgt_list[{DEPTH}]\t\t", field_logits_tgt[DEPTH].numpy())
print(f"gun_tgt_list[{DEPTH}]\t\t\t", gun_logits_tgt[DEPTH].numpy())
print("\nshoot_tgt\t\t", shoot_tgt_t_b[i].numpy())
print("\nPredictions:")
for d in range(DEPTH):
    print(f"\nDepth: {d}")
    print(f"comms_logits_list[{d}]\t\t", comms_logits_list[d].numpy())
    print(f"gun_logits_list[{d}]\t\t\t", gun_logits_list[d].numpy())
    print(f"prev_meas/model_a_meas[{d}]\t", meas_list[d].numpy())
    print(f"meas_b_logits_list[{d}]\t\t", meas_b_logits_list[d].numpy())
    print(f"out_b_logits_list[{d}]\t\t", out_b_logits_list[d].numpy())
print(f"\nDepth: {DEPTH}")
print(f"comms_logits_list[{DEPTH}]\t\t", comms_logits_list[DEPTH].numpy())
print(f"gun_logits_list[{DEPTH}]\t\t\t", gun_logits_list[DEPTH].numpy())
print("shoot_logit\t\t", shoot_logit.numpy())


# %%

import tensorflow as tf
import numpy as np
import random

# --------------------------
# Diagnostic helper functions
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
    tgt = tf.cast(tgt, tf.float32)
    return tf.reduce_mean(tf.abs(pred - tgt))

def report_line(name: str, pred: tf.Tensor, tgt: tf.Tensor):
    sm = float(sign_match_pct(pred, tgt).numpy())
    m = float(mae(pred, tgt).numpy())
    ap = float(mean_abs(pred).numpy())
    at = float(mean_abs(tgt).numpy())
    print(f"{name:30s} sign_match={sm:7.2f}%   mae={m:9.4f}   |pred|={ap:9.4f}   |tgt|={at:9.4f}")

def crop_to(t: tf.Tensor, width: int) -> tf.Tensor:
    """Crop last dim to width (safe if already <= width)."""
    width = int(width)
    return t[..., :width]

def level_sizes(n2: int, d: int):
    """(L_d, k_d) per spec: L_d = n2/2^d, k_d = L_d/2."""
    Ld = n2 // (2 ** d)
    kd = Ld // 2
    return Ld, kd


# %%

# --------------------------
# Diagnostic 3: single-trace + batch-wide analytics on the 100-game dataset
# --------------------------
for X, Y in tfds.take(1):
    gun_batch, field_batch, prev_out_batch = X
    (
        comm_tgt_list_batch,
        field_tgt_list_batch,
        gun_tgt_list_batch,
        prev_meas_batch,
        meas_tgt_batch,
        out_tgt_batch,
        shoot_tgt_batch
    ) = Y

    B = int(gun_batch.shape[0])
    n2 = int(gun_batch.shape[1])
    assert field_batch.shape[0] == B and field_batch.shape[1] == n2

    i = random.randrange(B)

    gun_1 = gun_batch[i:i+1]
    field_1 = field_batch[i:i+1]
    prev_out_1_list = [t[i:i+1] for t in prev_out_batch]

    comm_tgt_1_list = [t[i:i+1] for t in comm_tgt_list_batch]
    field_tgt_1_list = [t[i:i+1] for t in field_tgt_list_batch]
    gun_tgt_1_list = [t[i:i+1] for t in gun_tgt_list_batch]
    prev_meas_tgt_1 = [t[i:i+1] for t in prev_meas_batch]
    meas_tgt_1_list = [t[i:i+1] for t in meas_tgt_batch]
    out_tgt_1_list = [t[i:i+1] for t in out_tgt_batch]
    shoot_tgt_1 = shoot_tgt_batch[i:i+1]

    comm0_1, meas_a_1_list, out_a_1_list = model_a.compute_with_internal(
        field_logits=field_1,
        replay_out_a_logits_list=prev_out_1_list,
        training=False,
    )

    shoot_1, meas_b_1_list, out_b_1_list, comms_b_1_list, gun_b_1_list = model_b.compute_with_internal(
        gun_1,
        comm0_1,
        list(meas_a_1_list),
        list(out_a_1_list),
        training=False,
    )

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
    report_line("A comm[d=0]", comm0_1, comm_tgt_1_list[0])
    for d in range(DEPTH):
        Ld, kd = level_sizes(n2, d)
        report_line(f"A meas[d={d}]", crop_to(meas_a_1_list[d], kd), crop_to(prev_meas_tgt_1[d], kd))
        report_line(f"A out[d={d}] (vs prev_out)", crop_to(out_a_1_list[d], kd), crop_to(prev_out_1_list[d], kd))

    print("\nB outputs vs targets:")
    print("B shoot_logit:", shoot_1.numpy(), "  |shoot| mean:", float(mean_abs(shoot_1).numpy()))
    for d in range(DEPTH):
        Ld, kd = level_sizes(n2, d)
        report_line(f"B meas[d={d}]", crop_to(meas_b_1_list[d], kd), crop_to(meas_tgt_1_list[d], kd))
        report_line(f"B out[d={d}]", crop_to(out_b_1_list[d], kd), crop_to(out_tgt_1_list[d], kd))

    print("\nB internal comms/gun traces vs targets (where available):")
    for d in range(DEPTH + 1):
        Ld, _ = level_sizes(n2, d)
        report_line(f"B comm[d={d}]", comms_b_1_list[d], comm_tgt_1_list[d])
        report_line(f"B gun[d={d}]", crop_to(gun_b_1_list[d], Ld), crop_to(gun_tgt_1_list[d], Ld))

    print("\n====================")
    print("BATCH-WIDE SIGN-MATCH % (first batch)")
    print("====================")

    comm0_B, meas_a_B_list, out_a_B_list = model_a.compute_with_internal(
        field_logits=field_batch,
        replay_out_a_logits_list=list(prev_out_batch),
        training=False,
    )

    shoot_B, meas_b_B_list, out_b_B_list, comms_b_B_list, gun_b_B_list = model_b.compute_with_internal(
        gun_batch,
        comm0_B,
        list(meas_a_B_list),
        list(out_a_B_list),
        training=False,
    )
    report_line("B shoot_logit (batch)", shoot_B, shoot_tgt_batch)
    report_line("A comm[d=0] (batch)", comm0_B, comm_tgt_list_batch[0])

    for d in range(DEPTH):
        Ld, kd = level_sizes(n2, d)
        report_line(f"A meas[d={d}] (batch)", crop_to(meas_a_B_list[d], kd), crop_to(prev_meas_batch[d], kd))
        report_line(f"A out[d={d}] vs prev_out (batch)", crop_to(out_a_B_list[d], kd), crop_to(prev_out_batch[d], kd))

    for d in range(DEPTH):
        Ld, kd = level_sizes(n2, d)
        report_line(f"B meas[d={d}] (batch)", crop_to(meas_b_B_list[d], kd), crop_to(meas_tgt_batch[d], kd))
        report_line(f"B out[d={d}] (batch)", crop_to(out_b_B_list[d], kd), crop_to(out_tgt_batch[d], kd))

    for d in range(DEPTH + 1):
        Ld, _ = level_sizes(n2, d)
        report_line(f"B comm[d={d}] (batch)", comms_b_B_list[d], comm_tgt_list_batch[d])
        report_line(f"B gun[d={d}] (batch)", crop_to(gun_b_B_list[d], Ld), crop_to(gun_tgt_list_batch[d], Ld))

    break


# %%
from helpers import harden_ste
BETA_HARDEN = 10.0
# --------------------------
# Diagnostic 4: single-trace + batch-wide analytics on the 100-game dataset with binarization/hardening
# --------------------------
for X, Y in tfds.take(1):
    gun_batch, field_batch, prev_out_batch = X
    (
        comm_tgt_list_batch,
        field_tgt_list_batch,
        gun_tgt_list_batch,
        prev_meas_batch,
        meas_tgt_batch,
        out_tgt_batch,
        shoot_tgt_batch
    ) = Y

    B = int(gun_batch.shape[0])
    n2 = int(gun_batch.shape[1])
    assert field_batch.shape[0] == B and field_batch.shape[1] == n2

    i = random.randrange(B)

    gun_1 = gun_batch[i:i+1]
    field_1 = field_batch[i:i+1]
    prev_out_1_list = [t[i:i+1] for t in prev_out_batch]

    comm_tgt_1_list = [t[i:i+1] for t in comm_tgt_list_batch]
    field_tgt_1_list = [t[i:i+1] for t in field_tgt_list_batch]
    gun_tgt_1_list = [t[i:i+1] for t in gun_tgt_list_batch]
    prev_meas_tgt_1 = [t[i:i+1] for t in prev_meas_batch]
    meas_tgt_1_list = [t[i:i+1] for t in meas_tgt_batch]
    out_tgt_1_list = [t[i:i+1] for t in out_tgt_batch]
    shoot_tgt_1 = shoot_tgt_batch[i:i+1]

    comm0_1, meas_a_1_list, out_a_1_list = model_a.compute_with_internal(
        field_logits=field_1,
        replay_out_a_logits_list=prev_out_1_list,
        training=False,
    )

    comm0_1_hard = harden_ste(comm0_1, beta = BETA_HARDEN)
    meas_a_1_list_hard = [harden_ste(m, beta = BETA_HARDEN) for m in meas_a_1_list]
    out_a_1_list_hard = [harden_ste(o, beta = BETA_HARDEN) for o in out_a_1_list]

    shoot_1, meas_b_1_list, out_b_1_list, comms_b_1_list, gun_b_1_list = model_b.compute_with_internal(
        gun_1,
        comm0_1_hard,
        list(meas_a_1_list),
        list(out_a_1_list),
        training=False,
    )

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
    report_line("A comm[d=0]", comm0_1, comm_tgt_1_list[0])
    for d in range(DEPTH):
        Ld, kd = level_sizes(n2, d)
        report_line(f"A meas[d={d}]", crop_to(meas_a_1_list[d], kd), crop_to(prev_meas_tgt_1[d], kd))
        report_line(f"A out[d={d}] (vs prev_out)", crop_to(out_a_1_list[d], kd), crop_to(prev_out_1_list[d], kd))

    print("\nB outputs vs targets:")
    print("B shoot_logit:", shoot_1.numpy(), "  |shoot| mean:", float(mean_abs(shoot_1).numpy()))
    for d in range(DEPTH):
        Ld, kd = level_sizes(n2, d)
        report_line(f"B meas[d={d}]", crop_to(meas_b_1_list[d], kd), crop_to(meas_tgt_1_list[d], kd))
        report_line(f"B out[d={d}]", crop_to(out_b_1_list[d], kd), crop_to(out_tgt_1_list[d], kd))

    print("\nB internal comms/gun traces vs targets (where available):")
    for d in range(DEPTH + 1):
        Ld, _ = level_sizes(n2, d)
        report_line(f"B comm[d={d}]", comms_b_1_list[d], comm_tgt_1_list[d])
        report_line(f"B gun[d={d}]", crop_to(gun_b_1_list[d], Ld), crop_to(gun_tgt_1_list[d], Ld))

    print("\n====================")
    print("BATCH-WIDE SIGN-MATCH % (first batch)")
    print("====================")

    comm0_B, meas_a_B_list, out_a_B_list = model_a.compute_with_internal(
        field_logits=field_batch,
        replay_out_a_logits_list=list(prev_out_batch),
        training=False,
    )

    comm0_B_hard = harden_ste(comm0_B, beta = BETA_HARDEN)
    meas_a_B_list_hard = [harden_ste(m, beta = BETA_HARDEN) for m in meas_a_B_list]
    out_a_B_list_hard = [harden_ste(o, beta = BETA_HARDEN) for o in out_a_B_list]

    shoot_B, meas_b_B_list, out_b_B_list, comms_b_B_list, gun_b_B_list = model_b.compute_with_internal(
        gun_batch,
        comm0_B_hard,
        list(meas_a_1_list),
        list(out_a_1_list),
        training=False,
    )
    report_line("B shoot_logit (batch)", shoot_B, shoot_tgt_batch)
    report_line("A comm[d=0] (batch)", comm0_B, comm_tgt_list_batch[0])

    for d in range(DEPTH):
        Ld, kd = level_sizes(n2, d)
        report_line(f"A meas[d={d}] (batch)", crop_to(meas_a_B_list[d], kd), crop_to(prev_meas_batch[d], kd))
        report_line(f"A out[d={d}] vs prev_out (batch)", crop_to(out_a_B_list[d], kd), crop_to(prev_out_batch[d], kd))

    for d in range(DEPTH):
        Ld, kd = level_sizes(n2, d)
        report_line(f"B meas[d={d}] (batch)", crop_to(meas_b_B_list[d], kd), crop_to(meas_tgt_batch[d], kd))
        report_line(f"B out[d={d}] (batch)", crop_to(out_b_B_list[d], kd), crop_to(out_tgt_batch[d], kd))

    for d in range(DEPTH + 1):
        Ld, _ = level_sizes(n2, d)
        report_line(f"B comm[d={d}] (batch)", comms_b_B_list[d], comm_tgt_list_batch[d])
        report_line(f"B gun[d={d}] (batch)", crop_to(gun_b_B_list[d], Ld), crop_to(gun_tgt_list_batch[d], Ld))

    break


# %% [markdown]
# 
# ## Notes
# 
# - This notebook intentionally does **not** train.
# - It assumes the configured model-weight files exist.
# - The diagnostics still regenerate the datasets in-notebook, matching the source notebook's approach.
# 


