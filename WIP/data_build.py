from __future__ import annotations

from pathlib import Path
from typing import Any
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

from config import get_config

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Any


def _ensure_repo_paths(config: dict[str, Any]) -> None:
    """Ensure WIP/src and src are importable (same pattern as training script)."""
    root = Path(config.get("ROOT", Path.cwd())).resolve()
    wip_src = root / "WIP" / "src"
    core_src = root / "src"
    wip = root / "WIP"

    for p in (wip_src, core_src, wip):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def _get_source_tensors(config: dict[str, Any]) -> tuple[Any, ...]:
    """
    Build source tensors exactly like the notebook/script dataset block.
    """
    _ensure_repo_paths(config)

    from Q_Sea_Battle_New.pyr_dataset_generation_utilities import generate_pyr_dataset
    from Q_Sea_Battle_New.pyr_dataset_conversion_utilities import convert_all_traces

    N2 = int(config["N2"])
    NUM_GAMES_DATASET = int(config["NUM_GAMES_DATASET"])
    SEED = int(config["SEED"])
    BETA_INPUT = float(config["BETA_INPUT"])

    DEPTH = int(config.get("DEPTH", np.log2(N2)))
    if 2 ** DEPTH != N2:
        raise ValueError(f"N2 must be power of 2, got N2={N2}, DEPTH={DEPTH}")

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

    field0 = tr["field"][0]                  # (N, n2)
    gun0 = tr["gun"][0]                      # (N, n2)
    shoot_tgt_logits = tr["shoot"]           # (N, 1)
    meas_in_a_tgt_list = [tr["meas_in_a"][d] for d in range(DEPTH)]
    meas_out_a_tgt_list = [tr["meas_out_a"][d] for d in range(DEPTH)]
    comms_tgt_list = [tr["comms"][d] for d in range(DEPTH + 1)]

    return (
        tf.convert_to_tensor(field0, tf.float32),
        tf.convert_to_tensor(gun0, tf.float32),
        tf.convert_to_tensor(shoot_tgt_logits, tf.float32),
        tuple(tf.convert_to_tensor(x, tf.float32) for x in meas_in_a_tgt_list),
        tuple(tf.convert_to_tensor(x, tf.float32) for x in meas_out_a_tgt_list),
        tuple(tf.convert_to_tensor(x, tf.float32) for x in comms_tgt_list),
    )

def _dataset_path(config: dict[str, Any]) -> Path:
    return Path(config["DATASET_DIR"])




def build_and_save_dataset(config: dict[str, Any]) -> Path:
    """
    Build `raw_ds` from source tensors and save it to DATASET_DIR using tf.data.Dataset.save().
    """
    save_dir = _dataset_path(config)
    save_dir.mkdir(parents=True, exist_ok=True)

    source = _get_source_tensors(config)

    # Keep dataset construction exactly aligned with notebook ordering/structure.
    raw_ds = tf.data.Dataset.from_tensor_slices(source)
    raw_ds.save(str(save_dir))

    print(f"[dataset] saved raw_ds -> {save_dir}")
    return save_dir


def load_dataset(config: dict[str, Any]) -> tf.data.Dataset:
    """
    Load previously saved raw dataset from DATASET_DIR.
    """
    save_dir = _dataset_path(config)
    if not save_dir.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {save_dir}")

    raw_ds = tf.data.Dataset.load(str(save_dir))
    return raw_ds


def build_train_pipeline(raw_ds: tf.data.Dataset, config: dict[str, Any], N: int | None = None) -> tf.data.Dataset:
    """
    Build training pipeline from raw_ds, preserving notebook behavior as closely as possible.
    """
    BATCH = int(config["BATCH"])
    SEED = int(config["SEED"])
    shuffle_n = int(min(N, 50_000)) if N is not None else 50_000

    tfds_train = (
        raw_ds
        .shuffle(shuffle_n, seed=SEED, reshuffle_each_iteration=True)
        .batch(BATCH, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    return tfds_train


if __name__ == "__main__":
    cfg = get_config()
    build_and_save_dataset(cfg)