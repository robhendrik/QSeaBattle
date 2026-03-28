from __future__ import annotations

import ast
import os
import pickle
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keep this before importing tensorflow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


# -----------------------------
# Config helpers
# -----------------------------

def _parse_cfg_value(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    low = text.lower()
    if low == "none":
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def load_txt_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    config: dict[str, Any] = {}
    for lineno, line in enumerate(cfg_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ValueError(f"Invalid config line {lineno}: {line!r}. Expected KEY = VALUE.")
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid config line {lineno}: empty key.")
        config[key] = _parse_cfg_value(value)
    return config


def merge_with_defaults(user_cfg: dict[str, Any]) -> dict[str, Any]:
    root_default = Path.cwd().resolve()
    defaults: dict[str, Any] = {
        "ROOT": str(root_default),
        "FIELD_SIZE": 4,
        "N2": 16,
        "COMMS_SIZE": 1,
        "P_HIGH": 1.0,
        "NUM_GAMES_DATASET": 150_000,
        "SEED": 1234,
        "BETA_INPUT": 10.0,
        "BATCH": 256,
        "ALPHA_FOR_SR_LAYER": 5.0,
        "CHECKPOINT_DIR": str(root_default / "WIP" / "checkpoints" / "weights_pyr_models"),
        "MODEL_A_WEIGHTS_IN": None,
        "MODEL_B_WEIGHTS_IN": None,
        "LOAD_LATEST_IF_NONE": True,
        "HARDEN_COMMS": True,
        "HARDEN_MEAS": True,
        "HARDEN_OUTS": True,
        "MAX_GOOD_TRACES": None,
        "MAX_BAD_TRACES": None,
        "OUTPUT_DIR": str(root_default / "WIP" / "diagnostics_logs"),
        "GOOD_LOG_FILENAME": "diagnostics_good.pkl",
        "BAD_LOG_FILENAME": "diagnostics_bad.pkl",
        "SAVE_BAD_DATASET": True,
        "BAD_DATASET_DIR": str(root_default / "WIP" / "diagnostics_logs" / "diagnostics_bad_raw_ds"),
        "FLUSH_EVERY": 500,
    }
    defaults.update(user_cfg)
    return defaults


# -----------------------------
# Repo / imports
# -----------------------------

def _ensure_repo_paths(root: Path) -> None:
    for p in (root / "WIP" / "src", root / "src", root / "WIP"):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


# -----------------------------
# Weight helpers copied from notebook
# -----------------------------

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


# -----------------------------
# Runtime data structures
# -----------------------------

@dataclass
class RuntimeContext:
    config: dict[str, Any]
    root: Path
    output_dir: Path
    checkpoint_dir: Path
    depth: int
    n2: int
    beta_input: float
    model_a: tf.keras.Model
    model_b: tf.keras.Model


# -----------------------------
# Model builders copied from notebook style, with alpha as config
# -----------------------------

def build_model_a(config: dict[str, Any], layout) -> tf.keras.Model:
    from Q_Sea_Battle_New.pyr_internal_model_a import PyrInternalModelA

    return PyrInternalModelA(
        layout,
        sr_mode="replay",
        p_high=float(config.get("P_HIGH", 1.0)),
        beta=float(config.get("BETA_INPUT", 10.0)),
        alpha=float(config.get("ALPHA_FOR_SR_LAYER", 5.0)),
        seed=int(config.get("SEED", 1234)) + 10,
    )



def build_model_b(config: dict[str, Any], layout) -> tf.keras.Model:
    from Q_Sea_Battle_New.pyr_internal_model_b import PyrInternalModelB

    return PyrInternalModelB(
        layout,
        sr_mode="replay",
        p_high=float(config.get("P_HIGH", 1.0)),
        beta=float(config.get("BETA_INPUT", 10.0)),
        alpha=float(config.get("ALPHA_FOR_SR_LAYER", 5.0)),
    )



def _force_build_models_if_needed(model_a: tf.keras.Model, model_b: tf.keras.Model, n2: int, depth: int) -> None:
    dummy_field = tf.zeros((1, n2), tf.float32)
    dummy_gun = tf.zeros((1, n2), tf.float32)
    dummy_prev_out = [tf.zeros((1, n2 // (2 ** (d + 1))), tf.float32) for d in range(depth)]

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
    model_a._ensure_built()
    model_b._ensure_built()

# -----------------------------
# Diagnostics utilities
# -----------------------------

def _to_numpy(x: Any) -> np.ndarray:
    return tf.convert_to_tensor(x).numpy()



def _tensor_list_to_numpy(seq: list[tf.Tensor]) -> list[np.ndarray]:
    return [_to_numpy(x) for x in seq]



def harden_logits(logits: tf.Tensor, beta: float) -> tf.Tensor:
    logits = tf.cast(logits, tf.float32)
    beta_t = tf.cast(beta, tf.float32)
    return tf.where(logits >= 0.0, beta_t, -beta_t)



def maybe_harden_interface(
    comm_logits: tf.Tensor,
    meas_list: list[tf.Tensor],
    out_list: list[tf.Tensor],
    *,
    harden_comms: bool,
    harden_meas: bool,
    harden_outs: bool,
    beta: float,
) -> tuple[tf.Tensor, list[tf.Tensor], list[tf.Tensor]]:
    comm_for_b = harden_logits(comm_logits, beta) if harden_comms else tf.identity(comm_logits)
    meas_for_b = [harden_logits(t, beta) if harden_meas else tf.identity(t) for t in meas_list]
    out_for_b = [harden_logits(t, beta) if harden_outs else tf.identity(t) for t in out_list]
    return comm_for_b, meas_for_b, out_for_b



def sign01_from_logits(x: tf.Tensor | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return (arr >= 0.0).astype(np.float32)



def _resolve_weight_paths(config: dict[str, Any], checkpoint_dir: Path) -> tuple[Path | None, Path | None]:
    def _resolve_one(value: Any) -> Path | None:
        if value is None:
            return None
        p = Path(str(value))
        return p if p.is_absolute() else Path(config["ROOT"]) / p

    a_path = _resolve_one(config.get("MODEL_A_WEIGHTS_IN"))
    b_path = _resolve_one(config.get("MODEL_B_WEIGHTS_IN"))

    if a_path is None and b_path is None and bool(config.get("LOAD_LATEST_IF_NONE", True)):
        latest_a, latest_b = _latest_epoch_pair(checkpoint_dir)
        return latest_a, latest_b
    return a_path, b_path



def generate_dataset(config: dict[str, Any], depth: int) -> dict[str, Any]:
    from Q_Sea_Battle_New.pyr_dataset_generation_utilities import generate_pyr_dataset
    from Q_Sea_Battle_New.pyr_dataset_conversion_utilities import convert_all_traces

    ds_bits = generate_pyr_dataset(
        n2=int(config["N2"]),
        num_games=int(config["NUM_GAMES_DATASET"]),
        seed=int(config["SEED"]),
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
        beta=float(config["BETA_INPUT"]),
    )

    required_keys = ["field", "gun", "shoot", "meas_in_a", "meas_out_a", "comms"]
    missing = [k for k in required_keys if k not in tr]
    if missing:
        raise KeyError(f"Converted trace dict is missing keys: {missing}")
    if len(tr["meas_out_a"]) != depth:
        raise ValueError(f"Expected meas_out_a length {depth}, got {len(tr['meas_out_a'])}")
    return tr



def run_diagnostics(ctx: RuntimeContext) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = ctx.config
    tr = generate_dataset(cfg, ctx.depth)

    field = np.asarray(tr["field"][0], dtype=np.float32)
    gun = np.asarray(tr["gun"][0], dtype=np.float32)
    shoot_tgt = np.asarray(tr["shoot"], dtype=np.float32)
    meas_in_a_tgt_list = [np.asarray(tr["meas_in_a"][d], dtype=np.float32) for d in range(ctx.depth)]
    meas_out_a_tgt_list = [np.asarray(tr["meas_out_a"][d], dtype=np.float32) for d in range(ctx.depth)]
    comms_tgt_list = [np.asarray(tr["comms"][d], dtype=np.float32) for d in range(ctx.depth + 1)]

    num_games = int(field.shape[0])
    max_good = cfg.get("MAX_GOOD_TRACES")
    max_bad = cfg.get("MAX_BAD_TRACES")
    flush_every = int(cfg.get("FLUSH_EVERY", 500))

    good_traces: list[dict[str, Any]] = []
    bad_traces: list[dict[str, Any]] = []

    bad_field_items: list[np.ndarray] = []
    bad_gun_items: list[np.ndarray] = []
    bad_shoot_items: list[np.ndarray] = []
    bad_meas_in_a_items: list[list[np.ndarray]] = [[] for _ in range(ctx.depth)]
    bad_meas_out_a_items: list[list[np.ndarray]] = [[] for _ in range(ctx.depth)]
    bad_comms_items: list[list[np.ndarray]] = [[] for _ in range(ctx.depth + 1)]

    good_count = 0
    bad_count = 0
    shoot_margin_sum = 0.0
    comm_abs_sum = 0.0
    meas_abs_sum = np.zeros((ctx.depth,), dtype=np.float64)

    harden_comms = bool(cfg.get("HARDEN_COMMS", True))
    harden_meas = bool(cfg.get("HARDEN_MEAS", True))
    harden_outs = bool(cfg.get("HARDEN_OUTS", True))

    for i in range(num_games):
        field_i = tf.convert_to_tensor(field[i : i + 1], dtype=tf.float32)
        gun_i = tf.convert_to_tensor(gun[i : i + 1], dtype=tf.float32)
        meas_in_a_tgt_i = [tf.convert_to_tensor(meas_in_a_tgt_list[d][i : i + 1], dtype=tf.float32) for d in range(ctx.depth)]
        meas_out_i = [tf.convert_to_tensor(meas_out_a_tgt_list[d][i : i + 1], dtype=tf.float32) for d in range(ctx.depth)]
        shoot_tgt_i = tf.convert_to_tensor(shoot_tgt[i : i + 1], dtype=tf.float32)
        comms_tgt_i = [tf.convert_to_tensor(comms_tgt_list[d][i : i + 1], dtype=tf.float32) for d in range(ctx.depth + 1)]

        comm_logits, meas_list, out_list = ctx.model_a.compute_with_internal(
            field_logits=field_i,
            replay_out_a_logits_list=meas_out_i,
            training=False,
        )

        comm_for_b, meas_for_b, out_for_b = maybe_harden_interface(
            comm_logits,
            list(meas_list),
            list(out_list),
            harden_comms=harden_comms,
            harden_meas=harden_meas,
            harden_outs=harden_outs,
            beta=ctx.beta_input,
        )

        shoot_logit, meas_b_logits_list, out_b_logits_list, comms_b_logits_list, gun_b_logits_list = (
            ctx.model_b.compute_with_internal(
                gun_i,
                comm_for_b,
                list(meas_for_b),
                list(out_for_b),
                training=False,
            )
        )

        pred_ok = bool(np.array_equal(sign01_from_logits(shoot_logit), sign01_from_logits(shoot_tgt_i)))
        good_count += int(pred_ok)
        bad_count += int(not pred_ok)
        shoot_margin_sum += float(np.mean(np.abs(_to_numpy(shoot_logit))))
        comm_abs_sum += float(np.mean(np.abs(_to_numpy(comm_logits))))
        for d, meas_t in enumerate(meas_list):
            meas_abs_sum[d] += float(np.mean(np.abs(_to_numpy(meas_t))))

        trace = {
            "id": i,
            "prediction_right": pred_ok,
            "hardening": {
                "HARDEN_COMMS": harden_comms,
                "HARDEN_MEAS": harden_meas,
                "HARDEN_OUTS": harden_outs,
                "beta": ctx.beta_input,
            },
            "inputs": {
                "field": _to_numpy(field_i),
                "gun": _to_numpy(gun_i),
                "prev_out": _tensor_list_to_numpy(meas_out_i),
            },
            "teacher": {
                "shoot": _to_numpy(shoot_tgt_i),
                "comms": _tensor_list_to_numpy(comms_tgt_i),
                "meas_in_a": _tensor_list_to_numpy(meas_in_a_tgt_i),
                "meas_out_a": _tensor_list_to_numpy(meas_out_i),
            },
            "model_a": {
                "comm_logits": _to_numpy(comm_logits),
                "meas_list": _tensor_list_to_numpy(list(meas_list)),
                "out_list": _tensor_list_to_numpy(list(out_list)),
            },
            "interface_to_b": {
                "comm": _to_numpy(comm_for_b),
                "meas_list": _tensor_list_to_numpy(list(meas_for_b)),
                "out_list": _tensor_list_to_numpy(list(out_for_b)),
            },
            "model_b": {
                "shoot_logit": _to_numpy(shoot_logit),
                "meas_b_logits_list": _tensor_list_to_numpy(list(meas_b_logits_list)),
                "out_b_logits_list": _tensor_list_to_numpy(list(out_b_logits_list)),
                "comms_b_logits_list": _tensor_list_to_numpy(list(comms_b_logits_list)),
                "gun_b_logits_list": _tensor_list_to_numpy(list(gun_b_logits_list)),
            },
        }

        if pred_ok:
            if max_good is None or len(good_traces) < int(max_good):
                good_traces.append(trace)
        else:
            if max_bad is None or len(bad_traces) < int(max_bad):
                bad_traces.append(trace)
            bad_field_items.append(np.asarray(field[i], dtype=np.float32))
            bad_gun_items.append(np.asarray(gun[i], dtype=np.float32))
            bad_shoot_items.append(np.asarray(shoot_tgt[i], dtype=np.float32))
            for d in range(ctx.depth):
                bad_meas_in_a_items[d].append(np.asarray(meas_in_a_tgt_list[d][i], dtype=np.float32))
                bad_meas_out_a_items[d].append(np.asarray(meas_out_a_tgt_list[d][i], dtype=np.float32))
            for d in range(ctx.depth + 1):
                bad_comms_items[d].append(np.asarray(comms_tgt_list[d][i], dtype=np.float32))

        if ((i + 1) % flush_every == 0) or (i + 1 == num_games):
            acc = (good_count / (i + 1)) * 100.0
            print(
                f"[progress] {i + 1}/{num_games}  good={good_count}  bad={bad_count}  accuracy={acc:.2f}%"
            )

    mean_meas_abs = (meas_abs_sum / max(num_games, 1)).tolist()
    summary = {
        "num_games": num_games,
        "good_count": good_count,
        "bad_count": bad_count,
        "accuracy": good_count / max(num_games, 1),
        "mean_abs_shoot_logit": shoot_margin_sum / max(num_games, 1),
        "mean_abs_comm_logits": comm_abs_sum / max(num_games, 1),
        "mean_abs_meas_logits_per_level": mean_meas_abs,
        "hardening": {
            "HARDEN_COMMS": harden_comms,
            "HARDEN_MEAS": harden_meas,
            "HARDEN_OUTS": harden_outs,
            "beta": ctx.beta_input,
        },
        "alpha_for_sr_layer": float(cfg["ALPHA_FOR_SR_LAYER"]),
    }

    good_payload = {
        "metadata": summary | {"bucket": "good"},
        "config": dict(cfg),
        "traces": good_traces,
    }
    bad_payload = {
        "metadata": summary | {"bucket": "bad"},
        "config": dict(cfg),
        "traces": bad_traces,
    }

    bad_raw_ds_source = {
        "field": bad_field_items,
        "gun": bad_gun_items,
        "shoot": bad_shoot_items,
        "meas_in_a": bad_meas_in_a_items,
        "meas_out_a": bad_meas_out_a_items,
        "comms": bad_comms_items,
    }
    return good_payload, bad_payload, bad_raw_ds_source



def save_pickle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)



def _stack_or_empty(seq: list[np.ndarray], width: int) -> tf.Tensor:
    if seq:
        arr = np.stack(seq, axis=0).astype(np.float32, copy=False)
    else:
        arr = np.zeros((0, width), dtype=np.float32)
    return tf.convert_to_tensor(arr, tf.float32)



def save_bad_raw_dataset(path: Path, bad_raw_ds_source: dict[str, Any], depth: int, n2: int) -> tuple[Path, int]:
    field_t = _stack_or_empty(bad_raw_ds_source["field"], n2)
    gun_t = _stack_or_empty(bad_raw_ds_source["gun"], n2)
    shoot_t = _stack_or_empty(bad_raw_ds_source["shoot"], 1)
    meas_in_a_t = tuple(_stack_or_empty(bad_raw_ds_source["meas_in_a"][d], n2 // (2 ** (d + 1))) for d in range(depth))
    meas_out_a_t = tuple(_stack_or_empty(bad_raw_ds_source["meas_out_a"][d], n2 // (2 ** (d + 1))) for d in range(depth))
    comms_t = tuple(_stack_or_empty(bad_raw_ds_source["comms"][d], max(1, n2 // (2 ** d))) for d in range(depth + 1))

    raw_ds = tf.data.Dataset.from_tensor_slices((field_t, gun_t, shoot_t, meas_in_a_t, meas_out_a_t, comms_t))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        import shutil
        shutil.rmtree(path)
    raw_ds.save(str(path))
    return path, int(field_t.shape[0])


def print_summary(good_payload: dict[str, Any], bad_payload: dict[str, Any], good_path: Path, bad_path: Path, bad_ds_path: Path | None = None, bad_ds_count: int | None = None) -> None:
    meta = good_payload["metadata"]
    print("\n=== Diagnostics summary ===")
    print(f"games                : {meta['num_games']}")
    print(f"good / bad           : {meta['good_count']} / {meta['bad_count']}")
    print(f"accuracy             : {100.0 * float(meta['accuracy']):.2f}%")
    print(f"mean |shoot_logit|   : {float(meta['mean_abs_shoot_logit']):.4f}")
    print(f"mean |comm_logits|   : {float(meta['mean_abs_comm_logits']):.4f}")
    print(f"mean |meas| per lvl  : {meta['mean_abs_meas_logits_per_level']}")
    print(f"hardening            : {meta['hardening']}")
    print(f"alpha                : {meta['alpha_for_sr_layer']}")
    print(f"saved good traces    : {len(good_payload['traces'])} -> {good_path}")
    print(f"saved bad traces     : {len(bad_payload['traces'])} -> {bad_path}")
    if bad_ds_path is not None and bad_ds_count is not None:
        print(f"saved bad raw_ds     : {bad_ds_count} -> {bad_ds_path}")



def build_runtime(cfg_path: str | Path) -> RuntimeContext:
    user_cfg = load_txt_config(cfg_path)
    config = merge_with_defaults(user_cfg)

    root = Path(str(config["ROOT"])).resolve()
    _ensure_repo_paths(root)

    from Q_Sea_Battle.game_layout import GameLayout

    random.seed(int(config["SEED"]))
    np.random.seed(int(config["SEED"]))
    tf.random.set_seed(int(config["SEED"]))

    n2 = int(config["N2"])
    depth = int(np.log2(n2))
    if 2 ** depth != n2:
        raise ValueError(f"N2 must be a power of 2, got {n2}.")

    layout = GameLayout(
        field_size=int(config["FIELD_SIZE"]),
        comms_size=int(config["COMMS_SIZE"]),
        number_of_games_in_tournament=1000,
        channel_noise=0.0,
        enemy_probability=0.5,
    )

    model_a = build_model_a(config, layout)
    model_b = build_model_b(config, layout)
    _force_build_models_if_needed(model_a, model_b, n2=n2, depth=depth)

    checkpoint_dir = Path(str(config["CHECKPOINT_DIR"])).resolve()
    a_in, b_in = _resolve_weight_paths(config, checkpoint_dir)
    loaded = load_ab_weights(model_a, model_b, a_in, b_in)

    if not loaded:
        raise FileNotFoundError(
            "Could not load model weights. Set MODEL_A_WEIGHTS_IN and MODEL_B_WEIGHTS_IN in the cfg, "
            "or make sure CHECKPOINT_DIR contains a latest matching epoch pair."
        )

    output_dir = Path(str(config["OUTPUT_DIR"])).resolve()
    return RuntimeContext(
        config=config,
        root=root,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        depth=depth,
        n2=n2,
        beta_input=float(config["BETA_INPUT"]),
        model_a=model_a,
        model_b=model_b,
    )



def main() -> int:
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("WIP/diagnostics_cfg.txt")
    print(f"[config] using {cfg_path.resolve()}")
    ctx = build_runtime(cfg_path)
    good_payload, bad_payload, bad_raw_ds_source = run_diagnostics(ctx)

    good_path = ctx.output_dir / str(ctx.config["GOOD_LOG_FILENAME"])
    bad_path = ctx.output_dir / str(ctx.config["BAD_LOG_FILENAME"])
    save_pickle(good_path, good_payload)
    save_pickle(bad_path, bad_payload)

    bad_ds_path = None
    bad_ds_count = None
    if bool(ctx.config.get("SAVE_BAD_DATASET", True)):
        bad_ds_path = Path(str(ctx.config.get("BAD_DATASET_DIR", ctx.output_dir / "diagnostics_bad_raw_ds")))
        if not bad_ds_path.is_absolute():
            bad_ds_path = ctx.root / bad_ds_path
        bad_ds_path, bad_ds_count = save_bad_raw_dataset(bad_ds_path, bad_raw_ds_source, ctx.depth, ctx.n2)

    print_summary(good_payload, bad_payload, good_path, bad_path, bad_ds_path, bad_ds_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
