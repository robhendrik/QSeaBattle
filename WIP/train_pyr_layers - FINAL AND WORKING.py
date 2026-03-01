"""
train_pyr_layers.py

Train selected Pyramid layers (Measure/Combine A/B) and save per-layer weights files,
without overwriting weights for layers you did not train.

This is extracted from `debug_layers_and_models.ipynb` and kept import-friendly:
- You can `import train_pyr_layers as tpl` and call functions directly.
- Or run as a script: `python train_pyr_layers.py --help`

Key conventions (matches current notebook):
- Training targets are bits {0,1} (float32) with BCE(from_logits=True), except gun_next which is categorical.
- Layer outputs are logits; semantic bit is 1[logit >= 0] (adapter convention).
- Inputs are produced by converters; typically hard_logit for internal wires.

Notes:
- The converter spec in this project also supports logit-valued targets ("hard_logit").
  This script follows the notebook choice (bit targets). If you switch targets to hard_logit,
  you should also switch loss/metrics accordingly.

"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import os

# Silence TensorFlow C++ logs (INFO + WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=INFO off, 2=WARNING off, 3=ERROR only

# Optional: reduce Python-side TF logging
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf

LayerKind = Literal["comb_b"]#["meas_a", "comb_a", "meas_b"]#, "comb_b"]


# ----------------------------
# Repo path helper (optional)
# ----------------------------
def change_to_repo_root(marker: str = "WIP") -> None:
    """
    Change CWD to repository root by walking up until a directory containing `marker` is found.
    This mirrors the notebook's approach and makes imports work when launched from arbitrary folders.
    """
    here = Path.cwd()
    for parent in [here] + list(here.parents):
        if (parent / marker).is_dir():
            os.chdir(parent)
            return


def add_repo_to_syspath() -> None:
    """
    Add repo folders to sys.path so imports work regardless of launch location.

    This fixes packages that (incorrectly) import via `WIP.src...` by ensuring the
    repo root itself is on sys.path, making `import WIP...` resolvable.

    Adds (in this order if they exist):
      1) ROOT               (so `import WIP...` works)
      2) ROOT/WIP/src       (so `import Q_Sea_Battle_New...` works)
      3) ROOT/src           (optional core src)
    """
    root = Path.cwd()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    wip_src = root / "WIP" / "src"
    core_src = root / "src"
    if wip_src.is_dir() and str(wip_src) not in sys.path:
        sys.path.insert(0, str(wip_src))
    if core_src.is_dir() and str(core_src) not in sys.path:
        sys.path.insert(0, str(core_src))


# ----------------------------
# Settings
# ----------------------------
@dataclass(frozen=True)
class TrainSettings:
    # Game/layout
    field_size: int = 4            # 4x4 -> n2=16 (power of 2 required)
    comms_size: int = 1            # Pyramid requires 1 comm bit

    # Dataset
    dataset_size: int = 250_000
    seed: int = 1234

    # Training
    batch_size: int = 256
    epochs_meas_a: int = 100
    epochs_meas_b: int = 100
    epochs_comb_a: int = 350
    epochs_comb_b: int = 250
    hidden_units: int = 64
    fit_verbose: int = 0



    # Converter settings
    layer_training_betas: Sequence[float] = (0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0)

    # Which layers to train (per kind): list of level indices d.
    # Use None to mean "all levels".
    train_meas_a: Sequence[int] | None = None
    train_comb_a: Sequence[int] | None = None
    train_meas_b: Sequence[int] | None = None
    train_comb_b: Sequence[int] | None = None

    # Weights IO
    weights_dir: Path = Path("WIP/weights_pyr_layers")
    filename_template: str = "{kind}_d{d}_PATCH_TO_NOT_ACC_OVERWRITE.weights.h5"


# ----------------------------
# Small TF dataset helpers
# ----------------------------
def to_tf_dataset_xy(X: np.ndarray, y: np.ndarray, batch_size: int, *, shuffle: bool = True, seed: int = 0) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.float32)))
    if shuffle:
        ds = ds.shuffle(min(len(X), 10000), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def to_tf_dataset_multiX_y(Xs: tuple[np.ndarray, ...], y: np.ndarray, batch_size: int, *, shuffle: bool = True, seed: int = 0) -> tf.data.Dataset:
    Xs = tuple(x.astype(np.float32) for x in Xs)
    ds = tf.data.Dataset.from_tensor_slices((Xs, y.astype(np.float32)))
    if shuffle:
        ds = ds.shuffle(min(len(y), 10000), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ----------------------------
# Loss: BCE + hardness regularization (encourage modest/large |logit|)
# ----------------------------
def make_bce_with_hardness(*, margin: float, lam: float) -> tf.keras.losses.Loss:
    """
    Keras loss: BCE(from_logits=True) + lam * mean(softplus(margin - |logits|)).

    If you want "modest" logits, set margin accordingly (e.g. margin=3..8) and tune lam.
    """
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def loss_fn(y_true, y_pred_logits):
        task = bce(y_true, y_pred_logits)
        hard = tf.reduce_mean(tf.nn.softplus(tf.cast(margin, tf.float32) - tf.abs(y_pred_logits)))
        return task + tf.cast(lam, tf.float32) * hard

    return loss_fn


class HardnessMetric(tf.keras.metrics.Mean):
    def __init__(self, margin: float, name: str = "hard_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.margin = tf.constant(margin, tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        hard = tf.reduce_mean(tf.nn.softplus(self.margin - tf.abs(y_pred)))
        return super().update_state(hard, sample_weight=sample_weight)


def train_single_input_layer(
    layer: tf.keras.layers.Layer,
    X: np.ndarray,
    Y_bits: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    fit_verbose: int,
    hard_margin: float = 8.0,
    hard_lam: float = 0.02,
) -> tf.keras.layers.Layer:
    inp = tf.keras.Input(shape=(X.shape[1],), dtype=tf.float32)
    out = layer(inp)
    model = tf.keras.Model(inp, out)

    loss_fn = make_bce_with_hardness(margin=hard_margin, lam=hard_lam)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=loss_fn,
        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0), HardnessMetric(margin=hard_margin)],
    )
    tfds = to_tf_dataset_xy(X, Y_bits, batch_size, shuffle=True, seed=seed)
    model.fit(tfds, epochs=epochs, verbose=fit_verbose)
    return layer


def train_two_input_layer(
    layer: tf.keras.layers.Layer,
    X1: np.ndarray,
    X2: np.ndarray,
    Y_bits: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    fit_verbose: int,
    hard_margin: float = 8.0,
    hard_lam: float = 0.02,
) -> tf.keras.layers.Layer:
    i1 = tf.keras.Input(shape=(X1.shape[1],), dtype=tf.float32)
    i2 = tf.keras.Input(shape=(X2.shape[1],), dtype=tf.float32)
    out = layer(i1, i2)
    model = tf.keras.Model([i1, i2], out)

    loss_fn = make_bce_with_hardness(margin=hard_margin, lam=hard_lam)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=loss_fn,
        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0), HardnessMetric(margin=hard_margin)],
    )
    tfds = to_tf_dataset_multiX_y((X1, X2), Y_bits, batch_size, shuffle=True, seed=seed)
    model.fit(tfds, epochs=epochs, verbose=fit_verbose)
    return layer


# ----------------------------
# Weights IO helpers
# ----------------------------
def weights_path(settings: TrainSettings, kind: LayerKind, d: int) -> Path:
    return Path(settings.weights_dir) / settings.filename_template.format(kind=kind, d=d)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _wrap_layer_as_model(layer: tf.keras.layers.Layer, kind: LayerKind, *, n2: int, d: int) -> tf.keras.Model:
    """Create a tiny Model that uses `layer` so we can call model.save_weights/load_weights.

    Keras `Model` has save_weights/load_weights; `Layer` may not (depending on TF/Keras version).
    This wrapper shares the same Layer instance, so loading affects the original layer.
    """
    Ld, kd = _level_sizes(n2, d)

    if kind == "meas_a":
        inp = tf.keras.Input(shape=(Ld,), dtype=tf.float32)
        out = layer(inp)
        return tf.keras.Model(inp, out)

    if kind == "comb_a":
        f = tf.keras.Input(shape=(Ld,), dtype=tf.float32)
        o = tf.keras.Input(shape=(kd,), dtype=tf.float32)
        out = layer(f, o)
        return tf.keras.Model([f, o], out)

    if kind == "meas_b":
        inp = tf.keras.Input(shape=(Ld,), dtype=tf.float32)
        out = layer(inp)
        return tf.keras.Model(inp, out)

    if kind == "comb_b":
        g = tf.keras.Input(shape=(Ld,), dtype=tf.float32)
        o = tf.keras.Input(shape=(kd,), dtype=tf.float32)
        c = tf.keras.Input(shape=(1,), dtype=tf.float32)
        out = layer(g, o, c)
        return tf.keras.Model([g, o, c], out)

    raise ValueError(kind)


def save_layer_weights(layer: tf.keras.layers.Layer, path: Path, *, kind: LayerKind, n2: int, d: int) -> None:
    _ensure_parent(path)
    model = _wrap_layer_as_model(layer, kind, n2=n2, d=d)
    model.save_weights(str(path))


def load_layer_weights(layer: tf.keras.layers.Layer, path: Path, *, kind: LayerKind, n2: int, d: int) -> tf.keras.layers.Layer:
    model = _wrap_layer_as_model(layer, kind, n2=n2, d=d)
    model.load_weights(str(path))
    return layer


# ----------------------------
# Layer factories (build + optional load)
# ----------------------------
def _infer_n2_depth(field_size: int) -> tuple[int, int]:
    n2 = field_size * field_size
    depth = int(np.log2(n2))
    if 2 ** depth != n2:
        raise ValueError(f"field_size^2 must be a power of 2; got n2={n2}.")
    return n2, depth


def _level_sizes(n2: int, d: int) -> tuple[int, int]:
    # (Ld, kd) with kd = Ld//2
    Ld = n2 // (2 ** d)
    kd = Ld // 2
    return Ld, kd


def make_layer(kind: LayerKind, *, hidden_units: int) -> tf.keras.layers.Layer:
    # Imports are inside to keep module importable even if repo paths aren't set yet.
    from Q_Sea_Battle_New.pyr_measurement_layer_a import PyrMeasurementLayerA
    from Q_Sea_Battle_New.pyr_combine_layer_a import PyrCombineLayerA
    from Q_Sea_Battle_New.pyr_measurement_layer_b import PyrMeasurementLayerB
    from Q_Sea_Battle_New.pyr_combine_layer_b import PyrCombineLayerB

    if kind == "meas_a":
        return PyrMeasurementLayerA(hidden_units=hidden_units)
    if kind == "comb_a":
        return PyrCombineLayerA(hidden_units=hidden_units)
    if kind == "meas_b":
        return PyrMeasurementLayerB(hidden_units=hidden_units)
    if kind == "comb_b":
        return PyrCombineLayerB(hidden_units=hidden_units)
    raise ValueError(f"Unknown kind: {kind}")


def build_layer_for_level(layer: tf.keras.layers.Layer, kind: LayerKind, *, n2: int, d: int) -> None:
    """
    Build layer variables by calling once with dummy tensors (required for save/load_weights).
    Shapes follow the converter geometry.
    """
    Ld, kd = _level_sizes(n2, d)
    if kind == "meas_a":
        _ = layer(tf.zeros((1, Ld), tf.float32), training=False)
    elif kind == "comb_a":
        _ = layer(tf.zeros((1, Ld), tf.float32), tf.zeros((1, kd), tf.float32), training=False)
    elif kind == "meas_b":
        _ = layer(tf.zeros((1, Ld), tf.float32), training=False)
    elif kind == "comb_b":
        # gun_d is length Ld (onehot in bits, but logits in runtime), out_b is kd, comm is (1,1)
        _ = layer(tf.zeros((1, Ld), tf.float32), tf.zeros((1, kd), tf.float32), tf.zeros((1, 1), tf.float32), training=False)
    else:
        raise ValueError(kind)


def load_trained_layer(kind: LayerKind, d: int, settings: TrainSettings) -> tf.keras.layers.Layer:
    """
    Create a new layer instance of `kind` for level d and load weights from file.
    """
    n2, _ = _infer_n2_depth(settings.field_size)
    layer = make_layer(kind, hidden_units=settings.hidden_units)
    build_layer_for_level(layer, kind, n2=n2, d=d)

    path = weights_path(settings, kind, d)
    if not path.exists():
        raise FileNotFoundError(f"Missing weights file: {path}")
    return load_layer_weights(layer, path, kind=kind, n2=n2, d=d)


# ----------------------------
# Training routines (selected layers only)
# ----------------------------
def _normalize_layer_list(depth: int, ds: Sequence[int] | None) -> list[int]:
    if ds is None:
        return list(range(depth))
    out = sorted(set(int(x) for x in ds))
    for d in out:
        if d < 0 or d >= depth:
            raise ValueError(f"Invalid level d={d} for depth={depth}.")
    return out


def train_selected_layers(settings: TrainSettings) -> dict[LayerKind, dict[int, Path]]:
    """
    Generate dataset, run converters, train the requested layer kinds/levels, and save weights.

    Returns:
        mapping[kind][d] = weights_path that was written (only for trained layers).
    """
    # deterministic
    tf.random.set_seed(settings.seed)
    np.random.seed(settings.seed)

    n2, depth = _infer_n2_depth(settings.field_size)

    # Imports here (repo must be on sys.path already)
    from Q_Sea_Battle_New.pyr_dataset_generation_utilities import generate_pyr_dataset
    from Q_Sea_Battle_New.pyr_dataset_conversion_utilities import (
        convert_layer_measure_a,
        convert_layer_combine_a,
        convert_layer_measure_b,
        convert_layer_combine_b,
    )

    ds_np = generate_pyr_dataset(n2=n2, num_games=settings.dataset_size, seed=settings.seed, validate=True)

    out_written: dict[LayerKind, dict[int, Path]] = {"meas_a": {}, "comb_a": {}, "meas_b": {}, "comb_b": {}}

    # ---- Measure A ----
    meas_a_levels = _normalize_layer_list(depth, settings.train_meas_a)
    if meas_a_levels:
        conv = convert_layer_measure_a(ds_np, rep_x="hard_logit", rep_y="bits", beta=list(settings.layer_training_betas))
        for d in meas_a_levels:
            X, Y = conv[d]
            layer = make_layer("meas_a", hidden_units=settings.hidden_units)
            train_single_input_layer(
                layer, X, Y,
                epochs=settings.epochs_meas_a, batch_size=settings.batch_size,
                seed=settings.seed + 10 + d, fit_verbose=settings.fit_verbose,
            )
            build_layer_for_level(layer, "meas_a", n2=n2, d=d)
            p = weights_path(settings, "meas_a", d)
            save_layer_weights(layer, p, kind="meas_a", n2=n2, d=d)
            out_written["meas_a"][d] = p

    # ---- Combine A ----
    comb_a_levels = _normalize_layer_list(depth, settings.train_comb_a)
    if comb_a_levels:
        conv = convert_layer_combine_a(ds_np, rep_field="hard_logit", rep_outcome="hard_logit", rep_target="bits", beta=list(settings.layer_training_betas))
        for d in comb_a_levels:
            (field_d, out_a_d), field_d1 = conv[d]
            layer = make_layer("comb_a", hidden_units=settings.hidden_units)
            train_two_input_layer(
                layer, field_d, out_a_d, field_d1,
                epochs=settings.epochs_comb_a, batch_size=settings.batch_size,
                seed=settings.seed + 20 + d, fit_verbose=settings.fit_verbose,
            )
            build_layer_for_level(layer, "comb_a", n2=n2, d=d)
            p = weights_path(settings, "comb_a", d)
            save_layer_weights(layer, p, kind="comb_a", n2=n2, d=d)
            out_written["comb_a"][d] = p

    # ---- Measure B ----
    meas_b_levels = _normalize_layer_list(depth, settings.train_meas_b)
    if meas_b_levels:
        conv = convert_layer_measure_b(ds_np, rep_x="hard_logit", rep_y="bits", beta=list(settings.layer_training_betas))
        for d in meas_b_levels:
            X, Y = conv[d]
            layer = make_layer("meas_b", hidden_units=settings.hidden_units)
            train_single_input_layer(
                layer, X, Y,
                epochs=settings.epochs_meas_b, batch_size=settings.batch_size,
                seed=settings.seed + 30 + d, fit_verbose=settings.fit_verbose,
            )
            build_layer_for_level(layer, "meas_b", n2=n2, d=d)
            p = weights_path(settings, "meas_b", d)
            save_layer_weights(layer, p, kind="meas_b", n2=n2, d=d)
            out_written["meas_b"][d] = p

    # ---- Combine B ----
    comb_b_levels = _normalize_layer_list(depth, settings.train_comb_b)
    if comb_b_levels:
        conv = convert_layer_combine_b(
            ds_np,
            rep_gun="hard_logit",
            rep_outcome_b="hard_logit",
            rep_comm_in="hard_logit",
            rep_gun_next="bits",
            rep_comm_next="bits",
            beta=list(settings.layer_training_betas),
        )

        loss_gun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss_comm = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        for d in comb_b_levels:
            (gun_d, out_b_d, comm_d), (gun_d1_onehot, comm_d1_bits) = conv[d]
            layer = make_layer("comb_b", hidden_units=settings.hidden_units)

            # build vars
            build_layer_for_level(layer, "comb_b", n2=n2, d=d)

            # Keras 3 optimizer needs to "know" variables up-front; build it after layer is built.
            opt = tf.keras.optimizers.Adam(1e-3)
            opt.build(layer.trainable_variables)

            ds_tf = tf.data.Dataset.from_tensor_slices(((gun_d, out_b_d, comm_d), (gun_d1_onehot, comm_d1_bits))).shuffle(
                min(len(gun_d1_onehot), 10000), seed=settings.seed + 40 + d, reshuffle_each_iteration=True
            ).batch(settings.batch_size).prefetch(tf.data.AUTOTUNE)

            def train_step(x_gun, x_outb, x_comm, y_gun_onehot, y_comm_bits):
                y_gun_idx = tf.argmax(y_gun_onehot, axis=-1)
                with tf.GradientTape() as tape:
                    pred_gun_logits, pred_comm_logits = layer(x_gun, x_outb, x_comm, training=True)
                    lg = loss_gun(y_gun_idx, pred_gun_logits)
                    lc = loss_comm(y_comm_bits, pred_comm_logits)
                    # Small magnitude penalty to avoid runaway logits (tune or remove as needed)
                    mag_pen = 1e-4 * tf.reduce_mean(tf.square(pred_gun_logits))
                    mag_pen += 1e-4 * tf.reduce_mean(tf.square(pred_comm_logits))
                    loss = lg + lc + mag_pen
                grads = tape.gradient(loss, layer.trainable_variables)
                pairs = [(g, v) for g, v in zip(grads, layer.trainable_variables) if g is not None]
                opt.apply_gradients(pairs)
                return loss, lg, lc, mag_pen

            for epoch in range(settings.epochs_comb_b):
                losses = []
                for (xg, xo, xc), (yg, yc) in ds_tf:
                    loss, _, _, _ = train_step(xg, xo, xc, yg, yc)
                    losses.append(loss)
                if settings.fit_verbose:
                    tf.print("CombineB d=", d, "epoch", epoch + 1, "/", settings.epochs_comb_b, "loss=", tf.reduce_mean(losses))

            p = weights_path(settings, "comb_b", d)
            save_layer_weights(layer, p, kind="comb_b", n2=n2, d=d)
            out_written["comb_b"][d] = p

    # remove empty entries
    return {k: v for k, v in out_written.items() if v}


# ----------------------------
# CLI
# ----------------------------
def _parse_levels(s: str) -> Sequence[int] | None:
    """
    Parse a level list:
      "all" -> None
      "0,1,2" -> [0,1,2]
      "" -> [] (train none)
    """
    s = s.strip()
    if s == "" or s.lower() == "none":
        return []
    if s.lower() == "all":
        return None
    return [int(x) for x in s.split(",") if x.strip() != ""]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train selected Pyramid layers and save per-layer weights.")
    ap.add_argument("--field-size", type=int, default=4)
    ap.add_argument("--dataset-size", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden-units", type=int, default=64)
    ap.add_argument("--betas", type=str, default="10.0", help="Comma-separated betas for hard_logit mixing, e.g. '0.1,0.3,1,3,10'")
    ap.add_argument("--weights-dir", type=str, default="WIP/weights_pyr_layers")
    ap.add_argument("--filename-template", type=str, default="{kind}_d{d}_PATCH_TO_NOT_ACC_OVERWRITE.weights.h5")

    ap.add_argument("--epochs-meas-a", type=int, default=10)
    ap.add_argument("--epochs-comb-a", type=int, default=100)
    ap.add_argument("--epochs-meas-b", type=int, default=10)
    ap.add_argument("--epochs-comb-b", type=int, default=150)
    ap.add_argument("--fit-verbose", type=int, default=0)

    ap.add_argument("--train-meas-a", type=str, default="all", help="Levels for meas_a: all | none | '0,1,2'")
    ap.add_argument("--train-comb-a", type=str, default="all", help="Levels for comb_a: all | none | '0,1,2'")
    ap.add_argument("--train-meas-b", type=str, default="all", help="Levels for meas_b: all | none | '0,1,2'")
    ap.add_argument("--train-comb-b", type=str, default="all", help="Levels for comb_b: all | none | '0,1,2'")

    ap.add_argument("--repo-root-marker", type=str, default="WIP", help="Folder name that exists at repo root (used to set CWD).")
    args = ap.parse_args(argv)

    # Mirror notebook behavior: make imports work when run as a script.
    change_to_repo_root(args.repo_root_marker)
    add_repo_to_syspath()

    betas = [float(x) for x in args.betas.split(",") if x.strip() != ""]

    settings = TrainSettings(
        field_size=args.field_size,
        dataset_size=args.dataset_size,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs_meas_a=args.epochs_meas_a,
        epochs_comb_a=args.epochs_comb_a,
        epochs_meas_b=args.epochs_meas_b,
        epochs_comb_b=args.epochs_comb_b,
        hidden_units=args.hidden_units,
        fit_verbose=args.fit_verbose,
        layer_training_betas=tuple(betas),
        train_meas_a=_parse_levels(args.train_meas_a),
        train_comb_a=_parse_levels(args.train_comb_a),
        train_meas_b=_parse_levels(args.train_meas_b),
        train_comb_b=_parse_levels(args.train_comb_b),
        weights_dir=Path(args.weights_dir),
        filename_template=args.filename_template,
    )

    written = train_selected_layers(settings)
    if written:
        print("Wrote weights:")
        for kind, mapping in written.items():
            for d, p in mapping.items():
                print(f"  {kind} d={d}: {p}")
    else:
        print("No layers selected for training; nothing written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
