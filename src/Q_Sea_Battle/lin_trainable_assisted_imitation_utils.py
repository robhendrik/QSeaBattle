"""QSeaBattle linear assisted imitation utilities.

This module generates *synthetic supervised* datasets used to imitation-train the
linear trainable assisted layers:

- LinMeasurementLayerA: learn meas_bits == field
- LinMeasurementLayerB: learn meas_bits == gun (one-hot)
- LinCombineLayerA: learn comm == parity(outcomes_a) (replicated to m bits)
- LinCombineLayerB: learn shoot == parity(outcomes_b) XOR parity(comm)

The targets implement the "parity prototype" described in the design document.
All generators are reproducible via an explicit RNG seed.

Datasets are returned as a dict of NumPy arrays with leading dimension
`num_samples`. Arrays are float32 (values in {0.0, 1.0}) to be TF-friendly.

The helper `to_tf_dataset` converts such dicts into a `tf.data.Dataset` that
yields (x, y) batches, where:
- if `x_keys` has length 1, x is a single tensor (B, ...),
- if `x_keys` has length >1, x is a tuple of tensors in the same order as
  `x_keys`.

Note: weight-transfer helpers are intentionally minimal here; they can be
expanded later when the full assisted models are wired.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np


import tensorflow as tf



ArrayDict = Dict[str, np.ndarray]


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _n2_from_layout(layout: Any) -> int:
    # Supports either a GameLayout-like object with field_size, or a raw int.
    n = int(getattr(layout, "field_size", layout))
    return n * n


def _m_from_layout(layout: Any) -> int:
    return int(getattr(layout, "comms_size", 1))


def _as_float01(x: np.ndarray) -> np.ndarray:
    # Ensure float32 in {0.0, 1.0}
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    return x


def _parity_bits(x01: np.ndarray) -> np.ndarray:
    """Compute parity (XOR reduction) over the last dimension.

    Args:
        x01: array with last dim N, values in {0,1} (int/bool/float ok).

    Returns:
        parity: array with shape x01.shape[:-1], values in {0,1} int64.
    """
    x_int = (x01 > 0.5).astype(np.int64)
    return np.bitwise_xor.reduce(x_int, axis=-1)

def _layer_is_built(layer: tf.keras.layers.Layer) -> bool:
    # Keras layers have .built after build() / first call.
    # Some custom layers may not set .built reliably, so also check weights.
    return bool(getattr(layer, "built", False)) and (layer.weights is not None)

def generate_measurement_dataset_a(
    layout: Any,
    num_samples: int,
    p_one: float = 0.5,
    seed: Optional[int] = None,
) -> ArrayDict:
    """Generate (field -> meas_target) for LinMeasurementLayerA.

    Target: meas_target == field.
    """
    n2 = _n2_from_layout(layout)
    r = _rng(seed)
    field = r.binomial(1, p_one, size=(num_samples, n2)).astype(np.float32)
    meas_target = field.copy()
    return {"field": field, "meas_target": meas_target}


def generate_measurement_dataset_b(
    layout: Any,
    num_samples: int,
    seed: Optional[int] = None,
) -> ArrayDict:
    """Generate (gun -> meas_target) for LinMeasurementLayerB.

    Each sample has exactly one gun cell set to 1 (one-hot).
    Target: meas_target == gun.
    """
    n2 = _n2_from_layout(layout)
    r = _rng(seed)
    idx = r.integers(0, n2, size=(num_samples,))
    gun = np.zeros((num_samples, n2), dtype=np.float32)
    gun[np.arange(num_samples), idx] = 1.0
    meas_target = gun.copy()
    return {"gun": gun, "meas_target": meas_target}


def generate_combine_dataset_a(
    layout: Any,
    num_samples: int,
    seed: Optional[int] = None,
) -> ArrayDict:
    """Generate (outcomes_a -> comm_target) for LinCombineLayerA.

    outcomes_a: random {0,1} vector (B, n2)
    comm_target: parity(outcomes_a) replicated to (B, m)
    """
    n2 = _n2_from_layout(layout)
    m = _m_from_layout(layout)
    r = _rng(seed)
    outcomes_a = r.integers(0, 2, size=(num_samples, n2), dtype=np.int64).astype(np.float32)
    p = _parity_bits(outcomes_a).astype(np.float32)  # (B,)
    comm_target = np.repeat(p[:, None], m, axis=1).astype(np.float32)
    return {"outcomes_a": outcomes_a, "comm_target": comm_target}


def generate_combine_dataset_b(
    layout: Any,
    num_samples: int,
    seed: Optional[int] = None,
) -> ArrayDict:
    """Generate (outcomes_b, comm -> shoot_target) for LinCombineLayerB.

    outcomes_b: random {0,1} vector (B, n2)
    comm: random {0,1} vector (B, m)
    shoot_target: parity(outcomes_b) XOR parity(comm) as (B, 1)
    """
    n2 = _n2_from_layout(layout)
    m = _m_from_layout(layout)
    r = _rng(seed)
    outcomes_b = r.integers(0, 2, size=(num_samples, n2), dtype=np.int64).astype(np.float32)
    comm = r.integers(0, 2, size=(num_samples, m), dtype=np.int64).astype(np.float32)

    par_b = _parity_bits(outcomes_b)
    par_c = _parity_bits(comm)
    shoot = np.bitwise_xor(par_b, par_c).astype(np.float32)  # (B,)
    shoot_target = shoot[:, None]  # (B,1)
    return {"outcomes_b": outcomes_b, "comm": comm, "shoot_target": shoot_target}


def to_tf_dataset(
    dataset: Union[ArrayDict, Sequence[Mapping[str, Any]]],
    x_keys: Sequence[str],
    y_key: str,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> "tf.data.Dataset":
    """Convert generated dataset into a batched `tf.data.Dataset`.

    Args:
        dataset:
            Either a dict of NumPy arrays with leading dim `N`, or a sequence of
            row-mappings (e.g., list[dict]) where each value is array-like.
        x_keys:
            Keys to use as model inputs. If length == 1, dataset yields `x`
            as a single tensor. If > 1, yields `x` as a tuple of tensors in
            this same order.
        y_key:
            Key to use as training target.
        batch_size:
            Batch size for `.batch(...)`.
        shuffle:
            Whether to shuffle before batching.
        seed:
            Shuffle seed (reproducible ordering when provided).

    Returns:
        A `tf.data.Dataset` yielding `(x, y)` batches.
    """
    if tf is None:  # pragma: no cover
        raise ImportError("TensorFlow is required for to_tf_dataset().")

    if isinstance(dataset, Mapping):
        data_map: MutableMapping[str, np.ndarray] = {k: np.asarray(v) for k, v in dataset.items()}
        n = int(next(iter(data_map.values())).shape[0])
        for k, v in data_map.items():
            if int(v.shape[0]) != n:
                raise ValueError(f"All dataset arrays must have same leading dimension; {k} has {v.shape[0]} vs {n}.")
        x_arrays = [np.asarray(data_map[k]) for k in x_keys]
        y_array = np.asarray(data_map[y_key])
    else:
        # list-of-rows (dict-like)
        rows = list(dataset)
        if not rows:
            raise ValueError("Empty dataset.")
        x_arrays = [np.stack([np.asarray(r[k]) for r in rows], axis=0) for k in x_keys]
        y_array = np.stack([np.asarray(r[y_key]) for r in rows], axis=0)

    # Cast to float32 tensors for training convenience.
    x_tensors = [tf.convert_to_tensor(_as_float01(x)) for x in x_arrays]
    y_tensor = tf.convert_to_tensor(_as_float01(y_array))

    if len(x_tensors) == 1:
        x_out: Any = x_tensors[0]
    else:
        x_out = tuple(x_tensors)

    ds = tf.data.Dataset.from_tensor_slices((x_out, y_tensor))
    if shuffle:
        # buffer_size=N for full shuffle, seeded for reproducibility
        ds = ds.shuffle(buffer_size=int(y_array.shape[0]), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    return ds


def transfer_layer_weights(source_layer: Any, target_layer: Any) -> None:
    """Copy weights from `source_layer` into `target_layer`.

    Both layers must be built before calling (i.e., have variables created).
    """
    src = list(source_layer.get_weights())
    tgt = list(target_layer.get_weights())
    if len(src) != len(tgt):
        raise ValueError(f"Weight length mismatch: source={len(src)} target={len(tgt)}")
    for i, (sw, tw) in enumerate(zip(src, tgt)):
        if sw.shape != tw.shape:
            raise ValueError(f"Weight[{i}] shape mismatch: source={sw.shape} target={tw.shape}")
    target_layer.set_weights(src)





def transfer_layer_weights(
    source_layer: tf.keras.layers.Layer,
    target_layer: tf.keras.layers.Layer,
) -> None:
    """
    Copy trained weights from one layer instance into another.

    Rules:
      - Both layers must be built (their weight variables exist).
      - Weight list lengths must match.
      - Each corresponding weight array must have the same shape.
      - If mismatch occurs, raise a clear error indicating which weight failed.
    """
    if not _layer_is_built(source_layer) or len(source_layer.weights) == 0:
        raise ValueError(
            "transfer_layer_weights: source_layer must be built and have weights. "
            f"Got built={getattr(source_layer, 'built', None)} "
            f"num_weights={len(getattr(source_layer, 'weights', []) or [])}."
        )
    if not _layer_is_built(target_layer) or len(target_layer.weights) == 0:
        raise ValueError(
            "transfer_layer_weights: target_layer must be built and have weights. "
            f"Got built={getattr(target_layer, 'built', None)} "
            f"num_weights={len(getattr(target_layer, 'weights', []) or [])}."
        )

    src_w = source_layer.get_weights()
    tgt_w = target_layer.get_weights()

    if len(src_w) != len(tgt_w):
        raise ValueError(
            "transfer_layer_weights: weight list length mismatch: "
            f"source has {len(src_w)} weights, target has {len(tgt_w)} weights."
        )

    for i, (s, t) in enumerate(zip(src_w, tgt_w)):
        if np.shape(s) != np.shape(t):
            src_name = source_layer.weights[i].name if i < len(source_layer.weights) else f"src[{i}]"
            tgt_name = target_layer.weights[i].name if i < len(target_layer.weights) else f"tgt[{i}]"
            raise ValueError(
                "transfer_layer_weights: shape mismatch at index "
                f"{i} ({src_name} -> {tgt_name}): "
                f"source shape {np.shape(s)} != target shape {np.shape(t)}."
            )

    target_layer.set_weights(src_w)


def transfer_assisted_model_a_layer_weights(
    trained_measure_layer: tf.keras.layers.Layer,
    trained_combine_layer: tf.keras.layers.Layer,
    model_a: Any,
) -> None:
    """
    Apply trained layer weights into a full LinTrainableAssistedModelA.

    Expected model structure:
      - model_a.measure_layer exists and is compatible with trained_measure_layer
      - model_a.combine_layer exists and is compatible with trained_combine_layer

    Effect:
      - Copies weights into model_a.measure_layer
      - Copies weights into model_a.combine_layer
      - Does not modify model_a.sr_layer
    """
    if not hasattr(model_a, "measure_layer"):
        raise AttributeError("transfer_assisted_model_a_layer_weights: model_a has no attribute 'measure_layer'.")
    if not hasattr(model_a, "combine_layer"):
        raise AttributeError("transfer_assisted_model_a_layer_weights: model_a has no attribute 'combine_layer'.")

    transfer_layer_weights(trained_measure_layer, model_a.measure_layer)
    transfer_layer_weights(trained_combine_layer, model_a.combine_layer)


def transfer_assisted_model_b_layer_weights(
    trained_measure_layer: tf.keras.layers.Layer,
    trained_combine_layer: tf.keras.layers.Layer,
    model_b: Any,
) -> None:
    """
    Symmetric helper for LinTrainableAssistedModelB (if present in your design).

    Expected model structure:
      - model_b.measure_layer exists and is compatible with trained_measure_layer
      - model_b.combine_layer exists and is compatible with trained_combine_layer

    Effect:
      - Copies weights into model_b.measure_layer
      - Copies weights into model_b.combine_layer
      - Does not modify model_b.sr_layer
    """
    if not hasattr(model_b, "measure_layer"):
        raise AttributeError("transfer_assisted_model_b_layer_weights: model_b has no attribute 'measure_layer'.")
    if not hasattr(model_b, "combine_layer"):
        raise AttributeError("transfer_assisted_model_b_layer_weights: model_b has no attribute 'combine_layer'.")

    transfer_layer_weights(trained_measure_layer, model_b.measure_layer)
    transfer_layer_weights(trained_combine_layer, model_b.combine_layer)
