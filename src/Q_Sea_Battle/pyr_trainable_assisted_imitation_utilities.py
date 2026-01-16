
"""
Q_Sea_Battle.pyr_trainable_assisted_imitation_utilities
===================================================

Per-level dataset generation utilities for the **pyramid** trainable-assisted models.

This module mirrors the linear imitation utilities style, but is **level-aware**:
each pyramid level operates on an input vector of length ``L`` and produces outputs
of length ``L/2`` (with ``L`` a power of two, and ``L >= 2``).

The functions here generate *teacher* targets (supervised labels) for the pyramid
measurement and combine operations as described in the project v2 pyramid spec.

Design goals
------------
- Keep each dataset generator focused on a single training task at a single level.
- Provide a consistent dict-of-arrays sample format suitable for conversion to
  ``tf.data.Dataset`` via :func:`to_tf_dataset`.
- Keep training orchestration (looping over levels) in notebooks/scripts rather
  than hiding it in utilities.

All generated vectors use dtype ``np.float32`` with values in {0.0, 1.0}.

Public API
----------
- :func:`pyramid_levels`
- :func:`generate_measurement_dataset_a`
- :func:`generate_combine_dataset_a`
- :func:`generate_measurement_dataset_b`
- :func:`generate_combine_dataset_b`
- :func:`to_tf_dataset`
- :func:`train_layer`
- :func:`transfer_pyr_model_a_layer_weights`
- :func:`transfer_pyr_model_b_layer_weights`
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

# TensorFlow is optional at import time; functions that require TF will raise if unavailable.
try:
    import tensorflow as tf  # type: ignore
except Exception as _e:  # pragma: no cover
    tf = None  # type: ignore
    _tf_import_error = _e


ArrayLike = Union[np.ndarray, "tf.Tensor"]


def _require_tf() -> "tf":  # type: ignore
    """Return the tensorflow module or raise a helpful error."""
    global tf  # noqa: PLW0603
    if tf is None:  # pragma: no cover
        raise ModuleNotFoundError(
            "TensorFlow is required for this function but is not available in the active environment."
        ) from _tf_import_error
    return tf  # type: ignore


def pyramid_levels(field_size: int) -> List[int]:
    """Return input sizes per pyramid level.

    For ``N`` (power of two), levels are ``[N, N/2, ..., 2]``.

    Parameters
    ----------
    field_size:
        Initial size ``N`` (power of two, >=2).

    Returns
    -------
    list[int]
        Input sizes per level, e.g. ``16 -> [16, 8, 4, 2]``.
    """
    if field_size < 2:
        raise ValueError("field_size must be >= 2.")
    if field_size & (field_size - 1) != 0:
        raise ValueError("field_size must be a power of two.")
    levels: List[int] = []
    L = field_size
    while L >= 2:
        levels.append(L)
        L //= 2
    return levels


def _pairs(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split last dimension into even/odd indices."""
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return even, odd


def teacher_measure_a(field: np.ndarray) -> np.ndarray:
    """Teacher for measurement A: pairwise XOR."""
    even, odd = _pairs(field)
    return np.logical_xor(even > 0.5, odd > 0.5).astype(np.float32)


def teacher_combine_a(field: np.ndarray, sr_outcome: np.ndarray) -> np.ndarray:
    """Teacher for combine A: even-bit XOR SR outcome."""
    even, _ = _pairs(field)
    return np.logical_xor(even > 0.5, sr_outcome > 0.5).astype(np.float32)


def teacher_measure_b(gun: np.ndarray) -> np.ndarray:
    """Teacher for measurement B: (NOT even) AND odd."""
    even, odd = _pairs(gun)
    return (np.logical_not(even > 0.5) & (odd > 0.5)).astype(np.float32)


def teacher_combine_b(gun: np.ndarray, sr_outcome: np.ndarray, comm: np.ndarray):
    """Teacher for combine B.

    Inputs (per sample):
      - gun:        shape (L,), expected one-hot in the *game* setting.
      - sr_outcome: shape (L//2,), shared resource (SR) outcomes for this level.
      - comm:       shape (1,), current comm bit.

    Targets:
      - next_gun:  pairwise downsample of gun via XOR:
            next_gun[i] = gun[2i] ⊕ gun[2i+1]
        If gun is one-hot, next_gun is also one-hot (at index floor(k/2)).
      - next_comm: comm updated by XOR with a single SR bit selected by current comm:
            idx = np.argmax(next_gun)   # since next_gun is one-hot
            next_comm = comm ⊕ sr_outcome[idx]
    """
    even, odd = _pairs(gun)
    next_gun = (np.logical_xor(even > 0.5, odd > 0.5)).astype(np.float32)

    idx = np.argmax(next_gun)
    next_comm = np.array([float((comm[0] > 0.5) ^ (sr_outcome[idx] > 0.5))], dtype=np.float32)
    return next_gun, next_comm

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _check_L(L: int) -> None:
    if L < 2:
        raise ValueError("L must be >= 2.")
    if L & (L - 1) != 0:
        raise ValueError("L must be a power of two.")


def _sample_bits(num_samples: int, L: int, seed: int) -> np.ndarray:
    r = _rng(seed)
    return r.integers(0, 2, size=(num_samples, L), dtype=np.int32).astype(np.float32)


def _sample_sr(num_samples: int, L2: int, seed: int) -> np.ndarray:
    r = _rng(seed)
    return r.integers(0, 2, size=(num_samples, L2), dtype=np.int32).astype(np.float32)


def generate_measurement_dataset_a(L: int, num_samples: int, seed: int = 0) -> Dict[str, np.ndarray]:
    """Dataset for measurement A at a single level."""
    _check_L(L)
    field = _sample_bits(num_samples, L, seed)
    meas_target = teacher_measure_a(field)
    return {"field": field, "meas_target": meas_target}


def generate_combine_dataset_a(L: int, num_samples: int, seed: int = 0) -> Dict[str, np.ndarray]:
    """Dataset for combine A at a single level."""
    _check_L(L)
    field = _sample_bits(num_samples, L, seed)
    sr_outcome = _sample_sr(num_samples, L // 2, seed + 12345)
    next_field_target = teacher_combine_a(field, sr_outcome)
    return {"field": field, "sr_outcome": sr_outcome, "next_field_target": next_field_target}


def generate_measurement_dataset_b(L: int, num_samples: int, seed: int = 0):
    """Level-aware dataset for Pyramid Measurement-B.

    In the *game* setting, `gun` is a one-hot vector over L positions (single 1).
    We generate gun accordingly so that training matches inference.

    Returns dict with:
      - gun:         (N, L) one-hot
      - meas_target: (N, L//2) computed by `teacher_measure_b`
    """
    rng = np.random.default_rng(seed)

    # One-hot gun
    idx = rng.integers(0, L, size=num_samples)
    gun = np.zeros((num_samples, L), dtype=np.float32)
    gun[np.arange(num_samples), idx] = 1.0

    meas_target = np.stack([teacher_measure_b(g) for g in gun], axis=0)
    return dict(gun=gun, meas_target=meas_target)

def generate_combine_dataset_b(L: int, num_samples: int, seed: int = 0):
    """Level-aware dataset for Pyramid Combine-B.

    Conventions (matching the game / players):
      - gun is one-hot over L positions
      - sr_outcome is binary over L//2 positions (shared resource (SR) for this level)
      - comm is a single binary bit

    Targets:
      - next_gun_target: gun downsampled by pairwise XOR:
            next_gun[i] = gun[2i] ⊕ gun[2i+1]
        (preserves one-hot if gun is one-hot)
      - next_comm_target: comm updated by XOR with a single SR bit chosen by comm:
            idx = np.argmax(next_gun)   # since next_gun is one-hot
            next_comm = comm ⊕ sr_outcome[idx]
    """
    rng = np.random.default_rng(seed)

    # One-hot gun (N, L)
    idx = rng.integers(0, L, size=num_samples)
    gun = np.zeros((num_samples, L), dtype=np.float32)
    gun[np.arange(num_samples), idx] = 1.0

    # Shared resource (SR) outcomes for this level (binary, N x L//2)
    sr_outcome = (rng.random((num_samples, L // 2)) < 0.5).astype(np.float32)

    # Comm bit (binary, N x 1)
    comm = (rng.random((num_samples, 1)) < 0.5).astype(np.float32)

    # Targets
    next_gun_target = np.zeros((num_samples, L // 2), dtype=np.float32)
    next_comm_target = np.zeros((num_samples, 1), dtype=np.float32)
    for i in range(num_samples):
        ng, nc = teacher_combine_b(gun[i], sr_outcome[i], comm[i])
        next_gun_target[i] = ng
        next_comm_target[i] = nc

    return dict(
        gun=gun,
        sr_outcome=sr_outcome,
        comm=comm,
        next_gun_target=next_gun_target,
        next_comm_target=next_comm_target,
    )

def to_tf_dataset(
    ds: Mapping[str, np.ndarray],
    x_keys: Sequence[str],
    y_key: str,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
    ) -> "tf.data.Dataset":
    """Convert dict-of-arrays to a tf.data.Dataset yielding (x, y)."""
    tfm = _require_tf()
    n = int(ds[y_key].shape[0])
    xs = [ds[k] for k in x_keys]
    y = ds[y_key]
    x = xs[0] if len(xs) == 1 else tuple(xs)
    tds = tfm.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        tds = tds.shuffle(buffer_size=min(n, 10_000), seed=int(seed), reshuffle_each_iteration=True)
    return tds.batch(int(batch_size)).prefetch(tfm.data.AUTOTUNE)


def train_layer(layer: Any, ds: "tf.data.Dataset", loss: Any, epochs: int, metrics: Optional[Sequence[Any]] = None, verbose: int = 1) -> "tf.keras.Model":
    """Train a Keras layer as a standalone model (wrapper model)."""
    tfm = _require_tf()
    metrics = list(metrics) if metrics is not None else []
    sample_x, _ = next(iter(ds.take(1)))

    if isinstance(sample_x, (tuple, list)):
        inputs = [tfm.keras.Input(shape=x.shape[1:], dtype=x.dtype) for x in sample_x]
        outputs = layer(*inputs)
        model = tfm.keras.Model(inputs, outputs)
    else:
        inp = tfm.keras.Input(shape=sample_x.shape[1:], dtype=sample_x.dtype)
        out = layer(inp)
        model = tfm.keras.Model(inp, out)

    model.compile(optimizer="adam", loss=loss, metrics=metrics)
    model.fit(ds, epochs=int(epochs), verbose=verbose)
    return model


def _assert_len(name: str, lst: Sequence[Any], expected: int) -> None:
    if len(lst) != expected:
        raise ValueError(f"{name} must have length {expected}, got {len(lst)}.")


def transfer_pyr_model_a_layer_weights(model_a: Any, measure_layers_a: Sequence[Any], combine_layers_a: Sequence[Any]) -> None:
    """Copy per-level trained weights into Model A (expects measure_layers/combine_layers)."""

    if not hasattr(model_a, "measure_layers") or not hasattr(model_a, "combine_layers"):
        raise ValueError("model_a must have attributes 'measure_layers' and 'combine_layers' (sequences).")
    model_meas = list(getattr(model_a, "measure_layers"))
    model_comb = list(getattr(model_a, "combine_layers"))
    _assert_len("measure_layers_a", measure_layers_a, len(model_meas))
    _assert_len("combine_layers_a", combine_layers_a, len(model_comb))
    if len(model_meas) != len(model_comb):
        raise ValueError("model_a.measure_layers and model_a.combine_layers must have the same length.")
    for src, dst in zip(measure_layers_a, model_meas):
        dst.set_weights(src.get_weights())
    for src, dst in zip(combine_layers_a, model_comb):
        dst.set_weights(src.get_weights())


def transfer_pyr_model_b_layer_weights(model_b: Any, measure_layers_b: Sequence[Any], combine_layers_b: Sequence[Any]) -> None:
    """Copy per-level trained weights into Model B (expects measure_layers/combine_layers)."""

    if not hasattr(model_b, "measure_layers") or not hasattr(model_b, "combine_layers"):
        raise ValueError("model_b must have attributes 'measure_layers' and 'combine_layers' (sequences).")
    model_meas = list(getattr(model_b, "measure_layers"))
    model_comb = list(getattr(model_b, "combine_layers"))
    _assert_len("measure_layers_b", measure_layers_b, len(model_meas))
    _assert_len("combine_layers_b", combine_layers_b, len(model_comb))
    if len(model_meas) != len(model_comb):
        raise ValueError("model_b.measure_layers and model_b.combine_layers must have the same length.")
    for src, dst in zip(measure_layers_b, model_meas):
        dst.set_weights(src.get_weights())
    for src, dst in zip(combine_layers_b, model_comb):
        dst.set_weights(src.get_weights())
