"""Utilities for imitation training of NeuralNetPlayers.

This module generates synthetic imitation-learning datasets for
NeuralNetPlayers.model_a and NeuralNetPlayers.model_b based on the
majority player strategy.

Author: Rob Hendriks
Package: Q_Sea_Battle
Version: 0.1
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from Q_Sea_Battle.game_layout import GameLayout


def make_segments(layout: GameLayout) -> List[Tuple[int, int]]:
    """Compute contiguous segments over the flattened field.

    The field of length n2 = field_size**2 is partitioned into
    m = comms_size contiguous segments that are as even as possible.

    For the standard MajorityPlayers configuration, comms_size divides n2
    exactly, but this function also supports the general case.

    Args:
        layout:
            GameLayout instance providing field_size and comms_size.

    Returns:
        List of (start, end) index pairs (Python slice-style, end exclusive),
        of length m = layout.comms_size, covering range [0, n2) without
        gaps or overlaps.

    Raises:
        ValueError: if field_size < 1 or comms_size < 1 or comms_size > n2.
    """
    n = layout.field_size
    m = layout.comms_size

    if n < 1:
        raise ValueError("field_size must be >= 1.")
    n2 = n * n
    if m < 1 or m > n2:
        raise ValueError(
            f"comms_size must be in [1, {n2}], got {m}."
        )

    base = n2 // m
    rem = n2 % m

    segments: List[Tuple[int, int]] = []
    start = 0
    for j in range(m):
        # Distribute any remainder: first 'rem' segments get one extra.
        length = base + (1 if j < rem else 0)
        end = start + length
        segments.append((start, end))
        start = end

    # Safety check: ensure full coverage without gaps.
    if segments[0][0] != 0 or segments[-1][1] != n2:
        raise RuntimeError("Segment construction did not cover full field.")

    return segments


def compute_majority_comm(fields: np.ndarray, layout: GameLayout) -> np.ndarray:
    """Compute teacher majority communication bits for a batch of fields.

    For each field and each segment (as defined by `make_segments`),
    this function computes whether there is a majority of ones in that
    segment and sets the corresponding communication bit to 1 if so,
    otherwise 0.

    Args:
        fields:
            NumPy array of shape (N, n2) with flattened binary fields,
            values in {0, 1}.
        layout:
            GameLayout defining field_size and comms_size.

    Returns:
        NumPy array of shape (N, m) with majority comm bits in {0.0, 1.0},
        dtype float32.

    Raises:
        ValueError: if input shape is inconsistent with layout.
    """
    if fields.ndim != 2:
        raise ValueError("fields must be a 2D array of shape (N, n2).")

    n2 = layout.field_size * layout.field_size
    if fields.shape[1] != n2:
        raise ValueError(
            f"fields second dimension must be {n2}, got {fields.shape[1]}."
        )

    segments = make_segments(layout)
    m = len(segments)
    num_samples = fields.shape[0]

    comms = np.zeros((num_samples, m), dtype=np.float32)

    # Compute majority per segment.
    for j, (start, end) in enumerate(segments):
        seg = fields[:, start:end]  # (N, L_j)
        # Count ones per sample in this segment.
        counts = seg.sum(axis=1)
        L = end - start
        # Majority: 1 if count >= L/2, else 0.
        comms[:, j] = (counts >= (L / 2.0)).astype(np.float32)

    return comms


def _make_rng(seed: Optional[int]) -> np.random.Generator:
    """Create a NumPy random Generator from an optional seed."""
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(seed)


def generate_majority_dataset_model_a(
    layout: GameLayout,
    num_samples: int,
    p_one: float = 0.5,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate an imitation-learning dataset for Model A (field -> comm).

    Fields are drawn IID from Bernoulli(p_one) per cell. Communication
    targets are majority bits over segments defined by `make_segments`.

    Args:
        layout:
            GameLayout defining field_size and comms_size.
        num_samples:
            Number of samples to generate.
        p_one:
            Probability that any given field cell equals 1.
        seed:
            Optional RNG seed for reproducibility.

    Returns:
        pandas.DataFrame with at least two columns:

        - 'field': 1D NumPy arrays of shape (n2,), dtype float32.
        - 'comm' : 1D NumPy arrays of shape (m,), dtype float32.

        Additional columns may be added in the future.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    n2 = layout.field_size * layout.field_size
    rng = _make_rng(seed)

    # Sample IID binary fields.
    fields = rng.binomial(1, p_one, size=(num_samples, n2)).astype(np.float32)

    # Compute teacher majority comm bits.
    comms = compute_majority_comm(fields, layout).astype(np.float32)

    # Store arrays per row in the DataFrame (dtype=object column).
    df = pd.DataFrame(
        {
            "field": list(fields),
            "comm": list(comms),
        }
    )

    return df


def generate_majority_dataset_model_b(
    layout: GameLayout,
    num_samples: int,
    p_one: float = 0.5,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate an imitation-learning dataset for Model B (comm + gun -> shoot).

    For each sample, this function:

    1. Samples a binary field IID from Bernoulli(p_one).
    2. Computes majority comm bits using the same segmentation as
       `compute_majority_comm`.
    3. Samples a gun index uniformly over all cells and converts it to a
       one-hot vector.
    4. Assigns the shoot label as the majority bit of the segment that
       contains the gun index (segment-majority imitation).

    Args:
        layout:
            GameLayout defining field_size and comms_size.
        num_samples:
            Number of samples to generate.
        p_one:
            Probability that any given field cell equals 1.
        seed:
            Optional RNG seed for reproducibility.

    Returns:
        pandas.DataFrame with at least the columns:

        - 'field': 1D NumPy arrays of shape (n2,), dtype float32.
        - 'comm' : 1D NumPy arrays of shape (m,), dtype float32.
        - 'gun'  : 1D NumPy arrays of shape (n2,), one-hot, dtype float32.
        - 'shoot': scalar float32 in {0.0, 1.0}.

        This schema matches what NeuralNetPlayers.train_model_b expects.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    n2 = layout.field_size * layout.field_size
    rng = _make_rng(seed)
    segments = make_segments(layout)
    m = len(segments)

    fields_list: List[np.ndarray] = []
    comms_list: List[np.ndarray] = []
    guns_list: List[np.ndarray] = []
    shoots_list: List[float] = []

    for _ in range(num_samples):
        # 1. Sample field.
        field = rng.binomial(1, p_one, size=n2).astype(np.float32)

        # 2. Compute majority comm bits for this single field.
        comm = compute_majority_comm(field[np.newaxis, :], layout)[0]

        # 3. Sample gun index and one-hot encode.
        gun_index = int(rng.integers(0, n2))
        gun = np.zeros(n2, dtype=np.float32)
        gun[gun_index] = 1.0

        # 4. Determine segment of gun_index and teacher shoot label.
        segment_idx = 0
        for j, (start, end) in enumerate(segments):
            if start <= gun_index < end:
                segment_idx = j
                break

        shoot = float(comm[segment_idx])

        fields_list.append(field)
        comms_list.append(comm.astype(np.float32))
        guns_list.append(gun)
        shoots_list.append(shoot)

    df = pd.DataFrame(
        {
            "field": fields_list,
            "comm": comms_list,
            "gun": guns_list,
            "shoot": np.array(shoots_list, dtype=np.float32),
        }
    )

    return df


def generate_majority_imitation_datasets(
    layout: GameLayout,
    num_samples_a: int,
    num_samples_b: int,
    p_one: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate paired imitation-learning datasets for Model A and Model B.

    This is a convenience wrapper that calls both
    `generate_majority_dataset_model_a` and
    `generate_majority_dataset_model_b` with derived RNG seeds to ensure
    reproducible but distinct draws for the two datasets.

    Args:
        layout:
            GameLayout defining field_size and comms_size.
        num_samples_a:
            Number of samples for the Model A dataset.
        num_samples_b:
            Number of samples for the Model B dataset.
        p_one:
            Probability that any given field cell equals 1.
        seed:
            Optional RNG seed. If provided, the A- and B-datasets are
            generated with seeds `seed` and `seed + 1`, respectively.

    Returns:
        Tuple (dataset_a, dataset_b) where each element is a pandas.DataFrame
        as returned by the corresponding generator function.
    """
    seed_a: Optional[int]
    seed_b: Optional[int]
    if seed is None:
        seed_a = None
        seed_b = None
    else:
        seed_a = seed
        seed_b = seed + 1

    dataset_a = generate_majority_dataset_model_a(
        layout=layout,
        num_samples=num_samples_a,
        p_one=p_one,
        seed=seed_a,
    )

    dataset_b = generate_majority_dataset_model_b(
        layout=layout,
        num_samples=num_samples_b,
        p_one=p_one,
        seed=seed_b,
    )

    return dataset_a, dataset_b
