"""Utilities for generating imitation-learning datasets for NeuralNetPlayers.

The helpers in this module synthesize supervised training data for the
NeuralNetPlayers split-model setup:

- Model A learns a mapping from a flattened binary field to communication bits.
- Model B learns a mapping from (communication bits, gun position) to a shoot
  decision.

The "teacher" policy implemented here is the majority strategy used by the
MajorityPlayer: the field is partitioned into contiguous segments and each
communication bit indicates whether its segment contains a majority of ones.

Notes:
    - All fields are represented as flattened arrays of length
      ``n2 = field_size ** 2`` in row-major order (consistent with how the rest
      of the project flattens fields).
    - Communication vectors have length ``m = comms_size``.
    - This module uses NumPy arrays inside a pandas.DataFrame (object columns)
      to match the expected training interfaces.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from Q_Sea_Battle.game_layout import GameLayout


def make_segments(layout: GameLayout) -> List[Tuple[int, int]]:
    """Partition a flattened field into contiguous segments.

    The flattened field has length ``n2 = field_size ** 2`` and is partitioned
    into ``m = comms_size`` contiguous segments. Segment lengths are as even as
    possible; if ``n2`` is not divisible by ``m``, the first ``n2 % m`` segments
    are one element longer.

    Args:
        layout: GameLayout providing ``field_size`` and ``comms_size``.

    Returns:
        A list of ``(start, end)`` index pairs (Python slice-style, end
        exclusive), of length ``layout.comms_size``, covering ``[0, n2)``
        without gaps or overlaps.

    Raises:
        ValueError: If ``field_size < 1``, ``comms_size < 1``, or
            ``comms_size > n2``.
        RuntimeError: If the constructed segments do not cover ``[0, n2)``.
    """
    n = layout.field_size
    m = layout.comms_size

    if n < 1:
        raise ValueError("field_size must be >= 1.")
    n2 = n * n
    if m < 1 or m > n2:
        raise ValueError(f"comms_size must be in [1, {n2}], got {m}.")

    base = n2 // m
    rem = n2 % m

    segments: List[Tuple[int, int]] = []
    start = 0
    for j in range(m):
        # Distribute the remainder: the first `rem` segments get one extra
        # element so that all segments remain contiguous and cover the full
        # flattened field.
        length = base + (1 if j < rem else 0)
        end = start + length
        segments.append((start, end))
        start = end

    # Safety check: ensure full coverage without gaps.
    if segments[0][0] != 0 or segments[-1][1] != n2:
        raise RuntimeError("Segment construction did not cover full field.")

    return segments


def compute_majority_comm(fields: np.ndarray, layout: GameLayout) -> np.ndarray:
    """Compute teacher communication bits via per-segment majority voting.

    For each sample and each segment (as defined by :func:`make_segments`), this
    function computes whether ones are in the majority within that segment and
    emits a communication bit of 1.0 if so, otherwise 0.0.

    Tie-breaking: when a segment contains exactly half ones and half zeros, the
    output is 1.0 because the comparison is ``count >= L/2``.

    Args:
        fields: Array of shape ``(N, n2)`` containing flattened binary fields.
            Values are expected to be in ``{0, 1}`` (or floats equivalent).
        layout: GameLayout defining ``field_size`` and ``comms_size``.

    Returns:
        Array of shape ``(N, m)`` with values in ``{0.0, 1.0}``, dtype
        ``np.float32``.

    Raises:
        ValueError: If ``fields`` is not 2D or its second dimension does not
            equal ``field_size ** 2``.
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

    for j, (start, end) in enumerate(segments):
        seg = fields[:, start:end]  # shape: (N, L_j)
        counts = seg.sum(axis=1)
        L = end - start
        comms[:, j] = (counts >= (L / 2.0)).astype(np.float32)

    return comms


def _make_rng(seed: Optional[int]) -> np.random.Generator:
    """Create a NumPy random Generator for an optional seed.

    Args:
        seed: Optional integer seed.

    Returns:
        A NumPy ``Generator`` instance.
    """
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(seed)


def generate_majority_dataset_model_a(
    layout: GameLayout,
    num_samples: int,
    p_one: float = 0.5,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate an imitation dataset for Model A (field -> communication).

    Each field cell is sampled IID from a Bernoulli distribution with parameter
    ``p_one``. Targets are computed via per-segment majority voting.

    Args:
        layout: GameLayout defining ``field_size`` and ``comms_size``.
        num_samples: Number of samples to generate.
        p_one: Probability that a given field cell equals 1.
        seed: Optional RNG seed for reproducibility.

    Returns:
        A pandas.DataFrame with columns:

        - ``field``: 1D NumPy array of shape ``(n2,)``, dtype ``np.float32``.
        - ``comm``: 1D NumPy array of shape ``(m,)``, dtype ``np.float32``.

        Columns store NumPy arrays per row (object dtype columns).

    Raises:
        ValueError: If ``num_samples <= 0``.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    n2 = layout.field_size * layout.field_size
    rng = _make_rng(seed)

    # Sample IID binary fields.
    fields = rng.binomial(1, p_one, size=(num_samples, n2)).astype(np.float32)

    # Compute teacher majority comm bits.
    comms = compute_majority_comm(fields, layout).astype(np.float32)

    # Store arrays per row in the DataFrame (object columns).
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
    """Generate an imitation dataset for Model B (comm + gun -> shoot).

    For each sample:

    1. Sample a binary field IID from Bernoulli(p_one).
    2. Compute the teacher communication vector via majority voting.
    3. Sample a gun index uniformly from ``[0, n2)`` and one-hot encode it.
    4. Define the teacher ``shoot`` label as the communication bit of the
       segment containing the gun index.

    This corresponds to a segment-level teacher: the shot decision depends only
    on the segment majority signal, not the exact cell value.

    Args:
        layout: GameLayout defining ``field_size`` and ``comms_size``.
        num_samples: Number of samples to generate.
        p_one: Probability that a given field cell equals 1.
        seed: Optional RNG seed for reproducibility.

    Returns:
        A pandas.DataFrame with columns:

        - ``field``: 1D NumPy array of shape ``(n2,)``, dtype ``np.float32``.
        - ``comm``: 1D NumPy array of shape ``(m,)``, dtype ``np.float32``.
        - ``gun``: 1D one-hot NumPy array of shape ``(n2,)``, dtype
          ``np.float32``.
        - ``shoot``: Scalar ``np.float32`` in ``{0.0, 1.0}``.

    Raises:
        ValueError: If ``num_samples <= 0``.
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

        # 4. Find the segment containing the gun index and use that segment's
        # teacher communication bit as the shoot label.
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
    """Generate paired imitation datasets for Model A and Model B.

    This is a convenience wrapper around
    :func:`generate_majority_dataset_model_a` and
    :func:`generate_majority_dataset_model_b`. When a seed is provided, it uses
    ``seed`` for Model A and ``seed + 1`` for Model B so that both datasets are
    reproducible while remaining statistically independent draws.

    Args:
        layout: GameLayout defining ``field_size`` and ``comms_size``.
        num_samples_a: Number of samples for the Model A dataset.
        num_samples_b: Number of samples for the Model B dataset.
        p_one: Probability that a given field cell equals 1.
        seed: Optional RNG seed.

    Returns:
        A tuple ``(dataset_a, dataset_b)`` where each element is a
        pandas.DataFrame in the format returned by the corresponding generator.
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