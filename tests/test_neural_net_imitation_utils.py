"""Tests for neural_net_imitation_utils.

These tests validate basic shape and consistency properties of the
synthetic imitation-learning datasets and segmentation logic.
"""

from __future__ import annotations
import sys

# Ensure the src folder is on the Python path so Q_Sea_Battle can be imported.
sys.path.append("./src")
from typing import List, Tuple

import numpy as np
import pandas as pd

from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.neural_net_imitation_utils import (
    make_segments,
    compute_majority_comm,
    generate_majority_dataset_model_a,
    generate_majority_dataset_model_b,
    generate_majority_imitation_datasets,
)


def _make_layout(field_size: int, comms_size: int) -> GameLayout:
    """Helper to construct a minimal valid GameLayout via from_dict."""
    return GameLayout.from_dict(
        {
            "field_size": field_size,
            "comms_size": comms_size,
        }
    )


def test_make_segments_cover_and_non_overlap() -> None:
    """Segments must cover all indices without overlap or gaps."""
    layout = _make_layout(field_size=4, comms_size=2)  # n2 = 16
    segments: List[Tuple[int, int]] = make_segments(layout)

    assert len(segments) == 2

    # Collect all indices and ensure uniqueness and full coverage.
    indices = []
    for start, end in segments:
        assert 0 <= start < end <= layout.field_size ** 2
        indices.extend(range(start, end))

    indices_sorted = sorted(indices)
    assert indices_sorted == list(range(layout.field_size ** 2))


def test_compute_majority_comm_simple_cases() -> None:
    """Majority communication should behave sensibly for trivial fields."""
    layout = _make_layout(field_size=4, comms_size=1)  # global majority

    n2 = layout.field_size ** 2

    all_zeros = np.zeros((1, n2), dtype=np.float32)
    all_ones = np.ones((1, n2), dtype=np.float32)

    comm_zeros = compute_majority_comm(all_zeros, layout)
    comm_ones = compute_majority_comm(all_ones, layout)

    assert comm_zeros.shape == (1, 1)
    assert comm_ones.shape == (1, 1)

    assert comm_zeros[0, 0] == 0.0
    assert comm_ones[0, 0] == 1.0


def test_generate_majority_dataset_model_a_shapes() -> None:
    """Model A dataset must have correct column types and shapes."""
    layout = _make_layout(field_size=4, comms_size=2)
    num_samples = 10

    df_a = generate_majority_dataset_model_a(
        layout=layout,
        num_samples=num_samples,
        p_one=0.5,
        seed=123,
    )

    assert len(df_a) == num_samples
    assert "field" in df_a.columns
    assert "comm" in df_a.columns

    n2 = layout.field_size ** 2
    m = layout.comms_size

    # Check a few rows for correct shapes and binary values.
    for i in range(min(3, num_samples)):
        field = df_a.loc[i, "field"]
        comm = df_a.loc[i, "comm"]

        assert isinstance(field, np.ndarray)
        assert isinstance(comm, np.ndarray)
        assert field.shape == (n2,)
        assert comm.shape == (m,)

        assert set(np.unique(field)).issubset({0.0, 1.0})
        assert set(np.unique(comm)).issubset({0.0, 1.0})


def test_generate_majority_dataset_model_b_shapes_and_consistency() -> None:
    """Model B dataset must have expected columns and shapes."""
    layout = _make_layout(field_size=4, comms_size=2)
    num_samples = 10

    df_b = generate_majority_dataset_model_b(
        layout=layout,
        num_samples=num_samples,
        p_one=0.5,
        seed=456,
    )

    assert len(df_b) == num_samples
    for col in ("field", "comm", "gun", "shoot"):
        assert col in df_b.columns

    n2 = layout.field_size ** 2
    m = layout.comms_size

    for i in range(min(3, num_samples)):
        field = df_b.loc[i, "field"]
        comm = df_b.loc[i, "comm"]
        gun = df_b.loc[i, "gun"]
        shoot = df_b.loc[i, "shoot"]

        assert isinstance(field, np.ndarray)
        assert isinstance(comm, np.ndarray)
        assert isinstance(gun, np.ndarray)

        assert field.shape == (n2,)
        assert comm.shape == (m,)
        assert gun.shape == (n2,)

        # Gun must be one-hot.
        assert np.isclose(gun.sum(), 1.0)
        assert set(np.unique(gun)).issubset({0.0, 1.0})

        # Shoot label must be binary.
        assert shoot in (0.0, 1.0)


def test_generate_majority_imitation_datasets_reproducible_with_seed() -> None:
    """Paired datasets should be reproducible when a seed is provided."""
    layout = _make_layout(field_size=4, comms_size=1)

    ds_a1, ds_b1 = generate_majority_imitation_datasets(
        layout=layout,
        num_samples_a=5,
        num_samples_b=5,
        p_one=0.5,
        seed=999,
    )
    ds_a2, ds_b2 = generate_majority_imitation_datasets(
        layout=layout,
        num_samples_a=5,
        num_samples_b=5,
        p_one=0.5,
        seed=999,
    )

    # DataFrames should be identical when using the same seed.
    pd.testing.assert_frame_equal(ds_a1, ds_a2)
    pd.testing.assert_frame_equal(ds_b1, ds_b2)
