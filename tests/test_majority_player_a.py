"""Tests for the MajorityPlayerA class."""

from __future__ import annotations

import sys

# Ensure the src folder is on the Python path so Q_Sea_Battle can be imported.
sys.path.append("./src")

import numpy as np

from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.majority_player_a import MajorityPlayerA


def test_majority_player_a_decide_segments_and_majority() -> None:
    """MajorityPlayerA.decide should compute majority per segment correctly."""
    # Choose n2=16, m=4 so comms_size divides field_size**2.
    layout = GameLayout(field_size=4, comms_size=4)
    player_a = MajorityPlayerA(layout)

    n2 = layout.field_size ** 2
    segment_len = n2 // layout.comms_size

    # Construct a deterministic field with known majorities per segment.
    # Segment 0 (indices 0-3): ones=2, zeros=2 -> comm[0]=1
    # Segment 1 (4-7): ones=3, zeros=1 -> comm[1]=1
    # Segment 2 (8-11): ones=1, zeros=3 -> comm[2]=0
    # Segment 3 (12-15): ones=3, zeros=1 -> comm[3]=1
    field = np.array(
        [0, 0, 1, 1,   1, 1, 1, 0,   0, 0, 0, 1,   1, 0, 1, 1],
        dtype=int,
    )
    assert field.size == n2
    assert segment_len == 4

    comm = player_a.decide(field)

    assert isinstance(comm, np.ndarray)
    assert comm.shape == (layout.comms_size,)

    expected = np.array([1, 1, 0, 1], dtype=int)
    assert np.array_equal(comm, expected)


def test_majority_player_a_input_not_modified() -> None:
    """decide() should not modify the input field array."""
    layout = GameLayout(field_size=4, comms_size=2)
    player_a = MajorityPlayerA(layout)

    n2 = layout.field_size ** 2
    field = np.ones(n2, dtype=int)
    field_copy = field.copy()

    _ = player_a.decide(field)

    assert np.array_equal(field, field_copy)
