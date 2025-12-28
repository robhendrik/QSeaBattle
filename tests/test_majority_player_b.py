"""Tests for the MajorityPlayerB class."""

from __future__ import annotations

import sys

# Ensure the src folder is on the Python path so Q_Sea_Battle can be imported.
sys.path.append("./src")

import numpy as np

from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.majority_player_b import MajorityPlayerB


def test_majority_player_b_returns_comm_segment_bit() -> None:
    """MajorityPlayerB should return the bit of the segment where the gun lies."""
    layout = GameLayout(field_size=4, comms_size=4)
    player_b = MajorityPlayerB(layout)

    n2 = layout.field_size ** 2
    m = layout.comms_size
    segment_len = n2 // m

    # Define a communication vector with distinct bits per segment.
    comm = np.array([0, 1, 0, 1], dtype=int)

    # For each segment, pick a gun index inside that segment and check the mapping.
    for segment_index in range(m):
        gun = np.zeros(n2, dtype=int)
        gun_pos = segment_index * segment_len  # first index in segment
        gun[gun_pos] = 1

        shoot = player_b.decide(gun, comm)

        assert isinstance(shoot, int)
        assert shoot == int(comm[segment_index])
