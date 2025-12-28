"""Tests for the SimplePlayerA class."""

from __future__ import annotations

import sys

# Ensure the src folder is on the Python path so Q_Sea_Battle can be imported.
sys.path.append("./src")

import numpy as np

from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.simple_player_a import SimplePlayerA


def test_simple_player_a_decide_uses_first_m_bits() -> None:
    """SimplePlayerA.decide should return the first m bits of the field."""
    layout = GameLayout(field_size=4, comms_size=2)
    player_a = SimplePlayerA(layout)

    # Create a deterministic field: [0, 1, 2, ...] mod 2
    n2 = layout.field_size ** 2
    field = np.arange(n2) % 2  # 0,1,0,1,...

    comm = player_a.decide(field)

    assert isinstance(comm, np.ndarray)
    assert comm.shape == (layout.comms_size,)

    # Communication must equal the first m bits of the flattened field.
    expected = field[: layout.comms_size]
    assert np.array_equal(comm, expected)


def test_simple_player_a_does_not_modify_input() -> None:
    """decide() should not modify the input field array."""
    layout = GameLayout(field_size=4, comms_size=2)
    player_a = SimplePlayerA(layout)

    n2 = layout.field_size ** 2
    field = np.ones(n2, dtype=int)
    field_copy = field.copy()

    _ = player_a.decide(field)

    assert np.array_equal(field, field_copy)
