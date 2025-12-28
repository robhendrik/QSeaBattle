"""Tests for the SimplePlayerB class."""

from __future__ import annotations

import sys

# Ensure the src folder is on the Python path so Q_Sea_Battle can be imported.
sys.path.append("./src")

import numpy as np

from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.simple_player_b import SimplePlayerB


def test_simple_player_b_uses_comm_when_gun_in_first_m() -> None:
    """If gun index < m, SimplePlayerB should return comm[gun_index]."""
    layout = GameLayout(field_size=4, comms_size=2, enemy_probability=0.5)
    player_b = SimplePlayerB(layout)

    n2 = layout.field_size ** 2
    gun = np.zeros(n2, dtype=int)

    # Place gun at index 1 (< m).
    gun_index = 1
    gun[gun_index] = 1

    comm = np.array([0, 1, 0], dtype=int)

    shoot = player_b.decide(gun, comm)

    assert isinstance(shoot, int)
    assert shoot == int(comm[gun_index])


def test_simple_player_b_random_branch_respects_probability_zero() -> None:
    """With enemy_probability = 0.0, shoot should always be 0 in random branch."""
    # Use m=1 so gun can point outside first m easily.
    layout = GameLayout(field_size=4, comms_size=1, enemy_probability=0.0)
    player_b = SimplePlayerB(layout)

    n2 = layout.field_size ** 2
    gun = np.zeros(n2, dtype=int)

    # Place gun at index >= m to trigger random branch.
    gun[2] = 1  # m == 1, so 2 >= 1

    comm = np.array([1], dtype=int)

    shoot = player_b.decide(gun, comm)
    assert shoot == 0


def test_simple_player_b_random_branch_respects_probability_one() -> None:
    """With enemy_probability = 1.0, shoot should always be 1 in random branch."""
    layout = GameLayout(field_size=4, comms_size=1, enemy_probability=1.0)
    player_b = SimplePlayerB(layout)

    n2 = layout.field_size ** 2
    gun = np.zeros(n2, dtype=int)
    gun[3] = 1  # index >= m

    comm = np.array([0], dtype=int)

    shoot = player_b.decide(gun, comm)
    assert shoot == 1
