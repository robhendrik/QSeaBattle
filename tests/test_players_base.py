"""Tests for the base Players, PlayerA, and PlayerB classes."""

from __future__ import annotations

import sys

# Ensure the src folder is on the Python path so Q_Sea_Battle can be imported.
sys.path.append("./src")

from typing import Tuple

import numpy as np

from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.players_base import PlayerA, PlayerB, Players


def test_players_factory_returns_base_players() -> None:
    """Players.players() should return PlayerA and PlayerB instances."""
    layout = GameLayout(field_size=4, comms_size=2)
    players = Players(layout)

    player_a, player_b = players.players()

    assert isinstance(player_a, PlayerA)
    assert isinstance(player_b, PlayerB)
    # Both players should share the same layout object.
    assert player_a.game_layout is layout
    assert player_b.game_layout is layout


def test_player_a_decide_returns_correct_shape_and_values() -> None:
    """PlayerA.decide should return a binary vector of length m."""
    layout = GameLayout(field_size=4, comms_size=2)
    player_a = PlayerA(layout)

    n2 = layout.field_size ** 2
    dummy_field = np.zeros(n2, dtype=int)

    comm = player_a.decide(dummy_field, supp=None)

    assert isinstance(comm, np.ndarray)
    assert comm.shape == (layout.comms_size,)
    unique_vals = np.unique(comm)
    assert set(unique_vals).issubset({0, 1})


def test_player_b_decide_returns_scalar_binary() -> None:
    """PlayerB.decide should return an int in {0, 1}."""
    layout = GameLayout(field_size=4, comms_size=2)
    player_b = PlayerB(layout)

    n2 = layout.field_size ** 2
    dummy_gun = np.zeros(n2, dtype=int)
    dummy_gun[0] = 1  # valid one-hot gun

    dummy_comm = np.array([0, 1], dtype=int)

    shoot = player_b.decide(dummy_gun, dummy_comm, supp=None)

    assert isinstance(shoot, int)
    assert shoot in (0, 1)


def test_players_reset_is_noop() -> None:
    """Calling Players.reset() should not raise and leave layout intact."""
    layout = GameLayout(field_size=4, comms_size=1)
    players = Players(layout)

    # Should not raise an exception.
    players.reset()

    # Layout should still be the same object and unchanged.
    assert players.game_layout is layout
    assert players.game_layout.field_size == 4
    assert players.game_layout.comms_size == 1
