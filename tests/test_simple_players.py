"""Tests for the SimplePlayers factory class."""

from __future__ import annotations

import sys

# Ensure the src folder is on the Python path so Q_Sea_Battle can be imported.
sys.path.append("./src")

import numpy as np

from Q_Sea_Battle.game_env import GameEnv
from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.simple_players import SimplePlayers
from Q_Sea_Battle.simple_player_a import SimplePlayerA
from Q_Sea_Battle.simple_player_b import SimplePlayerB


def test_simple_players_factory_returns_simple_players() -> None:
    """SimplePlayers.players() should return SimplePlayerA/B instances."""
    layout = GameLayout(field_size=4, comms_size=2)
    factory = SimplePlayers(layout)

    player_a, player_b = factory.players()

    assert isinstance(player_a, SimplePlayerA)
    assert isinstance(player_b, SimplePlayerB)

    # Both players should share the same layout as the factory.
    assert player_a.game_layout is layout
    assert player_b.game_layout is layout


def test_simple_players_integration_with_game_env() -> None:
    """SimplePlayers should work end-to-end with GameEnv for a single step."""
    layout = GameLayout(field_size=4, comms_size=2, enemy_probability=0.5)
    env = GameEnv(layout)
    factory = SimplePlayers(layout)

    env.reset()
    player_a, player_b = factory.players()

    field, gun = env.provide()
    comm = player_a.decide(field)
    shoot = player_b.decide(gun, comm)

    # Basic sanity checks.
    assert isinstance(comm, np.ndarray)
    assert comm.shape == (layout.comms_size,)
    assert set(np.unique(comm)).issubset({0, 1})

    assert isinstance(shoot, int)
    assert shoot in (0, 1)
