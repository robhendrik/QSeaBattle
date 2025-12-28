"""Tests for the MajorityPlayers factory class."""

from __future__ import annotations

import sys

# Ensure the src folder is on the Python path so Q_Sea_Battle can be imported.
sys.path.append("./src")

import numpy as np

from Q_Sea_Battle.game_env import GameEnv
from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.game import Game
from Q_Sea_Battle.majority_players import MajorityPlayers
from Q_Sea_Battle.majority_player_a import MajorityPlayerA
from Q_Sea_Battle.majority_player_b import MajorityPlayerB


def test_majority_players_factory_returns_majority_players() -> None:
    """MajorityPlayers.players() should return MajorityPlayerA/B instances."""
    layout = GameLayout(field_size=4, comms_size=4)
    factory = MajorityPlayers(layout)

    player_a, player_b = factory.players()

    assert isinstance(player_a, MajorityPlayerA)
    assert isinstance(player_b, MajorityPlayerB)

    # Both players should share the same layout as the factory.
    assert player_a.game_layout is layout
    assert player_b.game_layout is layout


def test_majority_players_integration_single_game() -> None:
    """MajorityPlayers should work in an end-to-end Game with GameEnv."""
    layout = GameLayout(field_size=4, comms_size=4, enemy_probability=0.5)
    env = GameEnv(layout)
    factory = MajorityPlayers(layout)
    game = Game(env, factory)

    reward, field, gun, comm, shoot = game.play()

    n2 = layout.field_size ** 2

    assert reward in (0.0, 1.0)
    assert field.shape == (n2,)
    assert gun.shape == (n2,)
    assert comm.shape == (layout.comms_size,)
    assert isinstance(shoot, int)
    assert shoot in (0, 1)

    # Basic sanity on values.
    assert set(np.unique(field)).issubset({0, 1})
    assert set(np.unique(gun)).issubset({0, 1})
    assert set(np.unique(comm)).issubset({0, 1})
