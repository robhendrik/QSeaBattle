"""Tests for the Game class."""

from __future__ import annotations

import sys

# Ensure the src folder is on the Python path so Q_Sea_Battle can be imported.
sys.path.append("./src")
sys.path.append("../src")
import numpy as np

from Q_Sea_Battle.game import Game
from Q_Sea_Battle.game_env import GameEnv
from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.players_base import Players


def test_game_play_returns_expected_shapes_and_values() -> None:
    """Game.play() should return reward, field, gun, comm, shoot in valid ranges."""
    layout = GameLayout(field_size=4, comms_size=2)
    env = GameEnv(layout)
    players = Players(layout)
    game = Game(env, players)

    reward, field, gun, comm, shoot = game.play()

    n2 = layout.field_size ** 2
    m = layout.comms_size

    # Reward is 0.0 or 1.0.
    assert reward in (0.0, 1.0)

    # Field, gun, comm shapes.
    assert isinstance(field, np.ndarray)
    assert isinstance(gun, np.ndarray)
    assert isinstance(comm, np.ndarray)
    assert field.shape == (n2,)
    assert gun.shape == (n2,)
    assert comm.shape == (m,)

    # Field and gun, comm are 0/1 arrays.
    assert set(np.unique(field)).issubset({0, 1})
    assert set(np.unique(gun)).issubset({0, 1})
    assert set(np.unique(comm)).issubset({0, 1})

    # Gun should be one-hot.
    assert int(gun.sum()) == 1

    # Shoot is a scalar int 0 or 1.
    assert isinstance(shoot, int)
    assert shoot in (0, 1)
