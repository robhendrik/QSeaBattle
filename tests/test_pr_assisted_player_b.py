"""Tests for PRAssistedPlayerB."""

import sys

sys.path.append("./src")

import numpy as np

from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.pr_assisted_players import PRAssistedPlayers
from Q_Sea_Battle.pr_assisted_player_b import PRAssistedPlayerB


def test_pr_assisted_player_b_decide_scalar() -> None:
    """PRAssistedPlayerB.decide should return a scalar 0/1 without error.

    Under the updated specification, any valid one-hot gun vector
    produces a valid active pair at each level. No special index is required.
    """
    layout = GameLayout(field_size=4, comms_size=1)
    assisted = PRAssistedPlayers(game_layout=layout, p_high=0.8)
    _, player_b = assisted.players()

    assert isinstance(player_b, PRAssistedPlayerB)

    n2 = layout.field_size ** 2

    # Choose any valid one-hot gun vector.
    gun = np.zeros(n2, dtype=int)
    gun[3] = 1  # arbitrary valid position

    comm = np.array([0], dtype=int)

    shoot = player_b.decide(gun, comm)
    assert shoot in (0, 1)
