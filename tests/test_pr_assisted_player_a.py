"""Tests for PRAssistedPlayerA."""

import sys

sys.path.append("./src")

import numpy as np

from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.pr_assisted_players import PRAssistedPlayers
from Q_Sea_Battle.pr_assisted_player_a import PRAssistedPlayerA


def test_pr_assisted_player_a_decide_returns_single_bit_comm() -> None:
    layout = GameLayout(field_size=4, comms_size=1)
    assisted = PRAssistedPlayers(game_layout=layout, p_high=0.8)
    player_a, _ = assisted.players()

    assert isinstance(player_a, PRAssistedPlayerA)

    n2 = layout.field_size ** 2
    field = np.zeros(n2, dtype=int)
    field[0] = 1  # simple deterministic pattern

    comm = player_a.decide(field)

    assert comm.shape == (1,)
    assert comm.dtype == int
    assert int(comm[0]) in (0, 1)
