"""Tests for the PRAssistedPlayers factory class."""

import sys

sys.path.append("./src")

from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.pr_assisted_players import PRAssistedPlayers
from Q_Sea_Battle.pr_assisted_player_a import PRAssistedPlayerA
from Q_Sea_Battle.pr_assisted_player_b import PRAssistedPlayerB


def test_pr_assisted_players_creation_and_players_types() -> None:
    layout = GameLayout(field_size=4, comms_size=1)
    players = PRAssistedPlayers(game_layout=layout, p_high=0.8)
    player_a, player_b = players.players()

    assert isinstance(player_a, PRAssistedPlayerA)
    assert isinstance(player_b, PRAssistedPlayerB)

    # Both players should share the same layout and parent.
    assert player_a.game_layout is layout
    assert player_b.game_layout is layout
    assert player_a.parent is players
    assert player_b.parent is players
