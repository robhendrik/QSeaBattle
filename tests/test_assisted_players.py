"""Tests for the AssistedPlayers factory class."""

import sys

sys.path.append("./src")


from Q_Sea_Battle.assisted_players import AssistedPlayers
from Q_Sea_Battle.assisted_player_a import AssistedPlayerA
from Q_Sea_Battle.assisted_player_b import AssistedPlayerB
from Q_Sea_Battle.game_layout import GameLayout


def test_assisted_players_creation_and_players_types() -> None:
    layout = GameLayout(field_size=4, comms_size=1)
    players = AssistedPlayers(game_layout=layout, p_high=0.8)
    player_a, player_b = players.players()

    assert isinstance(player_a, AssistedPlayerA)
    assert isinstance(player_b, AssistedPlayerB)

    # Both players should share the same layout and parent.
    assert player_a.game_layout is layout
    assert player_b.game_layout is layout
    assert player_a.parent is players
    assert player_b.parent is players
