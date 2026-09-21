"""Factory for creating a paired set of simple players.

This module provides :class:`SimplePlayers`, a concrete :class:`~.players_base.Players`
factory that instantiates :class:`~.simple_player_a.SimplePlayerA` and
:class:`~.simple_player_b.SimplePlayerB` using a shared
:class:`~.game_layout.GameLayout` configuration.
"""

from __future__ import annotations

from typing import Tuple

from .game_layout import GameLayout
from .players_base import Players, PlayerA, PlayerB
from .simple_player_a import SimplePlayerA
from .simple_player_b import SimplePlayerB


class SimplePlayers(Players):
    """Factory that produces a :class:`SimplePlayerA` / :class:`SimplePlayerB` pair.

    The returned player instances share the same :class:`GameLayout`, ensuring both
    sides agree on board dimensions and other layout parameters.
    """

    def __init__(self, game_layout: GameLayout | None = None) -> None:
        """Initialize the factory.

        Args:
            game_layout: Optional shared game configuration. If ``None``, the base
                :class:`~.players_base.Players` class creates a default
                :class:`GameLayout`.
        """
        super().__init__(game_layout)

    def players(self) -> Tuple[PlayerA, PlayerB]:
        """Instantiate and return the concrete player pair.

        Returns:
            A tuple ``(player_a, player_b)`` containing :class:`SimplePlayerA` and
            :class:`SimplePlayerB` instances that share ``self.game_layout``.
        """
        # Both players must reference the same GameLayout instance so they operate
        # on an identical configuration.
        player_a = SimplePlayerA(self.game_layout)
        player_b = SimplePlayerB(self.game_layout)
        return player_a, player_b