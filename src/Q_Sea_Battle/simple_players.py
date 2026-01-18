"""Factory for SimplePlayers pair.

Author: Rob Hendriks
Package: Q_Sea_Battle
Version: 0.1
"""

from __future__ import annotations

from typing import Tuple

from .game_layout import GameLayout
from .players_base import Players, PlayerA, PlayerB
from .simple_player_a import SimplePlayerA
from .simple_player_b import SimplePlayerB


class SimplePlayers(Players):
    """Players factory producing a SimplePlayerA/B pair.

    This factory reuses the shared :class:`GameLayout` configuration
    and constructs matching :class:`SimplePlayerA` and
    :class:`SimplePlayerB` instances.
    """

    def __init__(self, game_layout: GameLayout | None = None) -> None:
        """Initialise a :class:`SimplePlayers` factory.

        Args:
            game_layout: Optional shared configuration. If None, a
                default :class:`GameLayout` is created.
        """
        super().__init__(game_layout)

    def players(self) -> Tuple[PlayerA, PlayerB]:
        """Create a ``(SimplePlayerA, SimplePlayerB)`` pair.

        Returns:
            Tuple of concrete player instances that share the same
            :class:`GameLayout`.
        """
        player_a = SimplePlayerA(self.game_layout)
        player_b = SimplePlayerB(self.game_layout)
        return player_a, player_b

