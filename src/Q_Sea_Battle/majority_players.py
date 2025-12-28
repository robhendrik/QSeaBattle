"""Factory for MajorityPlayers pair.

Author: Rob Hendriks
Version: 0.1
"""

from __future__ import annotations

from typing import Tuple

from .game_layout import GameLayout
from .players_base import Players, PlayerA, PlayerB
from .majority_player_a import MajorityPlayerA
from .majority_player_b import MajorityPlayerB


class MajorityPlayers(Players):
    """Players factory producing a MajorityPlayerA/B pair.

    Uses a shared :class:`GameLayout` configuration for both players.
    """

    def __init__(self, game_layout: GameLayout | None = None) -> None:
        """Initialise a :class:`MajorityPlayers` factory.

        Args:
            game_layout: Optional shared configuration. If None, a
                default :class:`GameLayout` is created.
        """
        super().__init__(game_layout)

    def players(self) -> Tuple[PlayerA, PlayerB]:
        """Create a ``(MajorityPlayerA, MajorityPlayerB)`` pair.

        Returns:
            Tuple of concrete player instances that share the
            factory's :class:`GameLayout`.
        """
        player_a = MajorityPlayerA(self.game_layout)
        player_b = MajorityPlayerB(self.game_layout)
        return player_a, player_b
