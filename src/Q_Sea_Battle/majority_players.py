"""Factory for creating a paired set of majority-rule players.

This module defines :class:`MajorityPlayers`, a concrete :class:`~.players_base.Players`
factory that instantiates :class:`~.majority_player_a.MajorityPlayerA` and
:class:`~.majority_player_b.MajorityPlayerB` with a shared :class:`~.game_layout.GameLayout`
instance.
"""

from __future__ import annotations

from typing import Tuple

from .game_layout import GameLayout
from .players_base import Players, PlayerA, PlayerB
from .majority_player_a import MajorityPlayerA
from .majority_player_b import MajorityPlayerB


class MajorityPlayers(Players):
    """Factory producing a paired :class:`MajorityPlayerA` and :class:`MajorityPlayerB`.

    Both players are constructed with the same :class:`~.game_layout.GameLayout`
    instance, ensuring consistent game configuration across the pair.
    """

    def __init__(self, game_layout: GameLayout | None = None) -> None:
        """Initializes the player factory.

        Args:
            game_layout: Shared game configuration. If ``None``, the base class
                creates a default :class:`~.game_layout.GameLayout`.
        """
        super().__init__(game_layout)

    def players(self) -> Tuple[PlayerA, PlayerB]:
        """Creates a ``(MajorityPlayerA, MajorityPlayerB)`` pair.

        Returns:
            A 2-tuple ``(player_a, player_b)`` sharing this factory's
            :class:`~.game_layout.GameLayout`.
        """
        # Both players must share the exact same GameLayout instance.
        player_a = MajorityPlayerA(self.game_layout)
        player_b = MajorityPlayerB(self.game_layout)
        return player_a, player_b