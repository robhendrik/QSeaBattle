"""Deprecated compatibility wrapper for PR-assisted players factory.

`AssistedPlayers` is deprecated. Use :class:`PRAssistedPlayers` from
`Q_Sea_Battle.pr_assisted_players` instead.

Author: Rob Hendriks
"""

from __future__ import annotations

import warnings
from typing import Tuple

from .game_layout import GameLayout
from .players_base import PlayerA, PlayerB, Players
from .pr_assisted_players import PRAssistedPlayers as _PRAssistedPlayers
from .assisted_player_a import AssistedPlayerA
from .assisted_player_b import AssistedPlayerB


class AssistedPlayers(_PRAssistedPlayers):
    """Deprecated alias for :class:`PRAssistedPlayers`."""

    def __init__(self, game_layout: GameLayout, p_high: float) -> None:
        warnings.warn(
            "AssistedPlayers is deprecated; use PRAssistedPlayers from Q_Sea_Battle.pr_assisted_players.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(game_layout=game_layout, p_high=p_high)

    def players(self) -> Tuple[PlayerA, PlayerB]:
        """Return deprecated wrapper players to preserve isinstance checks."""
        return AssistedPlayerA(self.game_layout, self), AssistedPlayerB(self.game_layout, self)
