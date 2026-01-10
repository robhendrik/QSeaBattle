"""PR-assisted players using PR-assisted resources.

Author: Rob Hendriks
Version: 0.1
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .game_layout import GameLayout
from .players_base import Players, PlayerA, PlayerB
from .pr_assisted import PRAssisted
from .pr_assisted_player_a import PRAssistedPlayerA
from .pr_assisted_player_b import PRAssistedPlayerB


class PRAssistedPlayers(Players):
    """Factory for assisted players with shared randomness.

    This class creates and owns the hierarchy of
    :class:`PRAssisted` boxes and hands out paired
    :class:`PRAssistedPlayerA` / :class:`PRAssistedPlayerB` instances that
    query these boxes during play.
    """

    def __init__(self, game_layout: GameLayout, p_high: float) -> None:
        """Initialise assisted players for a given layout.

        Args:
            game_layout: Game configuration.
            p_high: Correlation parameter used for all shared resources.

        Raises:
            ValueError: If the layout is incompatible with assisted players.
        """
        super().__init__(game_layout)


        if self.game_layout.comms_size != 1:
            raise ValueError("PRAssistedPlayers requires comms_size == 1")


        n2 = self.game_layout.field_size ** 2
        if n2 <= 0:
            raise ValueError("field_size must be positive")
        # n2 must be a power of two.
        if n2 & (n2 - 1) != 0:
            raise ValueError("field_size ** 2 must be a power of 2 for PRAssistedPlayers")


        self.p_high: float = float(p_high)
        self._shared_randomness_array: list[PRAssisted] = self._create_shared_randomness_array()


        # Cached player instances; created lazily on first call to players().
        self._playerA: PRAssistedPlayerA | None = None
        self._playerB: PRAssistedPlayerB | None = None


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def players(self) -> Tuple[PlayerA, PlayerB]:
        """Create (or return cached) assisted player pair.

        Returns:
            Tuple ``(player_a, player_b)``.
        """
        if self._playerA is None or self._playerB is None:
            self._playerA = PRAssistedPlayerA(self.game_layout, parent=self)
            self._playerB = PRAssistedPlayerB(self.game_layout, parent=self)
        return self._playerA, self._playerB


    def reset(self) -> None:
        """Reset internal state and recreate shared resources.

        This is typically called between games in a tournament.
        """
        # Recreate fresh randomness boxes.
        self._shared_randomness_array = self._create_shared_randomness_array()


    def shared_randomness(self, index: int) -> PRAssisted:
        """Return the shared randomness resource at a given index.

        Args:
            index: Index into the internal shared randomness array.

        Returns:
            The :class:`PRAssisted` instance at that index.

        Raises:
            IndexError: If ``index`` is out of bounds.
        """
        return self._shared_randomness_array[index]


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_shared_randomness_array(self) -> list[PRAssisted]:
        """Create the list of shared randomness resources.

        For a field of size ``n2 = field_size ** 2 = 2**n`` the algorithm
        requires ``n`` resources with lengths::

            2**(n-1), 2**(n-2), ..., 2**1, 2**0

        Returns:
            List of :class:`PRAssisted` instances, one per level.
        """
        n2 = self.game_layout.field_size ** 2
        n = int(np.log2(n2))
        if 2 ** n != n2:
            raise ValueError("field_size ** 2 must be an exact power of 2")


        lengths = [2 ** exp for exp in range(n - 1, -1, -1)]
        return [PRAssisted(length=L, p_high=self.p_high) for L in lengths]
