"""Player B baseline interface for QSeaBattle.

This module provides the default (baseline) implementation for the B-side
player. It is split out of :mod:`Q_Sea_Battle.players_base` so the legacy import
path can remain stable while allowing the implementation to evolve.

The baseline policy implemented here is intentionally simple: it returns a
random binary decision and does not use the provided gun position or
communication vector.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .game_layout import GameLayout


class PlayerB:
    """Baseline Player B interface.

    The default behavior is a random shooting strategy. Given a gun position and
    a communication vector from Player A, Player B returns a random decision in
    ``{0, 1}``.

    Attributes:
        game_layout: Shared game configuration provided by the players factory.
    """

    def __init__(self, game_layout: GameLayout) -> None:
        """Initialize Player B.

        Args:
            game_layout: Game configuration for this player instance.
        """
        self.game_layout = game_layout

    def decide(
        self, gun: np.ndarray, comm: np.ndarray, supp: Optional[Any] = None
    ) -> int:
        """Return Player B's shoot / no-shoot decision.

        This baseline implementation ignores all inputs and samples a uniform
        random action.

        Args:
            gun: Gun position encoding (typically a flattened one-hot array).
                Ignored by the baseline implementation.
            comm: Communication vector from Player A. Ignored by the baseline
                implementation.
            supp: Optional supporting information. Unused by the baseline
                implementation.

        Returns:
            0 for "do not shoot" or 1 for "shoot".
        """
        return int(np.random.randint(0, 2))