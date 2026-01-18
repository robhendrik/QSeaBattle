"""Base Player B interface for QSeaBattle.

This module contains the default baseline implementation for the B-side player.
It is split out of :mod:`Q_Sea_Battle.players_base` so that the legacy import
path can remain stable while allowing the implementation to evolve.

Author: Rob Hendriks
Package: Q_Sea_Battle
Version: 0.2
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .game_layout import GameLayout


class PlayerB:
    """Base class for Player B in QSeaBattle.

    The default behaviour is a random shooting strategy: given a gun
    position and communication vector, Player B returns a random binary
    decision.

    Attributes:
        game_layout: Shared configuration from the Players factory.
    """

    def __init__(self, game_layout: GameLayout) -> None:
        """Initialise Player B.

        Args:
            game_layout: Game configuration for this player.
        """
        self.game_layout = game_layout

    def decide(
        self, gun: np.ndarray, comm: np.ndarray, supp: Optional[Any] = None
    ) -> int:
        """Decide whether to shoot based on gun position and message.

        The base implementation ignores the inputs and returns a random
        decision in {0, 1}.

        Args:
            gun: Flattened one-hot gun array. The base implementation
                does not depend on its content.
            comm: Communication vector from Player A. Ignored here.
            supp: Optional supporting information (unused in base class).

        Returns:
            An integer 0 (do not shoot) or 1 (shoot).
        """
        return int(np.random.randint(0, 2))
