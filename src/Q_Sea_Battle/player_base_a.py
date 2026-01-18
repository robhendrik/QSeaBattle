"""Base Player A interface for QSeaBattle.

This module contains the default baseline implementation for the A-side player.
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


class PlayerA:
    """Base class for Player A in QSeaBattle.

    The default behaviour is a random communication strategy: given a
    field, Player A returns a random binary communication vector of
    length m = comms_size.

    Attributes:
        game_layout: Shared configuration from the Players factory.
    """

    def __init__(self, game_layout: GameLayout) -> None:
        """Initialise Player A.

        Args:
            game_layout: Game configuration for this player.
        """
        self.game_layout = game_layout

    def decide(self, field: np.ndarray, supp: Optional[Any] = None) -> np.ndarray:
        """Decide on a communication vector given the field.

        The base implementation ignores the inputs and returns a random
        binary vector of length m = comms_size.

        Args:
            field: Flattened field array containing 0/1 values. The base
                implementation does not depend on its content.
            supp: Optional supporting information (unused in base class).

        Returns:
            A one-dimensional communication array of length m with
            entries in {0, 1}.
        """
        m = self.game_layout.comms_size
        # Random 0/1 vector as minimal baseline behaviour.
        return np.random.randint(0, 2, size=m, dtype=int)
