"""Player A baseline interface for QSeaBattle.

This module defines the default baseline implementation for the A-side player.
It is separated from :mod:`Q_Sea_Battle.players_base` so the legacy import path
can remain stable while the implementation evolves.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .game_layout import GameLayout


class PlayerA:
    """Baseline Player A implementation.

    The default behavior is a minimal random communication strategy: given a
    field, Player A returns a random binary communication vector of length
    ``m = game_layout.comms_size``.

    This class is intended as a simple reference/baseline; subclasses typically
    override :meth:`decide` to implement learned or rule-based strategies.

    Attributes:
        game_layout: Game configuration provided by the players factory.
    """

    def __init__(self, game_layout: GameLayout) -> None:
        """Initialize a Player A instance.

        Args:
            game_layout: Game configuration for this player.
        """
        self.game_layout = game_layout

    def decide(self, field: np.ndarray, supp: Optional[Any] = None) -> np.ndarray:
        """Return a communication vector based on the current field.

        The base implementation ignores ``field`` and ``supp`` and returns a
        random 0/1 vector.

        Args:
            field: Flattened field array. The base implementation does not
                depend on its content.
            supp: Optional supporting information. Unused by the base class.

        Returns:
            A 1-D NumPy array of length ``m = game_layout.comms_size`` with
            integer entries in ``{0, 1}``.
        """
        m = self.game_layout.comms_size
        # Minimal baseline: random 0/1 communication vector.
        return np.random.randint(0, 2, size=m, dtype=int)