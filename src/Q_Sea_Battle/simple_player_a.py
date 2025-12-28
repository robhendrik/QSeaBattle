"""Simple deterministic Player A implementation.

Author: Rob Hendriks
Version: 0.1
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .game_layout import GameLayout
from .players_base import PlayerA


class SimplePlayerA(PlayerA):
    """Deterministic Player A using the first m cells of the field.

    This player encodes the first ``m`` bits of the flattened field
    directly into the communication vector.
    """

    def __init__(self, game_layout: GameLayout) -> None:
        """Initialise a :class:`SimplePlayerA` instance.

        Args:
            game_layout: Game configuration for this player.
        """
        super().__init__(game_layout)

    def decide(self, field: np.ndarray, supp: Optional[Any] = None) -> np.ndarray:
        """Return the first ``m`` bits of the flattened field.

        Args:
            field: Flattened field array of 0/1 values. Any shape is
                accepted and will be flattened internally.
            supp: Optional supporting information (unused).

        Returns:
            Communication vector of length ``m`` derived from the field.
        """
        # Flatten the field and take the first m bits.
        flat_field = np.asarray(field, dtype=int).ravel()
        m = self.game_layout.comms_size
        return flat_field[:m].copy()

