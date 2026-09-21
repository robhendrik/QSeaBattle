"""Deterministic baseline implementation of Player A.

This module provides a simple Player A policy that encodes information about the
player's field directly into the communication vector. The policy is fully
deterministic and uses no shared resource (SR), replay buffer, or stochasticity.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .game_layout import GameLayout
from .players_base import PlayerA


class SimplePlayerA(PlayerA):
    """Deterministic Player A that transmits the first ``m`` field bits.

    The provided field is flattened in row-major order (NumPy default) and the
    first ``m = game_layout.comms_size`` values are returned as the communication
    vector. Values are coerced to integers via ``np.asarray(..., dtype=int)``.
    """

    def __init__(self, game_layout: GameLayout) -> None:
        """Initialize a :class:`SimplePlayerA` instance.

        Args:
            game_layout: Game configuration for this player.
        """
        super().__init__(game_layout)

    def decide(self, field: np.ndarray, supp: Optional[Any] = None) -> np.ndarray:
        """Compute the communication vector for Player B.

        Args:
            field: Field array containing 0/1 values. Any shape is accepted and
                will be flattened internally.
            supp: Optional supporting information (unused).

        Returns:
            A 1-D NumPy array of length ``m`` containing the first ``m`` values
            of the flattened field.
        """
        # Flatten the field (row-major) and take the first m entries as the
        # communication vector.
        flat_field = np.asarray(field, dtype=int).ravel()
        m = self.game_layout.comms_size
        return flat_field[:m].copy()