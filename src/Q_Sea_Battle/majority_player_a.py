"""Majority-based deterministic Player A implementation.

Author: Rob Hendriks
Package: Q_Sea_Battle
Version: 0.1
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .game_layout import GameLayout
from .players_base import PlayerA


class MajorityPlayerA(PlayerA):
    """Player A encoding majority information over field segments.

    The flattened field of length ``n2`` is split into ``m``
    contiguous segments of equal length ``segment_len = n2 // m``.
    For each segment, the communication bit is set to 1 if the
    number of ones is greater than or equal to the number of zeros
    in that segment, else 0.
    """

    def __init__(self, game_layout: GameLayout) -> None:
        """Initialise a :class:`MajorityPlayerA` instance.

        Args:
            game_layout: Game configuration for this player.
        """
        super().__init__(game_layout)

    def decide(self, field: np.ndarray, supp: Optional[Any] = None) -> np.ndarray:
        """Encode majority statistics in the communication vector.

        Args:
            field: Flattened field array of 0/1 values. Any shape is
                accepted and flattened internally.
            supp: Optional supporting information (unused).

        Returns:
            Communication vector of length ``m`` where each entry
            encodes the majority of a field segment.
        """
        flat_field = np.asarray(field, dtype=int).ravel()
        n2 = self.game_layout.field_size ** 2
        m = self.game_layout.comms_size

        # Basic assumption: comms_size divides n2 (enforced by GameLayout).
        segment_len = n2 // m

        comm = np.zeros(m, dtype=int)
        for i in range(m):
            start = i * segment_len
            end = start + segment_len
            segment = flat_field[start:end]
            ones = int(segment.sum())
            zeros = segment_len - ones
            comm[i] = 1 if ones >= zeros else 0

        return comm
