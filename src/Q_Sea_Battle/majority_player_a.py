"""Majority-based deterministic Player A implementation.

This module defines :class:`MajorityPlayerA`, a deterministic Player A strategy
that encodes a coarse summary of the field into the communication vector by
taking per-segment majorities.

The flattened field (of length ``field_size ** 2``) is partitioned into
``comms_size`` contiguous, equal-length segments. For each segment, a single
communication bit is produced:
- 1 if the number of ones is greater than or equal to the number of zeros
- 0 otherwise

Ties are resolved in favor of 1.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .game_layout import GameLayout
from .players_base import PlayerA


class MajorityPlayerA(PlayerA):
    """Player A that encodes per-segment majority bits into the comms vector.

    The field is flattened and split into ``m = game_layout.comms_size``
    contiguous segments, each of length ``segment_len = n2 // m`` where
    ``n2 = game_layout.field_size ** 2``.

    For segment ``i``, the outgoing communication bit is:

        ``comm[i] = 1`` if ``#ones >= #zeros`` else ``0``.

    Notes:
        This implementation assumes ``m`` divides ``n2``. The code comments
        indicate this is enforced by :class:`~.game_layout.GameLayout`.
    """

    def __init__(self, game_layout: GameLayout) -> None:
        """Initialize the player.

        Args:
            game_layout: Game configuration for this player.
        """
        super().__init__(game_layout)

    def decide(self, field: np.ndarray, supp: Optional[Any] = None) -> np.ndarray:
        """Compute the communication vector from the field.

        Args:
            field: Field values. Any input shape is accepted; it is converted
                with ``np.asarray(..., dtype=int)`` and flattened with
                ``ravel()``. Values are treated as integers when computing the
                per-segment sums.
            supp: Optional supporting information. Not used by this strategy.

        Returns:
            A NumPy array of dtype ``int`` with shape ``(m,)``, where
            ``m = game_layout.comms_size``. Each entry is the majority bit of
            the corresponding contiguous field segment, with ties mapped to 1.
        """
        flat_field = np.asarray(field, dtype=int).ravel()
        n2 = self.game_layout.field_size ** 2
        m = self.game_layout.comms_size

        # Assumes comms_size divides n2 (comment indicates GameLayout enforces).
        segment_len = n2 // m

        comm = np.zeros(m, dtype=int)
        for i in range(m):
            start = i * segment_len
            end = start + segment_len
            segment = flat_field[start:end]
            ones = int(segment.sum())
            zeros = segment_len - ones
            # Tie-break: ones >= zeros maps to 1.
            comm[i] = 1 if ones >= zeros else 0

        return comm