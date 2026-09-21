"""Majority-based deterministic Player B implementation.

This module provides a :class:`~Q_Sea_Battle.majority_player_b.MajorityPlayerB`
strategy that interprets Player A's communication bits via a fixed partitioning
of the flattened grid indices.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .game_layout import GameLayout
from .players_base import PlayerB


class MajorityPlayerB(PlayerB):
    """Player B that interprets majority-style communication by index segment.

    The flattened gun position (the index of the active entry in a one-hot gun
    vector) is mapped into one of ``m`` contiguous segments spanning the full
    index range. The player returns the corresponding communication bit as the
    decision.

    Notes:
        This implementation assumes the game layout is configured such that
        the communication length ``m`` divides the flattened grid size ``n2``,
        producing equal-length segments.
    """

    def __init__(self, game_layout: GameLayout) -> None:
        """Initialize a :class:`MajorityPlayerB`.

        Args:
            game_layout: Game configuration for this player.
        """
        super().__init__(game_layout)

    def decide(
        self, gun: np.ndarray, comm: np.ndarray, supp: Optional[Any] = None
    ) -> int:
        """Decide whether to shoot based on the segment-selected comm bit.

        Args:
            gun: Flattened one-hot gun vector of length ``n2``.
            comm: Communication vector from Player A of length ``m``.
            supp: Optional supporting information. Not used by this player.

        Returns:
            Decision bit: ``1`` to shoot or ``0`` to not shoot.
        """
        flat_gun = np.asarray(gun, dtype=int).ravel()
        comm = np.asarray(comm, dtype=int).ravel()

        n2 = flat_gun.size
        m = comm.size

        # Segment length is determined by the layout constraint that m divides n2.
        segment_len = n2 // m

        # Index of the gun (assumes valid one-hot input).
        gun_index = int(np.argmax(flat_gun))

        # Determine which segment the gun lies in.
        segment_index = gun_index // segment_len
        if segment_index >= m:
            # Defensive clamp if n2 is not an exact multiple of m.
            segment_index = m - 1

        return int(comm[segment_index])