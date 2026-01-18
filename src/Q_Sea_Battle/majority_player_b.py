"""Majority-based deterministic Player B implementation.

Author: Rob Hendriks
Package: Q_Sea_Battle
Version: 0.1
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .game_layout import GameLayout
from .players_base import PlayerB


class MajorityPlayerB(PlayerB):
    """Player B interpreting majority-based communication.

    The flattened gun index determines which segment of the field
    it lies in; the corresponding communication bit is returned as
    the shoot decision.
    """

    def __init__(self, game_layout: GameLayout) -> None:
        """Initialise a :class:`MajorityPlayerB` instance.

        Args:
            game_layout: Game configuration for this player.
        """
        super().__init__(game_layout)

    def decide(
        self, gun: np.ndarray, comm: np.ndarray, supp: Optional[Any] = None
    ) -> int:
        """Decide whether to shoot based on majority information.

        Args:
            gun: Flattened one-hot gun vector of length ``n2``.
            comm: Communication vector from Player A of length ``m``.
            supp: Optional supporting information (unused).

        Returns:
            ``1`` to shoot or ``0`` to not shoot.
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
            segment_index = m - 1

        return int(comm[segment_index])
