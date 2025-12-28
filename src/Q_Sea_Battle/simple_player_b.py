"""Simple deterministic Player B implementation.

Author: Rob Hendriks
Version: 0.1
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .game_layout import GameLayout
from .players_base import PlayerB


class SimplePlayerB(PlayerB):
    """Deterministic Player B reacting to SimplePlayerA messages.

    If the gun points at one of the first ``m`` cells of the field,
    this player uses the corresponding communication bit. Otherwise
    it shoots with probability equal to ``enemy_probability``.
    """

    def __init__(self, game_layout: GameLayout) -> None:
        """Initialise a :class:`SimplePlayerB` instance.

        Args:
            game_layout: Game configuration for this player.
        """
        super().__init__(game_layout)

    def decide(
        self, gun: np.ndarray, comm: np.ndarray, supp: Optional[Any] = None
    ) -> int:
        """Decide whether to shoot based on the message and gun.

        Behaviour:

        * Let ``i`` be the index of the 1 in the flattened gun vector.
        * If ``i < m``, return ``comm[i]``.
        * Otherwise, return 1 with probability ``enemy_probability``
          and 0 with the remaining probability.

        Args:
            gun: Flattened one-hot gun vector of length ``n2``.
            comm: Communication vector from Player A, length ``m``.
            supp: Optional supporting information (unused).

        Returns:
            1 to shoot or 0 to not shoot.
        """
        flat_gun = np.asarray(gun, dtype=int).ravel()
        comm = np.asarray(comm, dtype=int).ravel()

        m = self.game_layout.comms_size
        p = self.game_layout.enemy_probability

        # Index of the gun (assumes a valid one-hot input).
        gun_index = int(np.argmax(flat_gun))

        if gun_index < m:
            # Use the bit indicated by the gun position.
            return int(comm[gun_index])

        # Otherwise use a Bernoulli(enemy_probability) decision.
        shoot = int(np.random.rand() < p)
        return shoot

