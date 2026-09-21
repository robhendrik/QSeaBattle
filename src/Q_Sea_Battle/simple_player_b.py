"""Deterministic Player B implementation for baseline experiments.

This module defines :class:`SimplePlayerB`, a deterministic policy that reacts to
Player A's communication bits when the gun points within the communication
addressing range. For other target cells, the player shoots stochastically using
the environment's configured enemy hit probability.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .game_layout import GameLayout
from .players_base import PlayerB


class SimplePlayerB(PlayerB):
    """Deterministic Player B reacting to Player A messages.

    Let ``m`` be the number of communication bits (``game_layout.comms_size``).
    The gun is provided as a one-hot vector over the enemy field cells; the index
    of its active entry selects either a communication bit or a fallback policy:

    * If the gun index ``i`` satisfies ``i < m``, the action is exactly
      ``comm[i]``.
    * Otherwise, the player shoots stochastically with probability
      ``game_layout.enemy_probability``.

    Notes:
        This policy assumes the gun input is a valid one-hot vector. If it is
        not, the selected index is the argmax of the flattened array.
    """

    def __init__(self, game_layout: GameLayout) -> None:
        """Initialize the player.

        Args:
            game_layout: Game configuration for this player.
        """
        super().__init__(game_layout)

    def decide(
        self, gun: np.ndarray, comm: np.ndarray, supp: Optional[Any] = None
    ) -> int:
        """Decide whether to shoot based on the gun and Player A's message.

        Args:
            gun: One-hot gun vector over enemy cells. Expected to be 1D after
                flattening (via ``ravel()``).
            comm: Communication vector from Player A. Expected to be 1D after
                flattening (via ``ravel()``) and to have length
                ``game_layout.comms_size``.
            supp: Optional supporting information (unused).

        Returns:
            An integer action: ``1`` to shoot, ``0`` to not shoot.
        """
        flat_gun = np.asarray(gun, dtype=int).ravel()
        comm = np.asarray(comm, dtype=int).ravel()

        m = self.game_layout.comms_size
        p = self.game_layout.enemy_probability

        # Target cell index implied by the one-hot gun vector.
        gun_index = int(np.argmax(flat_gun))

        if gun_index < m:
            # Use the communication bit addressed by the gun index.
            return int(comm[gun_index])

        # Outside the communication addressing range: fallback stochastic action.
        return int(np.random.rand() < p)