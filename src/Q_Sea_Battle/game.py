"""Single-game orchestration logic for QSeaBattle.

Author: Rob Hendriks
Package: Q_Sea_Battle
Version: 0.1
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .game_env import GameEnv
from .players_base import Players


class Game:
    """Orchestrates a single QSeaBattle game between two players.

    This class ties together a GameEnv instance and a Players factory
    and runs a single round of the game.
    """

    def __init__(self, game_env: GameEnv, players: Players) -> None:
        """Initialise the game.

        Args:
            game_env: Game environment instance.
            players: Players factory providing Player A and B.
        """
        self.game_env = game_env
        self.players = players

    def play(self) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, int]:
        """Play a single game round.

        The sequence follows the specification:

        1. Reset environment and players.
        2. Obtain Player A and Player B instances.
        3. Get (field, gun) from the environment.
        4. Let Player A decide on comm based on field.
        5. Apply channel noise to comm via GameEnv.
        6. Let Player B decide on shoot based on (gun, noisy comm).
        7. Evaluate reward via GameEnv.

        Returns:
            A tuple (reward, field, gun, comm, shoot) capturing the
            outcome of the game. field, gun, and comm are flattened
            arrays.
        """
        # Reset the environment and players for a fresh game.
        self.game_env.reset()
        self.players.reset()

        # Get concrete player instances.
        player_a, player_b = self.players.players()

        # Provide field and gun (flattened) to the players.
        field, gun = self.game_env.provide()

        # Player A communicates based on the observed field.
        comm = player_a.decide(field, supp=None)

        # Apply channel noise to the communication.
        comm_noisy = self.game_env.apply_channel_noise(comm)

        # Player B decides whether to shoot.
        shoot = player_b.decide(gun, comm_noisy, supp=None)

        # Compute reward for the shooting decision.
        reward = self.game_env.evaluate(shoot)

        return reward, field, gun, comm_noisy, int(shoot)

