"""Single-game orchestration logic for QSeaBattle.

This module provides the top-level control flow for running one complete game
between two players mediated by a :class:`~.game_env.GameEnv`.

A single game consists of:
1) resetting environment and players,
2) sampling the environment state exposed to each player (field and gun),
3) Player A producing a communication message (comm) from the field,
4) applying channel noise to comm via the environment,
5) Player B producing an action (shoot) from the gun and the (noisy) comm,
6) evaluating the resulting reward via the environment.

Notes:
    The player/environment interface uses flattened NumPy arrays as observations
    and messages, consistent with :meth:`~.game_env.GameEnv.provide`.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .game_env import GameEnv
from .players_base import Players


class Game:
    """Orchestrate a single QSeaBattle game between two players.

    This class ties together:
    - a :class:`~.game_env.GameEnv` that defines the observations, channel noise,
      and reward function, and
    - a :class:`~.players_base.Players` factory that provides Player A and Player
      B instances.

    The game is strictly sequential: Player A communicates, noise is applied,
    then Player B acts, and the environment evaluates the action.
    """

    def __init__(self, game_env: GameEnv, players: Players) -> None:
        """Create a game orchestrator.

        Args:
            game_env: Environment instance providing observations, channel noise,
                and reward evaluation.
            players: Factory providing the two player instances and supporting
                reset between games.
        """
        self.game_env = game_env
        self.players = players

    def play(self) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, int]:
        """Run a single game and return its outcome.

        Control flow:
            1. Reset environment and players.
            2. Obtain Player A and Player B instances.
            3. Get (field, gun) observations from the environment.
            4. Player A decides on a communication message from field.
            5. Apply channel noise to the communication message.
            6. Player B decides whether to shoot given (gun, noisy comm).
            7. Evaluate the reward for the shooting decision.

        Returns:
            Tuple containing:
                - reward (float): Environment-evaluated reward for the episode.
                - field (np.ndarray): Flattened field observation for Player A.
                - gun (np.ndarray): Flattened gun observation for Player B.
                - comm_noisy (np.ndarray): Communication after channel noise.
                - shoot (int): Player B action cast to Python ``int``.
        """
        # Reset the environment and players to start a fresh episode.
        self.game_env.reset()
        self.players.reset()

        # Instantiate concrete players for this episode.
        player_a, player_b = self.players.players()

        # Obtain the per-player observations (typically already flattened).
        field, gun = self.game_env.provide()

        # Player A emits a communication message based on the field observation.
        comm = player_a.decide(field, supp=None)

        # Environment applies the communication channel model (e.g., noise).
        comm_noisy = self.game_env.apply_channel_noise(comm)

        # Player B chooses an action based on gun and received communication.
        shoot = player_b.decide(gun, comm_noisy, supp=None)

        # Environment computes the reward given Player B's action.
        reward = self.game_env.evaluate(shoot)

        return reward, field, gun, comm_noisy, int(shoot)