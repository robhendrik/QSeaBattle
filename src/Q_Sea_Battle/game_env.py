"""Core game environment implementation for QSeaBattle.

Author: Rob Hendriks
Package: Q_Sea_Battle
Version: 0.1
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .game_layout import GameLayout


class GameEnv:
    """Environment for a single QSeaBattle game.

    The environment is responsible for generating the field and gun arrays,
    providing them to the players, and evaluating the reward for Player B's
    shooting decision.

    Attributes:
        game_layout: Configuration object describing the game.
        field: Current field array of shape (n, n) with values in {0, 1}.
        gun: Current gun array of shape (n, n) with exactly one 1 (one-hot).
    """

    def __init__(self, game_layout: Optional[GameLayout] = None) -> None:
        """Initialise the game environment.

        Args:
            game_layout: Optional game configuration. If None, a default
                GameLayout is constructed.
        """
        self.game_layout: GameLayout = game_layout or GameLayout()
        self.field: Optional[np.ndarray] = None
        self.gun: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset the environment state for a new game.

        This creates a new random field and a new random one-hot gun position.
        """
        n = self.game_layout.field_size
        p = self.game_layout.enemy_probability

        # Field: Bernoulli(p) on each cell.
        self.field = np.random.binomial(1, p, size=(n, n)).astype(int)

        # Gun: one-hot over all n^2 positions.
        gun_flat = np.zeros(n * n, dtype=int)
        index = np.random.randint(0, n * n)
        gun_flat[index] = 1
        self.gun = gun_flat.reshape(n, n)

    def provide(self) -> Tuple[np.ndarray, np.ndarray]:
        """Provide inputs to the players.

        Returns the current field and gun arrays in flattened form.

        Returns:
            A tuple (field, gun) where both arrays are one-dimensional with
            length n^2 and dtype int.

        Raises:
            RuntimeError: If the environment has not been reset yet.
        """
        if self.field is None or self.gun is None:
            raise RuntimeError("GameEnv must be reset before calling provide().")

        # Return copies to prevent external modification of internal state.
        return self.field.ravel().copy(), self.gun.ravel().copy()

    def evaluate(self, shoot: int) -> float:
        """Evaluate the result of a shooting decision.

        The reward is 1.0 if the decision matches the true cell value at
        the gun position and 0.0 otherwise.

        Args:
            shoot: Shooting action from Player B, either 0 or 1.

        Returns:
            Reward value 1.0 if the decision is correct, otherwise 0.0.

        Raises:
            RuntimeError: If the environment has not been reset yet.
        """
        if self.field is None or self.gun is None:
            raise RuntimeError("GameEnv must be reset before calling evaluate().")

        # The cell value is the field entry at the one-hot gun index.
        cell_values = self.field[self.gun == 1]
        if cell_values.size != 1:
            raise RuntimeError("Gun array must contain exactly one '1'.")

        cell_value = int(cell_values[0])
        shoot_int = int(shoot)

        return 1.0 if shoot_int == cell_value else 0.0

    def apply_channel_noise(self, comm: np.ndarray) -> np.ndarray:
        """Apply channel noise to a communication vector.

        Each bit is flipped independently with probability channel_noise.

        Args:
            comm: One-dimensional array of integers in {0, 1} with length m.

        Returns:
            A noisy communication vector with the same shape and dtype as comm.
        """
        comm = np.asarray(comm, dtype=int)
        c = float(self.game_layout.channel_noise)

        if c <= 0.0:
            # No noise: return an unchanged copy.
            return comm.copy()
        if c >= 1.0:
            # Full noise: always flip all bits.
            return 1 - comm

        # Flip each bit with probability c.
        flip_mask = np.random.random(size=comm.shape) < c
        noisy = comm.copy()
        noisy[flip_mask] = 1 - noisy[flip_mask]
        return noisy

