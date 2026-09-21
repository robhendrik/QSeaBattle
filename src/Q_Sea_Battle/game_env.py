"""Core game environment implementation for QSeaBattle.

This module defines :class:`GameEnv`, a lightweight environment that:
- Samples a binary enemy field (Bernoulli per cell).
- Samples a one-hot "gun" position indicating the queried cell.
- Exposes flattened arrays to players.
- Scores Player B's binary guess about the queried cell.
- Optionally applies independent bit-flip noise to a communication vector.

The exact sizes and probabilities are configured by :class:`~.game_layout.GameLayout`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .game_layout import GameLayout


class GameEnv:
    """Environment for a single QSeaBattle game instance.

    The environment maintains two arrays:

    - ``field``: A binary grid indicating the presence/absence of an enemy at
      each location.
    - ``gun``: A one-hot grid indicating which single location is being queried.

    Players are typically given flattened copies of these arrays via
    :meth:`provide`. Player B produces a binary decision (``shoot``), which is
    scored against the true ``field`` value at the one-hot ``gun`` location.

    Attributes:
        game_layout: Configuration object describing the game.
        field: Field array of shape ``(n, n)`` with integer values in ``{0, 1}``,
            or ``None`` before :meth:`reset`.
        gun: One-hot gun array of shape ``(n, n)`` with integer values in
            ``{0, 1}`` and exactly one 1, or ``None`` before :meth:`reset`.
    """

    def __init__(self, game_layout: Optional[GameLayout] = None) -> None:
        """Initialise the game environment.

        Args:
            game_layout: Optional game configuration. If ``None``, a default
                :class:`~.game_layout.GameLayout` is constructed.
        """
        self.game_layout: GameLayout = game_layout or GameLayout()
        self.field: Optional[np.ndarray] = None
        self.gun: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset the environment state for a new game.

        This samples:
        - A new random ``field`` with independent Bernoulli trials per cell.
        - A new random one-hot ``gun`` position uniformly over all ``n^2`` cells.
        """
        n = self.game_layout.field_size
        p = self.game_layout.enemy_probability

        # Sample a Bernoulli(p) field independently per cell.
        self.field = np.random.binomial(1, p, size=(n, n)).astype(int)

        # Sample a single queried cell uniformly, represented as a one-hot grid.
        gun_flat = np.zeros(n * n, dtype=int)
        index = np.random.randint(0, n * n)
        gun_flat[index] = 1
        self.gun = gun_flat.reshape(n, n)

    def provide(self) -> Tuple[np.ndarray, np.ndarray]:
        """Provide inputs to the players.

        The returned arrays are flattened copies to prevent external mutation of
        the internal environment state.

        Returns:
            Tuple ``(field, gun)``, where both are 1D integer arrays of length
            ``n^2``.

        Raises:
            RuntimeError: If the environment has not been reset yet.
        """
        if self.field is None or self.gun is None:
            raise RuntimeError("GameEnv must be reset before calling provide().")

        return self.field.ravel().copy(), self.gun.ravel().copy()

    def evaluate(self, shoot: int) -> float:
        """Evaluate the result of a shooting decision.

        The reward is ``1.0`` iff the provided decision matches the true field
        value at the one-hot ``gun`` location, and ``0.0`` otherwise.

        Args:
            shoot: Player B's shooting action, intended to be a binary value
                (0 or 1). The value is cast to ``int`` before comparison.

        Returns:
            Reward value (``1.0`` if correct, otherwise ``0.0``).

        Raises:
            RuntimeError: If the environment has not been reset yet.
            RuntimeError: If ``gun`` does not contain exactly one 1.
        """
        if self.field is None or self.gun is None:
            raise RuntimeError("GameEnv must be reset before calling evaluate().")

        # Extract the single queried cell value using the one-hot mask.
        cell_values = self.field[self.gun == 1]
        if cell_values.size != 1:
            raise RuntimeError("Gun array must contain exactly one '1'.")

        cell_value = int(cell_values[0])
        shoot_int = int(shoot)

        return 1.0 if shoot_int == cell_value else 0.0

    def apply_channel_noise(self, comm: np.ndarray) -> np.ndarray:
        """Apply independent bit-flip noise to a communication vector.

        Each element of ``comm`` is flipped (0↔1) independently with probability
        ``game_layout.channel_noise``.

        Args:
            comm: Communication vector. It is converted to an integer NumPy array
                (typically containing values in ``{0, 1}``).

        Returns:
            Noisy communication vector with the same shape as ``comm`` (and
            integer dtype).
        """
        comm = np.asarray(comm, dtype=int)
        c = float(self.game_layout.channel_noise)

        if c <= 0.0:
            # No noise: return an unchanged copy.
            return comm.copy()
        if c >= 1.0:
            # Full noise: flip all bits deterministically.
            return 1 - comm

        # Flip each bit with probability c.
        flip_mask = np.random.random(size=comm.shape) < c
        noisy = comm.copy()
        noisy[flip_mask] = 1 - noisy[flip_mask]
        return noisy