"""Base player interfaces for QSeaBattle.

Author: Rob Hendriks
Package: Q_Sea_Battle
Version: 0.1
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from .game_layout import GameLayout


class Players:
    """Factory and container for a pair of QSeaBattle players.

    This class provides a simple interface to construct PlayerA and
    PlayerB instances that share the same GameLayout configuration.

    Attributes:
        game_layout: Shared configuration used by both players.
    """

    def __init__(self, game_layout: Optional[GameLayout] = None) -> None:
        """Initialise a pair of players.

        Args:
            game_layout: Optional shared configuration. If None, a
                default GameLayout is created.
        """
        self.game_layout: GameLayout = game_layout or GameLayout()

    def players(self) -> Tuple["PlayerA", "PlayerB"]:
        """Create the concrete Player A and Player B instances.

        The base implementation returns instances of PlayerA and PlayerB
        using the shared GameLayout. Child classes may override this
        method to return specialised player implementations.

        Returns:
            A tuple (player_a, player_b).
        """
        player_a = PlayerA(self.game_layout)
        player_b = PlayerB(self.game_layout)
        return player_a, player_b

    def reset(self) -> None:
        """Reset any internal state across both players.

        The base implementation has no internal state to reset, but the
        method is provided for compatibility with more complex child
        classes.
        """
        # No state to reset in the base implementation.
        return None


class PlayerA:
    """Base class for Player A in QSeaBattle.

    The default behaviour is a random communication strategy: given a
    field, Player A returns a random binary communication vector of
    length m = comms_size.

    Attributes:
        game_layout: Shared configuration from the Players factory.
    """

    def __init__(self, game_layout: GameLayout) -> None:
        """Initialise Player A.

        Args:
            game_layout: Game configuration for this player.
        """
        self.game_layout = game_layout

    def decide(self, field: np.ndarray, supp: Optional[Any] = None) -> np.ndarray:
        """Decide on a communication vector given the field.

        The base implementation ignores the inputs and returns a random
        binary vector of length m = comms_size.

        Args:
            field: Flattened field array containing 0/1 values. The base
                implementation does not depend on its content.
            supp: Optional supporting information (unused in base class).

        Returns:
            A one-dimensional communication array of length m with
            entries in {0, 1}.
        """
        m = self.game_layout.comms_size
        # Random 0/1 vector as minimal baseline behaviour.
        return np.random.randint(0, 2, size=m, dtype=int)


class PlayerB:
    """Base class for Player B in QSeaBattle.

    The default behaviour is a random shooting strategy: given a gun
    position and communication vector, Player B returns a random binary
    decision.

    Attributes:
        game_layout: Shared configuration from the Players factory.
    """

    def __init__(self, game_layout: GameLayout) -> None:
        """Initialise Player B.

        Args:
            game_layout: Game configuration for this player.
        """
        self.game_layout = game_layout

    def decide(
        self, gun: np.ndarray, comm: np.ndarray, supp: Optional[Any] = None
    ) -> int:
        """Decide whether to shoot based on gun position and message.

        The base implementation ignores the inputs and returns a random
        decision in {0, 1}.

        Args:
            gun: Flattened one-hot gun array. The base implementation
                does not depend on its content.
            comm: Communication vector from Player A. Ignored here.
            supp: Optional supporting information (unused in base class).

        Returns:
            An integer 0 (do not shoot) or 1 (shoot).
        """
        return int(np.random.randint(0, 2))
