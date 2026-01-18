"""Base player interfaces for QSeaBattle.

This module is the stable public facade for the core player types:
- :class:`~Q_Sea_Battle.players_base.Players`
- :class:`~Q_Sea_Battle.players_base.PlayerA` (deprecated import path)
- :class:`~Q_Sea_Battle.players_base.PlayerB` (deprecated import path)

Implementation note:
The concrete baseline implementations for PlayerA and PlayerB were moved to
:mod:`Q_Sea_Battle.players_base_a` and :mod:`Q_Sea_Battle.players_base_b`.
The names remain available here for backward compatibility.

Author: Rob Hendriks
Package: Q_Sea_Battle
Version: 0.2
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple
import warnings

from .game_layout import GameLayout
from .player_base_a import PlayerA as _PlayerA
from .player_base_b import PlayerB as _PlayerB


_DEPRECATION_MSG = (
    "Importing PlayerA/PlayerB from 'Q_Sea_Battle.players_base' is deprecated. "
    "Import from 'Q_Sea_Battle.players_base_a' (PlayerA) and "
    "'Q_Sea_Battle.players_base_b' (PlayerB) instead. "
    "The old import path will be removed in a future major release."
)


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
        player_a = _PlayerA(self.game_layout)
        player_b = _PlayerB(self.game_layout)
        return player_a, player_b

    def reset(self) -> None:
        """Reset any internal state across both players.

        The base implementation has no internal state to reset, but the
        method is provided for compatibility with more complex child
        classes.
        """
        # No state to reset in the base implementation.
        return None


def __getattr__(name: str) -> Any:
    """Resolve deprecated attribute access for PlayerA/PlayerB.

    This hook supports legacy imports like:
        from Q_Sea_Battle.players_base import PlayerA, PlayerB

    A DeprecationWarning is emitted once per interpreter process for each symbol.

    Args:
        name: Attribute name being accessed.

    Returns:
        The requested symbol.

    Raises:
        AttributeError: If the attribute is not provided by this module.
    """
    if name == "PlayerA":
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        globals()[name] = _PlayerA
        return _PlayerA
    if name == "PlayerB":
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        globals()[name] = _PlayerB
        return _PlayerB
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    # Make type checkers aware of the deprecated names without importing via __getattr__.
    from .players_base_a import PlayerA
    from .players_base_b import PlayerB


__all__ = [
    "Players",
    "PlayerA",
    "PlayerB",
]
