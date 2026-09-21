"""Base player interfaces for QSeaBattle.

This module provides the stable public façade for constructing the two
participants used by the game engine:

- :class:`Players`: Factory/container that supplies paired Player A and Player B
  instances sharing the same :class:`~Q_Sea_Battle.game_layout.GameLayout`.
- ``PlayerA`` / ``PlayerB``: Deprecated import paths kept for backward
  compatibility.

Implementation note:
The concrete baseline implementations for Player A and Player B live in
:mod:`Q_Sea_Battle.players_base_a` and :mod:`Q_Sea_Battle.players_base_b`. The
legacy names remain accessible from this module via :func:`__getattr__`, which
emits a :class:`DeprecationWarning`.
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
    """Factory/container for a pair of QSeaBattle players.

    Instances of this class hold a shared :class:`~Q_Sea_Battle.game_layout.GameLayout`
    and provide a :meth:`players` method that constructs Player A and Player B
    using that configuration.

    Attributes:
        game_layout: Shared configuration used by both players.
    """

    def __init__(self, game_layout: Optional[GameLayout] = None) -> None:
        """Initialize the container.

        Args:
            game_layout: Shared configuration for both players. If ``None``, a
                default :class:`~Q_Sea_Battle.game_layout.GameLayout` is created.
        """
        self.game_layout: GameLayout = game_layout or GameLayout()

    def players(self) -> Tuple["PlayerA", "PlayerB"]:
        """Create Player A and Player B instances.

        Subclasses may override this method to return specialized player
        implementations while still sharing the same :attr:`game_layout`.

        Returns:
            Tuple ``(player_a, player_b)``.
        """
        player_a = _PlayerA(self.game_layout)
        player_b = _PlayerB(self.game_layout)
        return player_a, player_b

    def reset(self) -> None:
        """Reset any container-level state.

        The base implementation has no internal state. This hook exists for
        compatibility with subclasses that may cache state across games.
        """
        # No state to reset in the base implementation.
        return None


def __getattr__(name: str) -> Any:
    """Provide deprecated module attributes.

    Supports legacy imports such as::

        from Q_Sea_Battle.players_base import PlayerA, PlayerB

    Accessing these names emits a :class:`DeprecationWarning`. The resolved
    symbol is cached in :func:`globals` so the warning is emitted at most once
    per interpreter process per symbol.

    Args:
        name: Attribute name being accessed.

    Returns:
        The requested attribute value.

    Raises:
        AttributeError: If *name* is not provided by this module.
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
    # Expose deprecated names to type checkers without triggering __getattr__.
    from .players_base_a import PlayerA
    from .players_base_b import PlayerB


__all__ = [
    "Players",
    "PlayerA",
    "PlayerB",
]