"""Deprecated compatibility wrapper for PR-assisted Player A.

`AssistedPlayerA` is deprecated. Use :class:`PRAssistedPlayerA` from
`Q_Sea_Battle.pr_assisted_player_a` instead.

Author: Rob Hendriks
"""

from __future__ import annotations

import warnings
from typing import Any

from .game_layout import GameLayout
from .pr_assisted_player_a import PRAssistedPlayerA as _PRAssistedPlayerA


class AssistedPlayerA(_PRAssistedPlayerA):
    """Deprecated alias for :class:`PRAssistedPlayerA`."""

    def __init__(self, game_layout: GameLayout, parent: Any) -> None:
        warnings.warn(
            "AssistedPlayerA is deprecated; use PRAssistedPlayerA from Q_Sea_Battle.pr_assisted_player_a.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(game_layout=game_layout, parent=parent)
