"""Deprecated compatibility wrapper for PR-assisted Player B.

`AssistedPlayerB` is deprecated. Use :class:`PRAssistedPlayerB` from
`Q_Sea_Battle.pr_assisted_player_b` instead.

Author: Rob Hendriks
"""

from __future__ import annotations

import warnings
from typing import Any

from .game_layout import GameLayout
from .pr_assisted_player_b import PRAssistedPlayerB as _PRAssistedPlayerB


class AssistedPlayerB(_PRAssistedPlayerB):
    """Deprecated alias for :class:`PRAssistedPlayerB`."""

    def __init__(self, game_layout: GameLayout, parent: Any) -> None:
        warnings.warn(
            "AssistedPlayerB is deprecated; use PRAssistedPlayerB from Q_Sea_Battle.pr_assisted_player_b.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(game_layout=game_layout, parent=parent)
