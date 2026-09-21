"""PR-assisted player factory and PR-assisted resource hierarchy.

This module implements :class:`PRAssistedPlayers`, a :class:`~.players_base.Players`
factory that owns a hierarchy of :class:`~.pr_assisted.PRAssisted` shared resources
and vends a paired :class:`~.pr_assisted_player_a.PRAssistedPlayerA` /
:class:`~.pr_assisted_player_b.PRAssistedPlayerB`.

The hierarchy is sized from the game layout. The implementation assumes:

* ``game_layout.comms_size == 1`` (a single inter-player communication bit).
* ``n2 = game_layout.field_size ** 2`` is a positive power of two.

Compatibility note:
The older "shared_randomness" terminology has been replaced by "pr_assisted".
A deprecated compatibility alias :meth:`PRAssistedPlayers.shared_randomness` is
retained to reduce breakage in older code paths.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .game_layout import GameLayout
from .players_base import Players, PlayerA, PlayerB
from .pr_assisted import PRAssisted
from .pr_assisted_player_a import PRAssistedPlayerA
from .pr_assisted_player_b import PRAssistedPlayerB


class PRAssistedPlayers(Players):
    """Factory for PR-assisted players.

    The factory constructs and owns a per-level list of
    :class:`~.pr_assisted.PRAssisted` resources. It then creates (and caches) a
    paired :class:`~.pr_assisted_player_a.PRAssistedPlayerA` /
    :class:`~.pr_assisted_player_b.PRAssistedPlayerB` that query these resources
    during play.
    """

    def __init__(self, game_layout: GameLayout, p_rule: float) -> None:
        """Initialize the factory for a specific game layout.

        Args:
            game_layout: Game configuration and board dimensions.
            p_rule: Correlation parameter used for all owned PR-assisted resources.

        Raises:
            ValueError: If ``comms_size != 1`` or if ``field_size ** 2`` is not a
                positive power of two.
        """
        super().__init__(game_layout)

        if self.game_layout.comms_size != 1:
            raise ValueError("PRAssistedPlayers requires comms_size == 1")

        n2 = self.game_layout.field_size ** 2
        if n2 <= 0:
            raise ValueError("field_size must be positive")

        # The PR-assisted hierarchy is defined for an address space whose size is
        # a power of two; here the address space is the flattened field of size
        # field_size**2.
        if n2 & (n2 - 1) != 0:
            raise ValueError("field_size ** 2 must be a power of 2 for PRAssistedPlayers")

        self.p_rule: float = float(p_rule)

        # PR-assisted resources per level (lengths halve each level).
        self._pr_assisted_array: list[PRAssisted] = self._create_pr_assisted_array()

        # Cached player instances; created lazily on first call to players().
        self._playerA: PRAssistedPlayerA | None = None
        self._playerB: PRAssistedPlayerB | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def players(self) -> Tuple[PlayerA, PlayerB]:
        """Return the paired players, creating them on first use.

        The returned players keep a reference to this factory (as their parent)
        and use it to access the per-level PR-assisted resources.

        Returns:
            A tuple ``(player_a, player_b)``.
        """
        if self._playerA is None or self._playerB is None:
            self._playerA = PRAssistedPlayerA(self.game_layout, parent=self)
            self._playerB = PRAssistedPlayerB(self.game_layout, parent=self)
        return self._playerA, self._playerB

    def reset(self) -> None:
        """Reset the owned PR-assisted resources.

        This recreates the internal PR-assisted hierarchy. Cached player objects
        are not recreated, but will observe the new resources through the parent.
        """
        self._pr_assisted_array = self._create_pr_assisted_array()

    def pr_assisted(self, index: int) -> PRAssisted:
        """Return the PR-assisted resource at the given level.

        Args:
            index: Index into the internal PR-assisted resource list.

        Returns:
            The :class:`~.pr_assisted.PRAssisted` instance at ``index``.

        Raises:
            IndexError: If ``index`` is out of bounds.
        """
        return self._pr_assisted_array[index]

    def shared_randomness(self, index: int) -> PRAssisted:
        """Deprecated alias for :meth:`pr_assisted`.

        Args:
            index: Index into the internal PR-assisted resource list.

        Returns:
            The :class:`~.pr_assisted.PRAssisted` instance at ``index``.
        """
        # TODO(review): Replace print-based warning with warnings.warn when API policy allows.
        print("Warning: shared_randomness() is deprecated; use pr_assisted() instead.")
        return self.pr_assisted(index)

    def set_replay_round(self, replay_specs: list[dict]) -> None:
        """Enable replay mode for all owned PR-assisted resources.

        In replay mode, each resource is configured to return prescribed outcomes
        rather than stochastic ones. This is intended for deterministic tests and
        verification across runs.

        This is a convenience wrapper around :meth:`PRAssisted.set_replay_round`
        applied to every owned resource.

        Args:
            replay_specs: A list of per-resource keyword dictionaries. The list
                length must match the number of owned resources. Each dictionary
                is passed to the corresponding resource's ``set_replay_round`` via
                ``**spec``.

        Raises:
            ValueError: If ``replay_specs`` does not match the number of owned
                resources, or if any element is not a dictionary.
        """
        if len(replay_specs) != len(self._pr_assisted_array):
            raise ValueError(
                f"replay_specs length {len(replay_specs)} does not match "
                f"number of PR-assisted resources {len(self._pr_assisted_array)}"
            )

        for i, spec in enumerate(replay_specs):
            if not isinstance(spec, dict):
                raise ValueError(f"replay_specs[{i}] must be a dict")
            self._pr_assisted_array[i].set_replay_round(**spec)

    def clear_replay_round(self) -> None:
        """Disable replay mode for all owned PR-assisted resources.

        This is a convenience wrapper around :meth:`PRAssisted.clear_replay_round`
        applied to every owned resource. After calling this, all resources revert
        to stochastic behavior.
        """
        for box in self._pr_assisted_array:
            box.clear_replay_round()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_pr_assisted_array(self) -> list[PRAssisted]:
        """Create the per-level PR-assisted hierarchy for the current layout.

        Let ``n2 = field_size ** 2`` and assume ``n2 = 2**n`` for some integer
        ``n``. The hierarchy contains ``n`` resources with lengths:

            ``2**(n-1), 2**(n-2), ..., 2**1, 2**0``.

        Returns:
            A list of :class:`~.pr_assisted.PRAssisted` instances, ordered from the
            largest resource (highest level) to the smallest (lowest level).

        Raises:
            ValueError: If ``field_size ** 2`` is not an exact power of two.
        """
        n2 = self.game_layout.field_size ** 2
        n = int(np.log2(n2))
        if 2**n != n2:
            raise ValueError("field_size ** 2 must be an exact power of 2")

        lengths = [2**exp for exp in range(n - 1, -1, -1)]
        return [PRAssisted(length=L, p_rule=self.p_rule) for L in lengths]