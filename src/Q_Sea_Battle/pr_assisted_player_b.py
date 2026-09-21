"""Player B using PR-assisted resources.

This module implements the Player B decision rule for the PR-assisted variant of
QSeaBattle. Player B receives:

- ``gun``: a one-hot vector identifying the targeted grid cell.
- ``comm``: a single classical communication bit.

The decision reduces the one-hot ``gun`` vector level-by-level by pairing
adjacent entries and querying a PR-assisted box at each level. Each level
produces one outcome bit for the unique "active" pair (the only pair that can be
``(0, 1)`` or ``(1, 0)`` given a one-hot input). The final action is the parity
(sum modulo 2) of all collected outcome bits plus the communicated bit.

Naming update:
- Uses ``parent.pr_assisted(level)`` rather than ``parent.shared_randomness(level)``.
- Internal variables/docs renamed from "shared randomness" to "PR-assisted".
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .game_layout import GameLayout
from .players_base import PlayerB


class PRAssistedPlayerB(PlayerB):
    """Player B implementation using PR-assisted resources.

    The PR-assisted resources are provided by the owning ``PRAssistedPlayers``
    instance (passed as ``parent``). The reduction proceeds over multiple
    ``level`` values; at each level, the corresponding PR-assisted box is used to
    generate an outcome bit from Player B's measurement string.
    """

    def __init__(self, game_layout: GameLayout, parent: "PRAssistedPlayers") -> None:
        """Initialize a PR-assisted Player B.

        Args:
            game_layout: Game configuration (field size, etc.).
            parent: Owning factory providing access to PR-assisted boxes via
                ``parent.pr_assisted(level)``.

        Raises:
            TypeError: If ``parent`` is not a ``PRAssistedPlayers`` instance.
        """
        from .pr_assisted_players import PRAssistedPlayers  # local import to avoid cycle

        if not isinstance(parent, PRAssistedPlayers):
            raise TypeError("parent must be an PRAssistedPlayers instance")  # noqa: TRY003

        super().__init__(game_layout)
        self.parent: PRAssistedPlayers = parent

    def decide(
        self,
        gun: np.ndarray,
        comm: np.ndarray,
        supp: Any | None = None,
    ) -> int:
        """Decide whether to shoot.

        This method validates inputs, iteratively compresses the one-hot ``gun``
        vector by a factor of two per level, and queries one PR-assisted box per
        level. At each level, Player B constructs a measurement string
        ``measurement`` of length ``len(gun)/2`` where each element corresponds
        to a pair ``(gun[2k], gun[2k+1])``:

        - ``measurement[k] = 1`` only for the ordered pair ``(0, 1)``.
        - ``measurement[k] = 0`` otherwise (including ``(1, 0)`` and ``(0, 0)``).

        Due to the one-hot constraint, exactly one pair per level is "active"
        (either ``(0, 1)`` or ``(1, 0)``). The outcome bit used for the parity
        computation is the PR-assisted measurement outcome at that active pair.

        Args:
            gun: One-hot gun vector of shape ``(n2,)`` with values in ``{0, 1}``,
                where ``n2 == field_size**2``.
            comm: Communication array of shape ``(1,)`` with values in ``{0, 1}``.
                The single element is appended to the outcome bits prior to the
                parity computation.
            supp: Optional supporting information (unused).

        Returns:
            ``1`` to shoot, or ``0`` to not shoot.

        Raises:
            ValueError: If ``gun`` or ``comm`` do not satisfy the expected shapes
                and bit-value constraints, or if intermediate invariants are
                violated during the level-by-level reduction.
        """
        del supp  # unused

        gun = np.asarray(gun, dtype=int)
        comm = np.asarray(comm, dtype=int)

        n2 = self.game_layout.field_size**2
        if gun.ndim != 1 or gun.shape[0] != n2:
            raise ValueError(f"gun must be a 1D array of length {n2}")
        if not np.all((gun == 0) | (gun == 1)):
            raise ValueError("gun must contain only 0/1 values")
        if gun.sum() != 1:
            raise ValueError("gun must be one-hot (sum equal to 1)")

        if comm.ndim != 1 or comm.shape[0] != 1:
            raise ValueError("comm must be a 1D array of length 1")
        if not np.all((comm == 0) | (comm == 1)):
            raise ValueError("comm must contain only 0/1 values")

        intermediate_gun = gun.copy()
        results: list[int] = []
        level = 0

        # Iteratively halve the one-hot vector. Each level selects the index of
        # the unique active pair and records the PR-assisted outcome bit at that
        # position.
        while intermediate_gun.size > 1:
            if intermediate_gun.size % 2 != 0:
                raise ValueError("intermediate_gun length must be even at each level")

            if intermediate_gun.sum() != 1:
                raise ValueError("intermediate_gun must remain one-hot at each level")

            half = intermediate_gun.size // 2
            measurement = np.zeros(half, dtype=int)
            pair_index: int | None = None

            # Build the measurement string and identify the unique active pair.
            # The one-hot constraint implies there can be at most one adjacent
            # pair with entries (0, 1) or (1, 0).
            for k in range(half):
                a = intermediate_gun[2 * k]
                b = intermediate_gun[2 * k + 1]

                if (a, b) in ((0, 1), (1, 0)):
                    if pair_index is not None:
                        raise ValueError(
                            "there must be at most one active pair "
                            "(0, 1) or (1, 0) per level"
                        )
                    pair_index = k

                measurement[k] = 1 if (a == 0 and b == 1) else 0

            if pair_index is None:
                raise ValueError(
                    "expected exactly one active pair (0, 1) or (1, 0); found none"
                )

            # Per construction, the measurement string can only contain a single
            # 1 (for the ordered active pair (0, 1)); the active pair (1, 0)
            # contributes a 0.
            if measurement.sum() not in (0, 1):
                raise ValueError(
                    "measurement_string must have sum 0 or 1 per specification"
                )

            pr_box = self.parent.pr_assisted(level)
            outcome_b = pr_box.measurement_b(measurement)

            # Record the outcome bit corresponding to the active pair index.
            results.append(int(outcome_b[pair_index]))

            # Reduce to a new one-hot vector over pairs.
            new_intermediate_gun = np.zeros(half, dtype=int)
            new_intermediate_gun[pair_index] = 1
            intermediate_gun = new_intermediate_gun
            level += 1

        # The final decision is the parity of PR-assisted outcomes and the
        # communication bit.
        results.append(int(comm[0]))
        shoot = 1 if (sum(results) % 2) == 1 else 0
        return int(shoot)