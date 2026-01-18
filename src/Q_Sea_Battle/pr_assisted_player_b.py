"""PR-Assisted Player B using PR-assisted resources.

Naming update:
- Uses parent.pr_assisted(level) rather than parent.shared_randomness(level).
- Internal variables/docs renamed from "shared randomness" to "PR-assisted".

Author: Rob Hendriks
Package: Q_Sea_Battle
Version: 0.1
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .game_layout import GameLayout
from .players_base import PlayerB


class PRAssistedPlayerB(PlayerB):
    """Player B implementation using PR-assisted resources."""

    def __init__(self, game_layout: GameLayout, parent: "PRAssistedPlayers") -> None:
        """Initialise an :class:`PRAssistedPlayerB` instance.

        Args:
            game_layout: Game configuration.
            parent: Owning :class:`PRAssistedPlayers` factory providing access
                to the PR-assisted boxes.

        Raises:
            TypeError: If ``parent`` is not an PRAssistedPlayers instance.
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
        """Decide whether to shoot using PR-assisted resources and communication.

        Args:
            gun: One-hot gun vector of shape ``(n2,)`` with values in ``{0, 1}``.
            comm: Communication array of shape ``(1,)`` with values in ``{0, 1}``.
            supp: Optional supporting information (unused).

        Returns:
            ``1`` to shoot or ``0`` to not shoot.
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

        while intermediate_gun.size > 1:
            if intermediate_gun.size % 2 != 0:
                raise ValueError("intermediate_gun length must be even at each level")

            if intermediate_gun.sum() != 1:
                raise ValueError("intermediate_gun must remain one-hot at each level")

            half = intermediate_gun.size // 2
            measurement = np.zeros(half, dtype=int)
            pair_index: int | None = None

            # Build measurement and identify the active pair.
            # measurement[k] = 1 only if pair == (0, 1); else 0.
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

            if measurement.sum() not in (0, 1):
                raise ValueError(
                    "measurement_string must have sum 0 or 1 per specification"
                )

            pr_box = self.parent.pr_assisted(level)
            outcome_b = pr_box.measurement_b(measurement)

            results.append(int(outcome_b[pair_index]))

            new_intermediate_gun = np.zeros(half, dtype=int)
            new_intermediate_gun[pair_index] = 1
            intermediate_gun = new_intermediate_gun
            level += 1

        results.append(int(comm[0]))
        shoot = 1 if (sum(results) % 2) == 1 else 0
        return int(shoot)
