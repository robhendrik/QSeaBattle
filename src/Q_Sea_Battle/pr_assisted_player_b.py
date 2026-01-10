"""PR-Assisted Player B using PR-assisted resources.

Author: Rob Hendriks
Version: 0.1
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .game_layout import GameLayout
from .players_base import PlayerB


class PRAssistedPlayerB(PlayerB):
    """Player B implementation using PR-assisted resources.

    This player mirrors the hierarchical structure used by
    :class:`PRAssistedPlayerA`. It traces the position of the gun through
    the same stack of PR-assisted boxes and collects one outcome
    bit per level, followed by the communication bit. The final decision
    is the parity (XOR) of all collected bits and the communication bit,
    as specified in the design document.
    """

    def __init__(self, game_layout: GameLayout, parent: "PRAssistedPlayers") -> None:
        """Initialise an :class:`PRAssistedPlayerB` instance.

        Args:
            game_layout: Game configuration.
            parent: Owning :class:`PRAssistedPlayers` factory providing access
                to the PR-assisted boxes.

        Raises:
            TypeError: If ``parent`` is not an PRAssistedPlayers instance.
        """
        # Local import to avoid an import cycle at module level.
        from .pr_assisted_players import PRAssistedPlayers  # type: ignore

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

        This follows the updated specification:

        * Maintain an ``intermediate_gun`` that is always one-hot.
        * At each level, locate the unique pair (0, 1) or (1, 0) in
          ``intermediate_gun``.
        * Build ``measurement_string`` (half the size) with bits:

              1  only if the pair is (0, 1);
              0  otherwise (including (1, 0) and equal pairs).

          and assert that its sum is 0 or 1.
        * Query the corresponding shared randomness box with
          ``measurement_string`` via ``measurement_b`` and append the
          outcome at the active pair index to ``results``.
        * Set the new ``intermediate_gun`` to a one-hot vector (length
          half the previous) with a 1 at that pair index.
        * After all levels, append the communication bit to ``results``.
          The shoot decision is the parity (XOR) of all collected outcome
          bits and the communication bit.

        Args:
            gun: One-hot gun vector of shape ``(n2,)`` with values in
                ``{0, 1}``.
            comm: Communication array of shape ``(1,)`` with values in
                ``{0, 1}``.
            supp: Optional supporting information (unused).

        Returns:
            ``1`` to shoot or ``0`` to not shoot.

        Raises:
            ValueError: If input validation fails or if the intermediate
                structures are inconsistent with the specification.
        """
        del supp  # unused

        gun = np.asarray(gun, dtype=int)
        comm = np.asarray(comm, dtype=int)

        n2 = self.game_layout.field_size ** 2
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

        # Trace the gun position through the hierarchy of shared boxes.
        while intermediate_gun.size > 1:
            if intermediate_gun.size % 2 != 0:
                raise ValueError(
                    "intermediate_gun length must be even at each level"
                )

            if intermediate_gun.sum() != 1:
                raise ValueError(
                    "intermediate_gun must remain one-hot at each level"
                )

            half = intermediate_gun.size // 2
            measurement = np.zeros(half, dtype=int)
            pair_index: int | None = None

            # Build measurement_string and identify the active pair.
            # - measurement[k] = 1 only if pair == (0, 1)
            # - pair_index is the unique k where pair is (0, 1) or (1, 0)
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

            shared_randomness = self.parent.shared_randomness(level)
            outcome_b = shared_randomness.measurement_b(measurement)

            # Store the outcome bit corresponding to the active pair.
            results.append(int(outcome_b[pair_index]))

            # New intermediate_gun is a one-hot vector of length `half`
            # with a 1 at `pair_index`.
            new_intermediate_gun = np.zeros(half, dtype=int)
            new_intermediate_gun[pair_index] = 1
            intermediate_gun = new_intermediate_gun

            level += 1

        # Combine collected randomness bits with the communication bit.
        results.append(int(comm[0]))
        total = sum(results)
        shoot = 1 if (total % 2) == 1 else 0
        return int(shoot)
