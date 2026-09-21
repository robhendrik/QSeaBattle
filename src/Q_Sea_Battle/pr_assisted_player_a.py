"""Player A implementation that uses PR-assisted resources.

This module defines :class:`PRAssistedPlayerA`, a concrete Player A strategy
that computes its single communication bit by iteratively "compressing" the
binary field through multiple PR-assisted boxes.

Naming notes:
- Uses ``parent.pr_assisted(level)`` rather than ``parent.shared_randomness(level)``.
- Comments and terminology use "PR-assisted" rather than "shared randomness".
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .game_layout import GameLayout
from .players_base import PlayerA


class PRAssistedPlayerA(PlayerA):
    """Player A implementation using PR-assisted resources.

    The algorithm repeatedly reduces a length-``2^k`` binary vector to length
    ``2^(k-1)`` by:
    1) Computing pairwise equality bits (a "measurement" vector).
    2) Querying the PR-assisted resource for Player A's outcomes at the current
       level.
    3) Combining the original first bit of each pair with the corresponding
       PR-assisted outcome and applying another pairwise equality reduction.

    The final single bit is returned as Player A's communication bit.
    """

    def __init__(self, game_layout: GameLayout, parent: "PRAssistedPlayers") -> None:
        """Initialise a :class:`PRAssistedPlayerA` instance.

        Args:
            game_layout: Game configuration.
            parent: Factory/owner providing access to PR-assisted boxes via
                ``parent.pr_assisted(level)``.

        Raises:
            TypeError: If ``parent`` is not a :class:`.PRAssistedPlayers` instance.
        """
        from .pr_assisted_players import PRAssistedPlayers  # local import to avoid cycle

        if not isinstance(parent, PRAssistedPlayers):
            raise TypeError("parent must be an PRAssistedPlayers instance")  # noqa: TRY003

        super().__init__(game_layout)
        self.parent: PRAssistedPlayers = parent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def decide(self, field: np.ndarray, supp: Any | None = None) -> np.ndarray:
        """Compute Player A's communication bit from the given field.

        Args:
            field: Flattened field array of shape ``(n2,)`` where
                ``n2 == game_layout.field_size**2`` and values are 0/1.
            supp: Optional supporting information (unused).

        Returns:
            A NumPy array of shape ``(1,)`` with dtype ``int`` containing the
            communication bit (0 or 1).

        Raises:
            ValueError: If ``field`` is not a 1D array of the expected length
                or contains values other than 0/1.
            RuntimeError: If the iterative reduction does not end in a single
                bit (should be unreachable if inputs match expectations).
        """
        del supp  # unused

        field = np.asarray(field, dtype=int)
        n2 = self.game_layout.field_size**2

        if field.ndim != 1 or field.shape[0] != n2:
            raise ValueError(f"field must be a 1D array of length {n2}")  # noqa: TRY003
        if not np.all(np.logical_or(field == 0, field == 1)):
            raise ValueError("field must contain only 0/1 values")  # noqa: TRY003

        intermediate_field = field.copy()
        level = 0

        while intermediate_field.size > 1:
            if intermediate_field.size % 2 != 0:
                # The reduction operates on disjoint pairs; each level requires
                # an even-length intermediate field.
                raise ValueError(
                    "intermediate_field length must be even at each level"
                )  # noqa: TRY003

            half = intermediate_field.size // 2

            # Pairwise equality measurement:
            # measurement[k] = 0 if the pair is equal, 1 if the pair differs.
            measurement = np.empty(half, dtype=int)
            for k in range(half):
                a = intermediate_field[2 * k]
                b = intermediate_field[2 * k + 1]
                measurement[k] = 0 if a == b else 1

            # Query Player A's side of the PR-assisted box at this level.
            pr_box = self.parent.pr_assisted(level)
            outcome_a = pr_box.measurement_a(measurement)

            # Interleave the original first bit with the PR-assisted outcome:
            # (original_first_bit, outcome_a_bit) for each pair.
            aux_intermediate = np.empty_like(intermediate_field)
            for k in range(half):
                aux_intermediate[2 * k] = intermediate_field[2 * k]
                aux_intermediate[2 * k + 1] = outcome_a[k]

            # Reduce again via pairwise equality.
            new_intermediate = np.empty(half, dtype=int)
            for k in range(half):
                a = aux_intermediate[2 * k]
                b = aux_intermediate[2 * k + 1]
                new_intermediate[k] = 0 if a == b else 1

            intermediate_field = new_intermediate
            level += 1

        if intermediate_field.size != 1:
            raise RuntimeError("Final intermediate_field must have length 1")  # noqa: TRY003

        comm_bit = int(intermediate_field[0])
        return np.array([comm_bit], dtype=int)