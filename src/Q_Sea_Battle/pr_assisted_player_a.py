"""PR-Assisted Player A using PR-assisted resources.

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
from .players_base import PlayerA


class PRAssistedPlayerA(PlayerA):
    """Player A implementation using PR-assisted resources.

    This player iteratively compresses the field using a stack of PR-assisted
    boxes as specified in the design document.
    """

    def __init__(self, game_layout: GameLayout, parent: "PRAssistedPlayers") -> None:
        """Initialise an :class:`PRAssistedPlayerA` instance.

        Args:
            game_layout: Game configuration.
            parent: Owning :class:`PRAssistedPlayers` factory providing access
                to the PR-assisted boxes.
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
        """Compute the communication bit from the field.

        Args:
            field: Flattened field array of shape ``(n2,)`` with 0/1 values.
            supp: Optional supporting information (unused).

        Returns:
            1D NumPy array of length 1 containing the communication bit.
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
                raise ValueError(
                    "intermediate_field length must be even at each level"
                )  # noqa: TRY003

            half = intermediate_field.size // 2

            # Build measurement string based on equality of pairs:
            # measurement[k] = 0 if equal, 1 if different.
            measurement = np.empty(half, dtype=int)
            for k in range(half):
                a = intermediate_field[2 * k]
                b = intermediate_field[2 * k + 1]
                measurement[k] = 0 if a == b else 1

            # First measurement on the PR-assisted resource at this level.
            pr_box = self.parent.pr_assisted(level)
            outcome_a = pr_box.measurement_a(measurement)

            # Build auxiliary array (original_first_bit, outcome_a_bit) pairs.
            aux_intermediate = np.empty_like(intermediate_field)
            for k in range(half):
                aux_intermediate[2 * k] = intermediate_field[2 * k]
                aux_intermediate[2 * k + 1] = outcome_a[k]

            # Collapse: new bit is 0 if pair equal, else 1.
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

