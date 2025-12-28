"""Assisted Player A using classical shared randomness.

Author: Rob Hendriks
Version: 0.1
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .game_layout import GameLayout
from .players_base import PlayerA


class AssistedPlayerA(PlayerA):
    """Player A implementation using shared randomness resources.

    This player iteratively compresses the field using a stack of
    shared-randomness boxes as specified in the design document. At
    each iteration it:

    * Builds a measurement string based on equality of neighbouring bits.
    * Performs the first measurement on the corresponding shared box.
    * Mixes measurement and outcome into a new intermediate field.
    * Collapses the intermediate representation until only one bit
      remains, from which the final communication bit is read.
    """

    def __init__(self, game_layout: GameLayout, parent: "AssistedPlayers") -> None:
        """Initialise an :class:`AssistedPlayerA` instance.

        Args:
            game_layout: Game configuration.
            parent: Owning :class:`AssistedPlayers` factory providing access
                to the shared randomness boxes.
        """
        from .assisted_players import AssistedPlayers  # local import to avoid cycle

        if not isinstance(parent, AssistedPlayers):
            raise TypeError("parent must be an AssistedPlayers instance")  # noqa: TRY003

        super().__init__(game_layout)
        self.parent: AssistedPlayers = parent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def decide(self, field: np.ndarray, supp: Any | None = None) -> np.ndarray:
        """Compute the communication bit from the field.

        This implements the updated specification for AssistedPlayerA:

        * Start from the flattened field as ``intermediate_field``.
        * While the length of ``intermediate_field`` is larger than 1:
          - Build a measurement string of half the length where the k-th
            bit is 0 if the pair (2k, 2k+1) is equal and 1 otherwise.
          - Query the corresponding shared randomness box with this
            measurement as the *first* measurement.
          - Build an auxiliary array whose pairs are
            (original_first_bit, outcome_a_bit).
          - Collapse this auxiliary array into a new intermediate field
            whose bits are 1 if the pair is equal and 0 otherwise.
        * When only one bits remain, return this
          as the communication bit.

        Args:
            field: Flattened field array of shape ``(n2,)`` with 0/1 values.
            supp: Optional supporting information (unused).

        Returns:
            1D NumPy array of length 1 containing the communication bit.
        """
        del supp  # unused

        field = np.asarray(field, dtype=int)
        n2 = self.game_layout.field_size ** 2

        if field.ndim != 1 or field.shape[0] != n2:
            raise ValueError(f"field must be a 1D array of length {n2}")  # noqa: TRY003
        if not np.all(np.logical_or(field == 0, field == 1)):
            raise ValueError("field must contain only 0/1 values")  # noqa: TRY003

        intermediate_field = field.copy()
        level = 0

        # Iterate until the intermediate field has length 1.
        while intermediate_field.size > 1:
            if intermediate_field.size % 2 != 0:
                raise ValueError(
                    "intermediate_field length must be even at each level"
                )  # noqa: TRY003

            half = intermediate_field.size // 2

            # Step 1: build measurement string based on equality of pairs.
            # According to the updated spec:
            #   - measurement[k] = 0 if pair bits are equal
            #   - measurement[k] = 1 if pair bits are different
            measurement = np.empty(half, dtype=int)
            for k in range(half):
                a = intermediate_field[2 * k]
                b = intermediate_field[2 * k + 1]
                measurement[k] = 0 if a == b else 1

            # Step 2: perform the first measurement on the shared randomness box.
            shared_randomness = self.parent.shared_randomness(level)
            outcome_a = shared_randomness.measurement_a(measurement)

            # Step 3: build an auxiliary intermediate array of the same size
            # as the current intermediate field: (original_first_bit, outcome) pairs.
            aux_intermediate = np.empty_like(intermediate_field)
            for k in range(half):
                aux_intermediate[2 * k] = intermediate_field[2 * k]
                aux_intermediate[2 * k + 1] = outcome_a[k]

            # Step 4: collapse into a new intermediate field where each bit
            # is 1 if the pair in aux_intermediate is equal, else 0.
            new_intermediate = np.empty(half, dtype=int)
            for k in range(half):
                a = aux_intermediate[2 * k]
                b = aux_intermediate[2 * k + 1]
                new_intermediate[k] = 1 if a == b else 0

            intermediate_field = new_intermediate
            level += 1

        # Final field must have length 1; the communication bit is the
        # only element (index 0).
        if intermediate_field.size != 1:
            raise RuntimeError("Final intermediate_field must have length 1")  # noqa: TRY003

        comm_bit = int(intermediate_field[0])
        return np.array([comm_bit], dtype=int)
