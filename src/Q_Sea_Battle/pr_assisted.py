"""Classical PR-assisted resource for assisted players.

This module provides the same functionality as :class:`SharedRandomness` in
``shared_randomness.py``, but uses the updated naming convention:

- Module: ``pr_assisted.py``
- Class: :class:`PRAssisted`

Author: Rob Hendriks
Package: Q_Sea_Battle
Version: 0.1
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class PRAssisted:
    """Two-party PR-assisted resource with biased correlations.

    This helper models a shared box queried twice per *round*:

    * The first call (by either party A or B) returns a uniformly random
      0/1 string.
    * The second call returns a 0/1 string that is correlated with the
      first according to the measurement settings and ``p_high``.

    The box is stateful within a round (tracking whether A/B already
    measured and what the previous measurement/outcome were) but can be
    reset between rounds.

    Notes:
        This class is functionally identical to ``SharedRandomness`` in
        ``shared_randomness.py``. Only the module/class names and
        documentation have been updated.
    """

    def __init__(self, length: int, p_high: float) -> None:
        """Initialise the PR-assisted resource.

        Args:
            length: Number of bits in each measurement/outcome string.
            p_high: Correlation parameter in [0.0, 1.0].

        Raises:
            TypeError: If argument types are incorrect.
            ValueError: If ``length`` < 1 or ``p_high`` is outside [0, 1].
        """
        if not isinstance(length, int):
            raise TypeError("length must be an int")
        if length < 1:
            raise ValueError("length must be >= 1")

        if not isinstance(p_high, (int, float)):
            raise TypeError("p_high must be a float")
        if not (0.0 <= float(p_high) <= 1.0):
            raise ValueError("p_high must be in the interval [0.0, 1.0]")

        self.length: int = length
        self.p_high: float = float(p_high)

        # Measurement bookkeeping
        self.a_measured: bool = False
        self.b_measured: bool = False

        # Store previous measurement and outcome for the second query.
        self.prev_party: Optional[str] = None  # "a" | "b" | None
        self.prev_measurement: Optional[np.ndarray] = None
        self.prev_outcome: Optional[np.ndarray] = None

        # Random number generator; in a larger system this can be seeded
        # from a global seed for full reproducibility.
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def measurement_a(self, measurement: np.ndarray) -> np.ndarray:
        """Perform a measurement by party A.

        Args:
            measurement: 1D array of shape ``(length,)`` with 0/1 entries.

        Returns:
            1D array of 0/1 outcomes with the same shape as ``measurement``.

        Raises:
            ValueError: If A already measured or if the measurement is invalid.
        """
        if self.a_measured:
            raise ValueError("Party A has already measured on this resource")  # noqa: TRY003

        meas = self._validate_measurement(measurement)
        self.a_measured = True

        if not self.b_measured:
            # First measurement on this box.
            return self._first_measurement("a", meas)

        # B measured first; this is the second measurement.
        return self._second_measurement("a", meas, self.prev_measurement, self.prev_outcome)

    def measurement_b(self, measurement: np.ndarray) -> np.ndarray:
        """Perform a measurement by party B.

        Args:
            measurement: 1D array of shape ``(length,)`` with 0/1 entries.

        Returns:
            1D array of 0/1 outcomes with the same shape as ``measurement``.

        Raises:
            ValueError: If B already measured or if the measurement is invalid.
        """
        if self.b_measured:
            raise ValueError("Party B has already measured on this resource")  # noqa: TRY003

        meas = self._validate_measurement(measurement)
        self.b_measured = True

        if not self.a_measured:
            # First measurement on this box.
            return self._first_measurement("b", meas)

        # A measured first; this is the second measurement.
        return self._second_measurement("b", meas, self.prev_measurement, self.prev_outcome)

    def reset(self) -> None:
        """Reset internal measurement state for reuse of the resource."""
        self.a_measured = False
        self.b_measured = False
        self.prev_party = None
        self.prev_measurement = None
        self.prev_outcome = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_measurement(self, measurement: np.ndarray) -> np.ndarray:
        """Validate and normalise the measurement vector.

        Args:
            measurement: Array-like measurement specification.

        Returns:
            A 1D NumPy array of dtype ``int`` with 0/1 values.

        Raises:
            ValueError: If the shape or value constraints are violated.
        """
        meas = np.asarray(measurement, dtype=int)
        if meas.ndim != 1:
            raise ValueError("measurement must be 1D")
        if meas.shape[0] != self.length:
            raise ValueError(f"measurement must have length {self.length}")
        if not np.all(np.logical_or(meas == 0, meas == 1)):
            raise ValueError("measurement must contain only 0/1 values")
        return meas

    def _random_string(self, n: int) -> np.ndarray:
        """Generate a random 0/1 string of length ``n``."""
        return self._rng.integers(0, 2, size=n, dtype=int)

    def _first_measurement(self, party: str, current_measurement: np.ndarray) -> np.ndarray:
        """Handle the first measurement on the resource.

        Args:
            party: ``"a"`` or ``"b"``.
            current_measurement: 1D measurement vector of length ``length``.

        Returns:
            Random 0/1 outcome array of length ``length``.
        """
        outcome = self._random_string(self.length)

        self.prev_party = party
        self.prev_measurement = current_measurement.copy()
        self.prev_outcome = outcome.copy()

        return outcome

    def _second_measurement(
        self,
        party: str,
        current_measurement: np.ndarray,
        previous_measurement: Optional[np.ndarray],
        previous_outcome: Optional[np.ndarray],
    ) -> np.ndarray:
        """Handle the second measurement with biased correlations.

        The per-index rule is:

        * If (previous_measurement[i], current_measurement[i]) is in
          (0, 0), (0, 1) or (1, 0) then the outcome equals the previous
          outcome with probability ``p_high`` and its complement with
          probability ``1 - p_high``.
        * If it is (1, 1) then the outcome equals the previous outcome
          with probability ``1 - p_high`` and its complement with
          probability ``p_high``.

        All indices are treated independently.

        Args:
            party: ``"a"`` or ``"b"`` (unused, kept for clarity).
            current_measurement: Measurement vector for the second party.
            previous_measurement: Measurement vector from the first party.
            previous_outcome: Outcome vector from the first party.

        Returns:
            1D array of 0/1 outcomes for the second measurement.

        Raises:
            RuntimeError: If called without a first measurement.
            ValueError: If input shapes are inconsistent.
        """
        del party  # symmetric behaviour; party label is for diagnostics only

        if previous_measurement is None or previous_outcome is None:
            raise RuntimeError("Second measurement called without a first measurement")  # noqa: TRY003

        if (
            previous_measurement.shape != current_measurement.shape
            or previous_outcome.shape != current_measurement.shape
        ):
            raise ValueError("Measurement and outcome shapes must all match")  # noqa: TRY003

        prev_meas = previous_measurement
        curr_meas = current_measurement
        prev_out = previous_outcome

        # Determine per index whether we are in the high- or low-correlation
        # configuration based on (prev_meas[i], curr_meas[i]).
        # High-probability case: (0,0), (0,1), (1,0).
        high_mask = (
            ((prev_meas == 0) & (curr_meas == 0))
            | ((prev_meas == 0) & (curr_meas == 1))
            | ((prev_meas == 1) & (curr_meas == 0))
        )
        # Low-probability case is the remaining combination (1,1).
        same_prob = np.where(high_mask, self.p_high, 1.0 - self.p_high)

        base_outcome = prev_out
        u = self._rng.random(self.length)
        keep_mask = u < same_prob

        outcome = np.where(keep_mask, base_outcome, 1 - base_outcome).astype(int)
        return outcome
