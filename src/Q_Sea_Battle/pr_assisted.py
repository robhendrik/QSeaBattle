"""PR-assisted shared resource (SR) for two-party correlations.

This module defines :class:`PRAssisted`, a classical PR-assisted shared resource
queried by two parties (A and B) at most once each per round. The first query in
a round returns a uniformly random bit-string; the second query returns a
bit-string correlated with the first according to the parties' measurement
settings and the parameter ``p_rule``.

The implementation also supports an optional per-round *replay* mode for
deterministic verification: prescribed outcomes can be returned instead of
sampling stochastically. Replay is additive (it does not change default
stochastic behavior when disabled) and is cleared by :meth:`PRAssisted.reset`.

"""

from __future__ import annotations

from typing import Optional

import numpy as np


class PRAssisted:
    """Two-party PR-assisted shared resource with biased correlations.

    The resource is stateful within a *round*:

    - The first call (by either party A or B) returns a uniformly random
      0/1 outcome string.
    - The second call returns a 0/1 outcome string correlated with the first,
      using the per-bit measurement settings and the correlation parameter
      ``p_rule``.

    Each party may query at most once per round. Use :meth:`reset` between
    rounds to clear state.

    Replay mode:
        Replay can be enabled per round via :meth:`set_replay_round`. When
        enabled, outcomes are taken from prescribed vectors rather than sampled
        stochastically. Replay configuration is cleared by :meth:`reset` (and by
        :meth:`clear_replay_round`).

    Attributes:
        length: Number of bits per measurement/outcome string.
        p_rule: Correlation parameter in ``[0.0, 1.0]``.
        a_measured: Whether party A has queried this round.
        b_measured: Whether party B has queried this round.
        prev_party: Party label ("a" or "b") for the first query this round.
        prev_measurement: Measurement vector from the first query (shape
            ``(length,)``).
        prev_outcome: Outcome vector from the first query (shape ``(length,)``).
    """

    def __init__(self, length: int, p_rule: float) -> None:
        """Initialise the PR-assisted resource.

        Args:
            length: Number of bits in each measurement/outcome string. Must be
                >= 1.
            p_rule: Correlation parameter in ``[0.0, 1.0]`` controlling how
                likely the second outcome matches (or flips) the first, per
                index, as a function of the two measurement settings.

        Raises:
            TypeError: If argument types are incorrect.
            ValueError: If ``length`` < 1 or ``p_rule`` is outside ``[0, 1]``.
        """
        if not isinstance(length, int):
            raise TypeError("length must be an int")
        if length < 1:
            raise ValueError("length must be >= 1")

        if not isinstance(p_rule, (int, float)):
            raise TypeError("p_rule must be a float")
        if not (0.0 <= float(p_rule) <= 1.0):
            raise ValueError("p_rule must be in the interval [0.0, 1.0]")

        self.length: int = length
        self.p_rule: float = float(p_rule)

        # Measurement bookkeeping (per round).
        self.a_measured: bool = False
        self.b_measured: bool = False

        # Cache the first query so the second query can correlate with it.
        self.prev_party: Optional[str] = None  # "a" | "b" | None
        self.prev_measurement: Optional[np.ndarray] = None
        self.prev_outcome: Optional[np.ndarray] = None

        # Replay mode state (per round; when disabled, stochastic behavior is
        # unchanged).
        self._replay_enabled: bool = False
        self._replay_a_outcome: Optional[np.ndarray] = None
        self._replay_b_outcome: Optional[np.ndarray] = None
        self._replay_first_party: Optional[str] = None
        self._replay_consumed_a: bool = False
        self._replay_consumed_b: bool = False

        # Local RNG. In an integrated training/evaluation setup this may be
        # seeded externally for reproducibility.
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def measurement_a(self, measurement: np.ndarray) -> np.ndarray:
        """Query the resource for party A.

        The first query of a round returns a uniformly random 0/1 vector. The
        second query returns a vector correlated with the first query according
        to :meth:`_second_measurement`.

        Args:
            measurement: Party A measurement setting as a 1D NumPy array of
                shape ``(length,)`` containing only 0/1 values.

        Returns:
            A 1D NumPy array of dtype ``int`` and shape ``(length,)`` with 0/1
            outcomes.

        Raises:
            ValueError: If party A already queried this round or if the
                measurement is invalid.
            RuntimeError: If replay mode is enabled but the prescribed outcome
                is missing, or if a ``first_party`` constraint is violated.
        """
        if self.a_measured:
            raise ValueError("Party A has already measured on this resource")  # noqa: TRY003

        meas = self._validate_measurement(measurement)
        self.a_measured = True

        # Replay mode: return prescribed outcome (does not enforce correlation).
        if self._replay_enabled:
            if self._replay_first_party is not None and not self.b_measured:
                if self._replay_first_party != "a":
                    raise RuntimeError(
                        f"Replay mode requires first measurement from {self._replay_first_party!r}, "
                        f"but A measured first"
                    )

            if self._replay_a_outcome is None:
                raise RuntimeError(
                    "Replay mode enabled but no outcome prescribed for party A"
                )

            outcome = self._replay_a_outcome.copy()
            self._replay_consumed_a = True

            # Preserve the same round bookkeeping as stochastic mode so that the
            # second call still sees a "first measurement" cached.
            self.prev_party = "a"
            self.prev_measurement = meas.copy()
            self.prev_outcome = outcome.copy()

            return outcome

        # Stochastic mode.
        if not self.b_measured:
            return self._first_measurement("a", meas)

        return self._second_measurement("a", meas, self.prev_measurement, self.prev_outcome)

    def measurement_b(self, measurement: np.ndarray) -> np.ndarray:
        """Query the resource for party B.

        The first query of a round returns a uniformly random 0/1 vector. The
        second query returns a vector correlated with the first query according
        to :meth:`_second_measurement`.

        Args:
            measurement: Party B measurement setting as a 1D NumPy array of
                shape ``(length,)`` containing only 0/1 values.

        Returns:
            A 1D NumPy array of dtype ``int`` and shape ``(length,)`` with 0/1
            outcomes.

        Raises:
            ValueError: If party B already queried this round or if the
                measurement is invalid.
            RuntimeError: If replay mode is enabled but the prescribed outcome
                is missing, or if a ``first_party`` constraint is violated.
        """
        if self.b_measured:
            raise ValueError("Party B has already measured on this resource")  # noqa: TRY003

        meas = self._validate_measurement(measurement)
        self.b_measured = True

        # Replay mode: return prescribed outcome (does not enforce correlation).
        if self._replay_enabled:
            if self._replay_first_party is not None and not self.a_measured:
                if self._replay_first_party != "b":
                    raise RuntimeError(
                        f"Replay mode requires first measurement from {self._replay_first_party!r}, "
                        f"but B measured first"
                    )

            if self._replay_b_outcome is None:
                raise RuntimeError(
                    "Replay mode enabled but no outcome prescribed for party B"
                )

            outcome = self._replay_b_outcome.copy()
            self._replay_consumed_b = True

            # Preserve the same round bookkeeping as stochastic mode.
            self.prev_party = "b"
            self.prev_measurement = meas.copy()
            self.prev_outcome = outcome.copy()

            return outcome

        # Stochastic mode.
        if not self.a_measured:
            return self._first_measurement("b", meas)

        return self._second_measurement("b", meas, self.prev_measurement, self.prev_outcome)

    def reset(self) -> None:
        """Reset the resource for the next round.

        Clears per-round measurement bookkeeping and disables/clears replay
        configuration.
        """
        self.a_measured = False
        self.b_measured = False
        self.prev_party = None
        self.prev_measurement = None
        self.prev_outcome = None
        self.clear_replay_round()

    def set_replay_round(
        self,
        *,
        a_outcome: Optional[np.ndarray] = None,
        b_outcome: Optional[np.ndarray] = None,
        first_party: Optional[str] = None,
    ) -> None:
        """Enable replay mode for the current round.

        When replay is enabled, :meth:`measurement_a` and :meth:`measurement_b`
        return the prescribed outcomes (if provided) instead of sampling.

        Args:
            a_outcome: Optional prescribed outcome for party A. Must be a 1D
                0/1 vector of shape ``(length,)``.
            b_outcome: Optional prescribed outcome for party B. Must be a 1D
                0/1 vector of shape ``(length,)``.
            first_party: Optional party label ("a" or "b"). If provided, enforces
                which party must make the first query in this round.

        Raises:
            ValueError: If outcome shapes/values are invalid or ``first_party``
                is not in ``{"a", "b", None}``.
        """
        if first_party is not None and first_party not in ("a", "b"):
            raise ValueError("first_party must be 'a', 'b', or None")

        validated_a = None
        validated_b = None

        if a_outcome is not None:
            validated_a = self._validate_replay_outcome(a_outcome)

        if b_outcome is not None:
            validated_b = self._validate_replay_outcome(b_outcome)

        self._replay_enabled = True
        self._replay_a_outcome = validated_a
        self._replay_b_outcome = validated_b
        self._replay_first_party = first_party
        self._replay_consumed_a = False
        self._replay_consumed_b = False

    def clear_replay_round(self) -> None:
        """Disable replay mode and clear replay configuration for this round."""
        self._replay_enabled = False
        self._replay_a_outcome = None
        self._replay_b_outcome = None
        self._replay_first_party = None
        self._replay_consumed_a = False
        self._replay_consumed_b = False

    def replay_enabled(self) -> bool:
        """Whether replay mode is enabled for the current round."""
        return self._replay_enabled

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_measurement(self, measurement: np.ndarray) -> np.ndarray:
        """Validate and normalize a measurement vector.

        Args:
            measurement: Array-like measurement setting.

        Returns:
            A 1D NumPy array of dtype ``int`` and shape ``(length,)`` containing
            only 0/1 values.

        Raises:
            ValueError: If the measurement is not 1D, has the wrong length, or
                contains values other than 0/1.
        """
        meas = np.asarray(measurement, dtype=int)
        if meas.ndim != 1:
            raise ValueError("measurement must be 1D")
        if meas.shape[0] != self.length:
            raise ValueError(f"measurement must have length {self.length}")
        if not np.all(np.logical_or(meas == 0, meas == 1)):
            raise ValueError("measurement must contain only 0/1 values")
        return meas

    def _validate_replay_outcome(self, outcome: np.ndarray) -> np.ndarray:
        """Validate and normalize a replay outcome vector.

        Args:
            outcome: Array-like prescribed outcome.

        Returns:
            A 1D NumPy array of dtype ``int`` and shape ``(length,)`` containing
            only 0/1 values.

        Raises:
            ValueError: If the outcome is not 1D, has the wrong length, or
                contains values other than 0/1.
        """
        out = np.asarray(outcome, dtype=int)
        if out.ndim != 1:
            raise ValueError("replay outcome must be 1D")
        if out.shape[0] != self.length:
            raise ValueError(f"replay outcome must have length {self.length}")
        if not np.all(np.logical_or(out == 0, out == 1)):
            raise ValueError("replay outcome must contain only 0/1 values")
        return out

    def _random_string(self, n: int) -> np.ndarray:
        """Sample a uniformly random 0/1 vector.

        Args:
            n: Number of bits.

        Returns:
            1D NumPy array of dtype ``int`` with shape ``(n,)``.
        """
        return self._rng.integers(0, 2, size=n, dtype=int)

    def _first_measurement(self, party: str, current_measurement: np.ndarray) -> np.ndarray:
        """Handle the first query in a round.

        Args:
            party: Party label, "a" or "b".
            current_measurement: 1D measurement vector of shape ``(length,)``.

        Returns:
            A uniformly random 0/1 outcome vector of shape ``(length,)``.
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
        """Handle the second query in a round with biased correlations.

        Per index ``i``, the correlation between the second outcome bit and the
        first outcome bit depends on the pair of measurement settings:

        - For settings (0,0), (0,1), (1,0): the second bit equals the first bit
          with probability ``p_rule``; otherwise it is flipped.
        - For settings (1,1): the second bit equals the first bit with
          probability ``1 - p_rule``; otherwise it is flipped.

        Indices are treated independently.

        Args:
            party: Party label, "a" or "b" (unused; symmetric behavior).
            current_measurement: Measurement vector for the second query (shape
                ``(length,)``).
            previous_measurement: Measurement vector from the first query (shape
                ``(length,)``).
            previous_outcome: Outcome vector from the first query (shape
                ``(length,)``).

        Returns:
            1D NumPy array of dtype ``int`` and shape ``(length,)`` for the
            second outcome.

        Raises:
            RuntimeError: If called without a cached first measurement/outcome.
            ValueError: If measurement/outcome shapes are inconsistent.
        """
        del party  # Symmetric behavior; kept in signature for call-site clarity.

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

        # Map each (prev_meas[i], curr_meas[i]) pair to the probability that the
        # second outcome bit matches the first.
        high_mask = (
            ((prev_meas == 0) & (curr_meas == 0))
            | ((prev_meas == 0) & (curr_meas == 1))
            | ((prev_meas == 1) & (curr_meas == 0))
        )
        same_prob = np.where(high_mask, self.p_rule, 1.0 - self.p_rule)

        u = self._rng.random(self.length)
        keep_mask = u < same_prob

        outcome = np.where(keep_mask, prev_out, 1 - prev_out).astype(int)
        return outcome
