"""SharedRandomnessLayer (Keras layer).

This module implements the core shared-randomness primitive used by the
trainable assisted players.

It mirrors the classical :class:`~Q_Sea_Battle.shared_randomness.SharedRandomness`
correlation rule described in the design document, but as a stateless Keras
layer that can operate in either:

* ``mode='expected'``: deterministic, differentiable expected outcomes in
  ``[0, 1]``.
* ``mode='sample'``: stochastic binary outcomes in ``{0, 1}``, sampled
  deterministically via stateless RNG given a seed.

Author: Rob Hendriks
Version: 0.1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import tensorflow as tf


@dataclass(frozen=True)
class _SRInputs:
    """Validated inputs for :class:`SharedRandomnessLayer`.

    Attributes:
        current_measurement: Tensor of shape ``(..., length)``.
        previous_measurement: Tensor of shape ``(..., length)``.
        previous_outcome: Tensor of shape ``(..., length)``.
        first_measurement: Tensor broadcastable to ``(..., length)``.
    """

    current_measurement: tf.Tensor
    previous_measurement: tf.Tensor
    previous_outcome: tf.Tensor
    first_measurement: tf.Tensor


class SharedRandomnessLayer(tf.keras.layers.Layer):
    """A stateless layer producing correlated outcomes for two-party measurements.

    Args:
        length: Number of bits in each measurement/outcome vector.
        p_high: Correlation parameter in ``[0, 1]``.
        mode: Either ``"expected"`` or ``"sample"``.
        resource_index: Optional identifier for this shared resource.
        seed: Optional integer seed used for deterministic sampling in
            ``mode='sample'``. If ``None``, sampling is still well-defined but
            non-deterministic across processes.
        name: Optional layer name.

    Notes:
        The layer is *stateless* w.r.t. measurement ordering. The caller must
        provide ``first_measurement`` (1 for first, 0 for second) and pass
        the corresponding ``previous_measurement`` and ``previous_outcome``.
    """

    def __init__(
        self,
        length: int,
        p_high: float,
        mode: str = "expected",
        resource_index: Optional[int] = None,
        seed: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        if length <= 0:
            raise ValueError("length must be > 0")
        if not (0.0 <= float(p_high) <= 1.0):
            raise ValueError("p_high must be in [0, 1]")
        if mode not in {"expected", "sample"}:
            raise ValueError("mode must be 'expected' or 'sample'")

        self.length = int(length)
        self.p_high = float(p_high)
        self.mode = mode
        self.resource_index = None if resource_index is None else int(resource_index)
        self.seed = None if seed is None else int(seed)

    def get_config(self) -> Dict[str, Any]:
        """Return layer config for Keras serialization."""

        base = super().get_config()
        base.update(
            {
                "length": self.length,
                "p_high": self.p_high,
                "mode": self.mode,
                "resource_index": self.resource_index,
                "seed": self.seed,
            }
        )
        return base

    def _validate_inputs(self, inputs: Dict[str, tf.Tensor]) -> _SRInputs:
        """Validate and normalize the input dictionary."""

        required = {
            "current_measurement",
            "previous_measurement",
            "previous_outcome",
            "first_measurement",
        }
        if set(inputs.keys()) != required:
            missing = required - set(inputs.keys())
            extra = set(inputs.keys()) - required
            raise ValueError(f"SharedRandomnessLayer.call expects keys {required}. Missing={missing}, extra={extra}")

        curr = tf.convert_to_tensor(inputs["current_measurement"], dtype=tf.float32)
        prev_m = tf.convert_to_tensor(inputs["previous_measurement"], dtype=tf.float32)
        prev_o = tf.convert_to_tensor(inputs["previous_outcome"], dtype=tf.float32)
        first = tf.convert_to_tensor(inputs["first_measurement"], dtype=tf.float32)

        # Shape checks: last dim must be length.
        tf.debugging.assert_equal(
            tf.shape(curr)[-1],
            self.length,
            message="current_measurement last dimension must equal length",
        )
        tf.debugging.assert_equal(
            tf.shape(prev_m)[-1],
            self.length,
            message="previous_measurement last dimension must equal length",
        )
        tf.debugging.assert_equal(
            tf.shape(prev_o)[-1],
            self.length,
            message="previous_outcome last dimension must equal length",
        )

        # Range checks (soft): values should be in [0, 1].
        # These asserts are cheap and prevent silent misuse.
        for name, t in [
            ("current_measurement", curr),
            ("previous_measurement", prev_m),
            ("previous_outcome", prev_o),
        ]:
            tf.debugging.assert_greater_equal(t, 0.0, message=f"{name} must be >= 0")
            tf.debugging.assert_less_equal(t, 1.0, message=f"{name} must be <= 1")

        return _SRInputs(
            current_measurement=curr,
            previous_measurement=prev_m,
            previous_outcome=prev_o,
            first_measurement=first,
        )

    def _stateless_seed(self, stream_id: int) -> tf.Tensor:
        """Create a TensorFlow stateless seed pair.

        Args:
            stream_id: Small integer identifying a random stream (e.g. 0 for
                first-measurement draws, 1 for second-measurement draws).

        Returns:
            ``tf.Tensor`` of shape ``(2,)`` and dtype ``int32``.
        """

        # If seed is None, fall back to TF's global RNG but still deterministic
        # within a graph execution.
        if self.seed is None:
            # Use a random but *process-local* seed; acceptable when user did
            # not request determinism.
            s0 = tf.random.uniform((), maxval=2**31 - 1, dtype=tf.int32)
        else:
            s0 = tf.constant(self.seed, dtype=tf.int32)

        ridx = 0 if self.resource_index is None else int(self.resource_index)
        # Mix in resource index and stream_id deterministically.
        s1 = tf.constant((ridx * 9973 + int(stream_id) * 101) & 0x7FFFFFFF, dtype=tf.int32)
        return tf.stack([s0, s1], axis=0)

    def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Compute correlated outcomes.

        Args:
            inputs: Dictionary with keys:
                - ``current_measurement``: ``(..., length)`` float in ``[0, 1]``
                - ``previous_measurement``: ``(..., length)`` float in ``[0, 1]``
                - ``previous_outcome``: ``(..., length)`` float in ``[0, 1]``
                - ``first_measurement``: ``(..., 1)`` or broadcastable float, where
                  values >= 0.5 mean "first measurement".

        Returns:
            Tensor of shape ``(..., length)``:
                - in ``mode='expected'``: expected outcomes in ``[0, 1]``
                - in ``mode='sample'``: sampled outcomes in ``{0, 1}``
        """

        v = self._validate_inputs(inputs)

        curr = v.current_measurement
        prev_m = v.previous_measurement
        prev_o = v.previous_outcome

        # Broadcast first_measurement to (..., length) in a graph-safe way.
        first = v.first_measurement
        first = tf.cast(first >= 0.5, tf.bool)

        # Ensure shape (..., 1) then broadcast to curr's shape (..., length).
        # Works whether `first` is scalar, (...,), (...,1), etc.
        first = tf.reshape(first, tf.concat([tf.shape(curr)[:-1], [1]], axis=0))
        first = tf.broadcast_to(first, tf.shape(curr))


        ones = tf.ones_like(curr, dtype=tf.float32)

        # First measurement: outcome is uniformly random. In expected mode, it's 0.5.
        out_first_expected = 0.5 * ones

        # Second measurement: compute probability of "same as previous outcome".
        # For hard bits, `both_one` is 1 iff prev_m==1 and curr==1.
        # For relaxed bits, interpret inputs as probabilities and approximate
        # P(both_one) = prev_m * curr.
        both_one = prev_m * curr
        p_high = tf.constant(self.p_high, dtype=tf.float32)
        p_same = p_high + both_one * (1.0 - 2.0 * p_high)

        # Expected second outcome: p_same * prev_o + (1 - p_same) * (1 - prev_o)
        out_second_expected = (1.0 - p_same) + prev_o * (2.0 * p_same - 1.0)

        if self.mode == "expected":
            return tf.where(first, out_first_expected, out_second_expected)

        # Sample mode
        # First measurement samples a random bit string.
        u_first = tf.random.stateless_uniform(tf.shape(curr), seed=self._stateless_seed(stream_id=0))
        out_first_sample = tf.cast(u_first < 0.5, tf.float32)

        # For sample-mode second measurement, require previous_outcome to be binary.
        tf.debugging.assert_near(
            prev_o,
            tf.round(prev_o),
            atol=1e-6,
            message="In mode='sample', previous_outcome must be binary (0/1).",
        )
        u_second = tf.random.stateless_uniform(tf.shape(curr), seed=self._stateless_seed(stream_id=1))
        same_mask = u_second < p_same
        out_second_sample = tf.where(same_mask, prev_o, 1.0 - prev_o)
        out_second_sample = tf.cast(tf.round(out_second_sample), tf.float32)

        return tf.where(first, out_first_sample, out_second_sample)
