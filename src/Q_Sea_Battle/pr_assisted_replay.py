"""PR-assisted replay shared-resource (SR) Keras layer.

This module implements the PR-assisted *replay* SR layer described in the project's
frozen markdown specifications. The layer is a logits-in/logits-out component used
to generate an outcome for either the first or second measurement in a turn, as
indicated explicitly by the caller via ``first_measurement``.

Key properties (contract-level):
- Stateless with respect to ordering: callers must provide ``first_measurement`` to
  indicate whether this call corresponds to the first or second measurement.
- Single-output API: each call produces exactly one outcome tensor (logits).
- Logits-only I/O: the layer does not apply sigmoid/thresholding to produce bits.
  Logical bit values represented as logits are interpreted by sign outside this layer.
- Two SR modes:
  * ``sr_mode="replay"``: training/trace-replay mode. For first measurements, the
    outcome is prescribed by ``replay_outcome_logits`` (identity). For second
    measurements, the outcome follows the PR rule with probability ``p_rule`` and
    violates it (negated logits) otherwise.
  * ``sr_mode="stochastic"``: gameplay mode. For first measurements, the outcome is
    sampled uniformly at random (50/50), independent of ``p_rule``. For second
    measurements, the outcome follows the PR rule with probability ``p_rule`` and
    violates it (negated logits) otherwise.

Differentiability requirement:
- The second-measurement PR gate is differentiable with respect to both measurement
  logits. Hard thresholding on logits (e.g., ``logit >= 0``) is avoided in the PR
  computation and replaced by a smooth "soft-high" gate.

Notes:
- "Opposite" in logits space is implemented as negation: ``-logits``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import tensorflow as tf


@dataclass(frozen=True)
class PRAssistedReplayConfig:
    """Immutable construction-time configuration for :class:`PRAssistedReplay`.

    This object captures the values passed to ``__init__`` for serialization and
    inspection. Runtime execution uses internal ``tf.Variable`` instances for
    selected parameters to support updates via ``set_*`` methods.

    Attributes:
        sr_mode: Shared-resource mode. One of ``"replay"`` or ``"stochastic"``.
        p_rule: Probability of following the PR rule on the second measurement.
        beta: Magnitude used by the hard-logit mapping from bits: {0,1} -> {-beta,+beta}.
        alpha: Sharpness (inverse temperature) for the differentiable soft-high gate.
        seed: RNG seed used for stochastic sampling (when provided).
    """

    sr_mode: str = "replay"  # {"replay", "stochastic"}
    p_rule: float = 1.0  # used ONLY for stochastic second measurement / forward noise
    beta: float = 10.0  # hard_logit mapping: {0,1} -> {-beta,+beta}
    alpha: float = 5.0  # soft-high sharpness/temperature for replay second measurement
    seed: Optional[int] = None  # RNG seed (only used for stochastic sampling)


class PRAssistedReplay(tf.keras.layers.Layer):
    """PR-assisted shared-resource layer (SR) with logits-only I/O.

    Inputs are passed as a dictionary with the following keys:

    - ``current_measurement``: ``tf.Tensor`` float32, shape ``(..., k)``, logits.
    - ``previous_measurement``: ``tf.Tensor`` float32, shape ``(..., k)``, logits.
    - ``previous_outcome``: ``tf.Tensor`` float32, shape ``(..., k)``, logits.
    - ``first_measurement``: ``tf.Tensor`` float32, shape ``(..., 1)``,
      broadcastable to ``(..., k)``. Values should be in {0,1}; a value >= 0.5
      is treated as "first measurement".
    - ``replay_outcome_logits`` (optional): ``tf.Tensor`` float32, shape ``(..., k)``,
      logits. Required in replay mode when any element indicates a first measurement.

    Output:
        A single ``tf.Tensor`` float32 of shape ``(..., k)`` containing outcome logits.

    Implementation notes:
    - The PR-rule outcome is computed via a differentiable "soft-high" gate:
      ``p_high(x) = sigmoid(alpha * x)``. The probability of a PR sign flip is the
      soft AND: ``p_flip = p_high(prev) * p_high(curr)``. The interpolated sign is
      ``(1 - 2 * p_flip)`` in [+1, -1], applied multiplicatively to
      ``previous_outcome``.
    - Noise on the PR outcome (follow vs violate) can be sampled or replaced by its
      expectation, controlled by ``set_pr_noise_mode()``.
    """

    _SR_MODE_REPLAY = 0
    _SR_MODE_STOCHASTIC = 1

    _PR_NOISE_SAMPLED = 0
    _PR_NOISE_EXPECTED = 1

    def __init__(
        self,
        sr_mode: str = "replay",
        *,
        p_rule: float = 1.0,
        beta: float = 10.0,
        alpha: float = 5.0,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the layer.

        Args:
            sr_mode: SR mode; one of ``"replay"`` or ``"stochastic"``.
            p_rule: Probability of following the PR rule for second measurements.
            beta: Magnitude for mapping bits to logits in stochastic first measurements.
            alpha: Sharpness for the differentiable PR gate.
            seed: Optional RNG seed for reproducible stochastic sampling.
            **kwargs: Forwarded to ``tf.keras.layers.Layer``.
        """
        super().__init__(**kwargs)

        self._validate_sr_mode(sr_mode)
        self._validate_p_rule(p_rule)
        self._validate_beta(beta)
        self._validate_alpha(alpha)

        self._cfg = PRAssistedReplayConfig(
            sr_mode=sr_mode,
            p_rule=float(p_rule),
            beta=float(beta),
            alpha=float(alpha),
            seed=seed,
        )

        # Runtime-settable, TF-friendly knobs. These variables drive execution.
        self._alpha = tf.Variable(float(alpha), dtype=tf.float32, trainable=False, name="alpha")
        self._p_rule = tf.Variable(float(p_rule), dtype=tf.float32, trainable=False, name="p_rule")
        self._beta = tf.Variable(float(beta), dtype=tf.float32, trainable=False, name="beta")
        self._sr_mode_code = tf.Variable(
            self._encode_sr_mode(sr_mode),
            dtype=tf.int32,
            trainable=False,
            name="sr_mode_code",
        )
        self._pr_noise_mode_code = tf.Variable(
            self._PR_NOISE_SAMPLED,
            dtype=tf.int32,
            trainable=False,
            name="pr_noise_mode_code",
        )  # default: sampled

        # Use a dedicated Generator so stochastic behavior is reproducible when a seed is provided.
        # Keep this construction unchanged for backward compatibility.
        if seed is None:
            self._rng = tf.random.Generator.from_non_deterministic_state()
        else:
            self._rng = tf.random.Generator.from_seed(int(seed))

    @property
    def config(self) -> PRAssistedReplayConfig:
        """Return the immutable construction-time configuration."""
        return self._cfg

    def get_config(self) -> dict[str, Any]:
        """Return the Keras-serializable configuration dict."""
        base = super().get_config()
        base.update(
            {
                "sr_mode": self._cfg.sr_mode,
                "p_rule": self._cfg.p_rule,
                "beta": self._cfg.beta,
                "alpha": self._cfg.alpha,
                "seed": self._cfg.seed,
            }
        )
        return base

    @staticmethod
    def _validate_sr_mode(sr_mode: str) -> None:
        """Validate the SR mode string."""
        if sr_mode not in {"replay", "stochastic"}:
            raise ValueError(f"sr_mode must be one of {{'replay','stochastic'}}, got {sr_mode!r}")

    @staticmethod
    def _validate_p_rule(p_rule: float) -> None:
        """Validate that ``p_rule`` is a probability."""
        if not (0.0 <= float(p_rule) <= 1.0):
            raise ValueError(f"p_rule must be in [0,1], got {p_rule!r}")

    @staticmethod
    def _validate_beta(beta: float) -> None:
        """Validate that ``beta`` is positive."""
        if float(beta) <= 0.0:
            raise ValueError(f"beta must be > 0, got {beta!r}")

    @staticmethod
    def _validate_alpha(alpha: float) -> None:
        """Validate that ``alpha`` is positive."""
        if float(alpha) <= 0.0:
            raise ValueError(f"alpha must be > 0, got {alpha!r}")

    @classmethod
    def _encode_sr_mode(cls, sr_mode: str) -> int:
        """Encode the mode string into an integer for TF control flow."""
        cls._validate_sr_mode(sr_mode)
        return cls._SR_MODE_REPLAY if sr_mode == "replay" else cls._SR_MODE_STOCHASTIC

    def set_alpha(self, alpha: float) -> None:
        """Update the runtime ``alpha`` (soft gate sharpness)."""
        a = float(alpha)
        self._validate_alpha(a)
        self._alpha.assign(a)

    def set_p_rule(self, p_rule: float) -> None:
        """Update the runtime ``p_rule`` (follow probability)."""
        p = float(p_rule)
        self._validate_p_rule(p)
        self._p_rule.assign(p)

    def set_beta(self, beta: float) -> None:
        """Update the runtime ``beta`` used for bit->logit mapping."""
        b = float(beta)
        self._validate_beta(b)
        self._beta.assign(b)

    def set_sr_mode(self, sr_mode: str) -> None:
        """Update the runtime SR mode."""
        code = self._encode_sr_mode(sr_mode)
        self._sr_mode_code.assign(code)

    def set_pr_noise_mode(self, mode: str) -> None:
        """Update PR noise handling for replay-mode second measurements.

        Args:
            mode: Either ``"sampled"`` to sample follow/violate decisions using
                ``p_rule``, or ``"expected"`` to use the expectation of that noise.
        """
        if mode == "sampled":
            self._pr_noise_mode_code.assign(self._PR_NOISE_SAMPLED)
        elif mode == "expected":
            self._pr_noise_mode_code.assign(self._PR_NOISE_EXPECTED)
        else:
            raise ValueError(
                f"pr_noise_mode must be one of {{'expected','sampled'}}, got {mode!r}"
            )

    def get_pr_noise_mode(self) -> str:
        """Return the current PR noise mode as a Python string.

        Note:
            This method reads the underlying ``tf.Variable`` value via ``.numpy()``,
            so it is intended for eager-mode inspection/debugging.
        """
        code = int(self._pr_noise_mode_code.numpy())
        return "expected" if code == self._PR_NOISE_EXPECTED else "sampled"

    @staticmethod
    def _require_key(inputs: dict[str, tf.Tensor], key: str) -> tf.Tensor:
        """Fetch a required key from the inputs dict."""
        if key not in inputs:
            raise ValueError(f"Missing required input key: {key!r}")
        return inputs[key]

    @staticmethod
    def _ensure_float32(x: tf.Tensor, name: str) -> tf.Tensor:
        """Convert input to a float32 tensor (casting if needed)."""
        x = tf.convert_to_tensor(x)
        if x.dtype != tf.float32:
            x = tf.cast(x, tf.float32, name=f"{name}_float32")
        return x

    def _hard_logit_from_bit(self, bit01: tf.Tensor) -> tf.Tensor:
        """Map {0,1} values to {-beta,+beta} logits.

        Args:
            bit01: Tensor containing 0/1 values (any numeric type).

        Returns:
            Float32 tensor of logits with the same shape as ``bit01``.
        """
        bit01 = tf.cast(bit01, tf.float32)
        return (2.0 * bit01 - 1.0) * self._beta

    def _sample_uniform_bits(self, shape: tf.Tensor) -> tf.Tensor:
        """Sample Bernoulli(0.5) bits using the internal RNG."""
        u = self._rng.uniform(shape=shape, minval=0.0, maxval=1.0, dtype=tf.float32)
        return tf.cast(u < 0.5, tf.float32)

    def _pr_outcome_logits(self, prev_meas: tf.Tensor, curr_meas: tf.Tensor, prev_out: tf.Tensor) -> tf.Tensor:
        """Compute PR-rule outcome logits using a differentiable soft gate.

        The gate is a smooth approximation to "both measurements are logically high".
        It uses ``sigmoid(alpha * logit)`` as a soft indicator of a high bit.

        Args:
            prev_meas: Previous measurement logits.
            curr_meas: Current measurement logits.
            prev_out: Previous outcome logits.

        Returns:
            Outcome logits after applying the soft PR sign flip to ``prev_out``.
        """
        # Soft-high mapping: p_high(x) = sigmoid(alpha * x)
        p_prev = tf.sigmoid(self._alpha * prev_meas)
        p_curr = tf.sigmoid(self._alpha * curr_meas)

        # Soft AND for "high, high".
        p_flip = p_prev * p_curr

        # Interpolated sign flip: (1 - 2*p_flip) in [+1,-1].
        return (1.0 - 2.0 * p_flip) * prev_out

    def _runtime_follow_mask(self, shape: tf.Tensor) -> tf.Tensor:
        """Sample a Bernoulli(p_rule) boolean mask using the runtime ``p_rule``."""
        u = self._rng.uniform(shape=shape, minval=0.0, maxval=1.0, dtype=tf.float32)
        return u < self._p_rule

    def _call_replay_branch(
        self,
        curr_meas: tf.Tensor,
        prev_meas: tf.Tensor,
        prev_out: tf.Tensor,
        is_first: tf.Tensor,
        inputs: dict[str, tf.Tensor],
    ) -> tf.Tensor:
        """Execute replay-mode behavior.

        In replay mode, first-measurement outcomes are prescribed by
        ``replay_outcome_logits`` (when any element indicates first measurement).
        Second-measurement outcomes are computed via the PR rule and then optionally
        noised by follow/violate decisions controlled by ``p_rule``.
        """
        if "replay_outcome_logits" not in inputs:
            # Strict validation: if any element indicates "first measurement", the
            # prescribed replay logits must be provided. Use TF assertions so the
            # check remains effective under tf.function.
            if tf.executing_eagerly():
                # Eager mode (unit tests): raise a Python exception so pytest can catch it naturally.
                if bool(tf.reduce_any(is_first).numpy()):
                    raise ValueError("replay_outcome_logits is required for replay-mode first measurement")
            else:
                # Graph mode: raise InvalidArgumentError via TF assertion when violated.
                tf.debugging.assert_equal(
                    tf.reduce_any(is_first),
                    False,
                    message="replay_outcome_logits is required for replay-mode first measurement",
                )

        replay_out = inputs.get("replay_outcome_logits", None)
        if replay_out is not None:
            replay_out = self._ensure_float32(replay_out, "replay_outcome_logits")

        pr_out_clean = self._pr_outcome_logits(prev_meas, curr_meas, prev_out)

        def expected_branch():
            # Expectation over follow/violate: E[sign] = (2*p_rule - 1).
            return (2.0 * self._p_rule - 1.0) * pr_out_clean

        def sampled_branch():
            follow = self._runtime_follow_mask(tf.shape(curr_meas))
            return tf.where(follow, pr_out_clean, -pr_out_clean)

        pr_out_noisy = tf.cond(
            tf.equal(self._pr_noise_mode_code, self._PR_NOISE_EXPECTED),
            expected_branch,
            sampled_branch,
        )

        # Keep current forward behavior unchanged.
        pr_out = pr_out_noisy  # pr_out_clean + tf.stop_gradient(pr_out_noisy - pr_out_clean)

        if replay_out is None:
            # No first measurement is allowed if replay_out is absent (validated above).
            return pr_out

        # Combine per element: if first -> replay_out, else -> PR outcome.
        # Broadcasting note: is_first (...,1) broadcasts to (...,k).
        return tf.where(is_first, replay_out, pr_out)

    def _call_stochastic_branch(
        self,
        curr_meas: tf.Tensor,
        prev_meas: tf.Tensor,
        prev_out: tf.Tensor,
        is_first: tf.Tensor,
    ) -> tf.Tensor:
        """Execute stochastic-mode behavior."""
        # First measurement: uniform 50/50 bits; p_rule is not used.
        # Second measurement: PR rule followed with probability p_rule; violated otherwise.
        pr_out_clean = self._pr_outcome_logits(prev_meas, curr_meas, prev_out)

        # Sample for first measurement.
        bits = self._sample_uniform_bits(tf.shape(curr_meas))
        first_logits = self._hard_logit_from_bit(bits)

        # Sample follow/violate mask for second measurement (Bernoulli(p_rule)).
        follow = self._runtime_follow_mask(tf.shape(curr_meas))
        second_logits = tf.where(follow, pr_out_clean, -pr_out_clean)
        return tf.where(is_first, first_logits, second_logits)

    def call(self, inputs: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """Compute outcome logits for either the first or second measurement.

        Args:
            inputs: Dictionary of tensors; see the class docstring for required keys.
            training: Keras training flag. Accepted for Keras compatibility; this
                layer's behavior is controlled by ``sr_mode`` and ``first_measurement``.

        Returns:
            ``tf.Tensor`` of outcome logits (float32) with shape ``(..., k)``.
        """
        del training  # accepted for Keras compatibility; semantics are controlled by SR mode and inputs

        if not isinstance(inputs, dict):
            raise TypeError(f"inputs must be a dict[str, tf.Tensor], got {type(inputs)!r}")

        curr_meas = self._ensure_float32(self._require_key(inputs, "current_measurement"), "current_measurement")
        prev_meas = self._ensure_float32(self._require_key(inputs, "previous_measurement"), "previous_measurement")
        prev_out = self._ensure_float32(self._require_key(inputs, "previous_outcome"), "previous_outcome")
        first_meas = self._ensure_float32(self._require_key(inputs, "first_measurement"), "first_measurement")

        # Basic shape compatibility checks. Keep these minimal to avoid over-constraining broadcasting.
        if curr_meas.shape.rank is not None and prev_meas.shape.rank is not None:
            if curr_meas.shape.rank != prev_meas.shape.rank:
                raise ValueError("current_measurement and previous_measurement must have the same rank")
        if curr_meas.shape.rank is not None and prev_out.shape.rank is not None:
            if curr_meas.shape.rank != prev_out.shape.rank:
                raise ValueError("current_measurement and previous_outcome must have the same rank")
        # first_measurement must be broadcastable to (..., 1). Check the last dim if statically known.
        if first_meas.shape.rank is not None and first_meas.shape[-1] is not None:
            if int(first_meas.shape[-1]) != 1:
                raise ValueError("first_measurement must have last dimension 1")

        # Determine first/second measurement per element (broadcastable).
        # Using >= 0.5 matches the contract wording for {0,1} inputs.
        is_first = first_meas >= 0.5

        return tf.cond(
            tf.equal(self._sr_mode_code, self._SR_MODE_REPLAY),
            lambda: self._call_replay_branch(curr_meas, prev_meas, prev_out, is_first, inputs),
            lambda: self._call_stochastic_branch(curr_meas, prev_meas, prev_out, is_first),
        )