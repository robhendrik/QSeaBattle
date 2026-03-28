"""PRAssistedReplay (Keras layer; logits in, logits out).

This module implements the *replay* shared-resource (SR) layer described in the frozen markdown specs.

Key properties (normative; see spec/contract):
- Stateless w.r.t. ordering: callers provide ``first_measurement``.
- Single-output API: each call produces exactly one outcome tensor (logits).
- Logits-only I/O: no sigmoid/threshold/binarization occurs inside this layer.
- Two SR modes:
  * ``sr_mode="replay"``: training mode. First measurement is prescribed by ``replay_outcome_logits`` (identity).
    Second measurement follows the PR rule with probability ``p_high`` and violates it (exact opposite) otherwise.
  * ``sr_mode="stochastic"``: gameplay mode. First measurement is uniform random (50/50) and independent of ``p_high``.
    Second measurement follows the PR rule with probability ``p_high`` and violates it (exact opposite) otherwise.

Differentiability requirement:
- In both modes mode, the second-measurement PR gate MUST be differentiable w.r.t. both measurement logits.
  Therefore, hard thresholding (e.g. ``logit >= 0``) is forbidden in replay mode and replaced by a soft-high gate.

Author: Rob Hendriks (project)
Package: Q_Sea_Battle_New (WIP)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import tensorflow as tf


@dataclass(frozen=True)
class PRAssistedReplayConfig:
    """Configuration for :class:`PRAssistedReplay`.
    NOTE: runtime execution may use internal tf.Variable state for selected fields.
    """
    sr_mode: str = "replay"  # {"replay", "stochastic"}
    p_high: float = 1.0      # used ONLY for stochastic second measurement / forward noise
    beta: float = 10.0       # hard_logit mapping: {0,1} -> {-beta,+beta}
    alpha: float = 5.0       # soft-high sharpness/temperature for replay second measurement
    seed: Optional[int] = None  # RNG seed (only used for stochastic sampling)


class PRAssistedReplay(tf.keras.layers.Layer):
    """PR-assisted shared-resource layer (stateless ordering; logits-only).

    Contract summary (normative):
    - Inputs are provided as a dict with the following keys:
        * ``current_measurement``: ``tf.Tensor`` float32 shape ``(..., k)``, logits
        * ``previous_measurement``: ``tf.Tensor`` float32 shape ``(..., k)``, logits
        * ``previous_outcome``: ``tf.Tensor`` float32 shape ``(..., k)``, logits
        * ``first_measurement``: ``tf.Tensor`` float32 shape ``(..., 1)`` (broadcastable), values in {0,1}
        * ``replay_outcome_logits`` (optional): logits, required iff sr_mode="replay" and first_measurement==1
    - Output:
        * ``outcome_logits``: ``tf.Tensor`` float32 shape ``(..., k)``, logits

    Notes for maintainers:
    - "Opposite" in logits space is implemented as negation: ``-logits``.
    - The replay-mode PR gate uses a differentiable soft-high mapping:
        p_high(x) = sigmoid(alpha * x), p_flip = p_high(prev) * p_high(curr),
        pr_outcome_logits = (1 - 2*p_flip) * previous_outcome.
    """

    _SR_MODE_REPLAY = 0
    _SR_MODE_STOCHASTIC = 1

    def __init__(
        self,
        sr_mode: str = "replay",
        *,
        p_high: float = 1.0,
        beta: float = 10.0,
        alpha: float = 5.0,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._validate_sr_mode(sr_mode)
        self._validate_p_high(p_high)
        self._validate_beta(beta)
        self._validate_alpha(alpha)

        self._cfg = PRAssistedReplayConfig(
            sr_mode=sr_mode,
            p_high=float(p_high),
            beta=float(beta),
            alpha=float(alpha),
            seed=seed,
        )

        # Runtime-settable, TF-friendly knobs. These drive execution.
        self._alpha = tf.Variable(float(alpha), dtype=tf.float32, trainable=False, name="alpha")
        self._p_high = tf.Variable(float(p_high), dtype=tf.float32, trainable=False, name="p_high")
        self._beta = tf.Variable(float(beta), dtype=tf.float32, trainable=False, name="beta")
        self._sr_mode_code = tf.Variable(
            self._encode_sr_mode(sr_mode),
            dtype=tf.int32,
            trainable=False,
            name="sr_mode_code",
        )

        # Use a dedicated Generator so stochastic behavior is reproducible when a seed is provided.
        # We keep this construction unchanged for backward compatibility.
        if seed is None:
            self._rng = tf.random.Generator.from_non_deterministic_state()
        else:
            self._rng = tf.random.Generator.from_seed(int(seed))

    @property
    def config(self) -> PRAssistedReplayConfig:
        """Read-only configuration."""
        return self._cfg

    def get_config(self) -> dict[str, Any]:
        """Keras serialization support."""
        base = super().get_config()
        base.update(
            {
                "sr_mode": self._cfg.sr_mode,
                "p_high": self._cfg.p_high,
                "beta": self._cfg.beta,
                "alpha": self._cfg.alpha,
                "seed": self._cfg.seed,
            }
        )
        return base

    @staticmethod
    def _validate_sr_mode(sr_mode: str) -> None:
        if sr_mode not in {"replay", "stochastic"}:
            raise ValueError(f"sr_mode must be one of {{'replay','stochastic'}}, got {sr_mode!r}")

    @staticmethod
    def _validate_p_high(p_high: float) -> None:
        if not (0.0 <= float(p_high) <= 1.0):
            raise ValueError(f"p_high must be in [0,1], got {p_high!r}")

    @staticmethod
    def _validate_beta(beta: float) -> None:
        if float(beta) <= 0.0:
            raise ValueError(f"beta must be > 0, got {beta!r}")

    @staticmethod
    def _validate_alpha(alpha: float) -> None:
        if float(alpha) <= 0.0:
            raise ValueError(f"alpha must be > 0, got {alpha!r}")

    @classmethod
    def _encode_sr_mode(cls, sr_mode: str) -> int:
        cls._validate_sr_mode(sr_mode)
        return cls._SR_MODE_REPLAY if sr_mode == "replay" else cls._SR_MODE_STOCHASTIC

    def set_alpha(self, alpha: float) -> None:
        """Update alpha at runtime."""
        a = float(alpha)
        self._validate_alpha(a)
        self._alpha.assign(a)

    def set_p_high(self, p_high: float) -> None:
        """Update p_high at runtime."""
        p = float(p_high)
        self._validate_p_high(p)
        self._p_high.assign(p)

    def set_beta(self, beta: float) -> None:
        """Update beta at runtime."""
        b = float(beta)
        self._validate_beta(b)
        self._beta.assign(b)

    def set_sr_mode(self, sr_mode: str) -> None:
        """Update sr_mode at runtime."""
        code = self._encode_sr_mode(sr_mode)
        self._sr_mode_code.assign(code)

    @staticmethod
    def _require_key(inputs: dict[str, tf.Tensor], key: str) -> tf.Tensor:
        if key not in inputs:
            raise ValueError(f"Missing required input key: {key!r}")
        return inputs[key]

    @staticmethod
    def _ensure_float32(x: tf.Tensor, name: str) -> tf.Tensor:
        x = tf.convert_to_tensor(x)
        if x.dtype != tf.float32:
            x = tf.cast(x, tf.float32, name=f"{name}_float32")
        return x

    def _hard_logit_from_bit(self, bit01: tf.Tensor) -> tf.Tensor:
        """Map {0,1} float tensor to {-beta,+beta} logits."""
        bit01 = tf.cast(bit01, tf.float32)
        return (2.0 * bit01 - 1.0) * self._beta

    def _sample_uniform_bits(self, shape: tf.Tensor) -> tf.Tensor:
        """Sample uniform Bernoulli(0.5) bits with the internal generator."""
        u = self._rng.uniform(shape=shape, minval=0.0, maxval=1.0, dtype=tf.float32)
        return tf.cast(u < 0.5, tf.float32)

    def _pr_outcome_logits(self, prev_meas: tf.Tensor, curr_meas: tf.Tensor, prev_out: tf.Tensor) -> tf.Tensor:
        """Deterministic PR-rule outcome logits using a differentiable soft gate.

        Parameter ``self._alpha`` controls the sharpness/temperature of the soft-high mapping
        and can be updated at runtime with ``set_alpha()``.
        """
        # Soft-high mapping: p_high(x)=sigmoid(alpha*x)
        p_prev = tf.sigmoid(self._alpha * prev_meas)
        p_curr = tf.sigmoid(self._alpha * curr_meas)

        # Soft AND for (high,high).
        p_flip = p_prev * p_curr

        # Interpolated sign flip: (1 - 2*p_flip) in [+1,-1]
        return (1.0 - 2.0 * p_flip) * prev_out

    def _runtime_follow_mask(self, shape: tf.Tensor) -> tf.Tensor:
        """Sample Bernoulli(p_high) follow mask using the runtime variable."""
        u = self._rng.uniform(shape=shape, minval=0.0, maxval=1.0, dtype=tf.float32)
        return u < self._p_high

    def _call_replay_branch(
        self,
        curr_meas: tf.Tensor,
        prev_meas: tf.Tensor,
        prev_out: tf.Tensor,
        is_first: tf.Tensor,
        inputs: dict[str, tf.Tensor],
    ) -> tf.Tensor:
        """Runtime replay branch."""
        if "replay_outcome_logits" not in inputs:
            # Strict validation: if any element in the batch indicates first measurement, the prescribed
            # replay logits MUST be provided. We use a TF assertion so this also works under tf.function.
            if tf.executing_eagerly():
                # Eager mode (unit tests): raise a Python exception so pytest can catch it naturally.
                if bool(tf.reduce_any(is_first).numpy()):
                    raise ValueError("replay_outcome_logits is required for replay-mode first measurement")
            else:
                # Graph mode: keep a TF assertion (raises InvalidArgumentError) for correctness under tf.function.
                tf.debugging.assert_equal(
                    tf.reduce_any(is_first),
                    False,
                    message="replay_outcome_logits is required for replay-mode first measurement",
                )

        replay_out = inputs.get("replay_outcome_logits", None)
        if replay_out is not None:
            replay_out = self._ensure_float32(replay_out, "replay_outcome_logits")

        pr_out_clean = self._pr_outcome_logits(prev_meas, curr_meas, prev_out)

        # Forward: apply p_high noise from runtime variable.
        follow = self._runtime_follow_mask(tf.shape(curr_meas))
        pr_out_noisy = tf.where(follow, pr_out_clean, -pr_out_clean)

        # Keep current forward behavior unchanged.
        pr_out = pr_out_noisy  # pr_out_clean + tf.stop_gradient(pr_out_noisy - pr_out_clean)

        if replay_out is None:
            # No first measurement allowed if replay_out is absent (checked above).
            return pr_out

        # Combine per-element: if first -> replay_out, else -> pr_out.
        # Broadcasting: is_first (...,1) broadcasts to (...,k)
        return tf.where(is_first, replay_out, pr_out)

    def _call_stochastic_branch(
        self,
        curr_meas: tf.Tensor,
        prev_meas: tf.Tensor,
        prev_out: tf.Tensor,
        is_first: tf.Tensor,
    ) -> tf.Tensor:
        """Runtime stochastic branch."""
        # First measurement: uniform 50/50 bits; p_high unused.
        # Second measurement: PR rule followed with probability p_high; violated otherwise.
        pr_out_clean = self._pr_outcome_logits(prev_meas, curr_meas, prev_out)

        # Sample for first measurement
        bits = self._sample_uniform_bits(tf.shape(curr_meas))
        first_logits = self._hard_logit_from_bit(bits)

        # Sample follow/violate mask for second measurement (Bernoulli(p_high)).
        follow = self._runtime_follow_mask(tf.shape(curr_meas))
        second_logits = tf.where(follow, pr_out_clean, -pr_out_clean)
        return tf.where(is_first, first_logits, second_logits)

    def call(self, inputs: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """Compute outcome logits for either the first or second measurement.

        Parameters
        ----------
        inputs:
            Dictionary of tensors; see class docstring for required keys.
        training:
            Keras training flag. This layer's behavior is controlled by ``sr_mode`` and the provided inputs;
            ``training`` does not change semantics directly, but is accepted for Keras compatibility.

        Returns
        -------
        tf.Tensor
            ``outcome_logits`` float32, shape ``(..., k)``, logits.
        """
        del training  # accepted for Keras compatibility; semantics are controlled by SR mode and inputs

        if not isinstance(inputs, dict):
            raise TypeError(f"inputs must be a dict[str, tf.Tensor], got {type(inputs)!r}")

        curr_meas = self._ensure_float32(self._require_key(inputs, "current_measurement"), "current_measurement")
        prev_meas = self._ensure_float32(self._require_key(inputs, "previous_measurement"), "previous_measurement")
        prev_out = self._ensure_float32(self._require_key(inputs, "previous_outcome"), "previous_outcome")
        first_meas = self._ensure_float32(self._require_key(inputs, "first_measurement"), "first_measurement")

        # Basic shape compatibility checks. We keep these minimal to avoid over-constraining broadcasting.
        if curr_meas.shape.rank is not None and prev_meas.shape.rank is not None:
            if curr_meas.shape.rank != prev_meas.shape.rank:
                raise ValueError("current_measurement and previous_measurement must have the same rank")
        if curr_meas.shape.rank is not None and prev_out.shape.rank is not None:
            if curr_meas.shape.rank != prev_out.shape.rank:
                raise ValueError("current_measurement and previous_outcome must have the same rank")
        # first_measurement must be broadcastable to (..., 1). We check the last dim if static.
        if first_meas.shape.rank is not None and first_meas.shape[-1] is not None:
            if int(first_meas.shape[-1]) != 1:
                raise ValueError("first_measurement must have last dimension 1")

        # Determine first/second measurement per element (broadcastable).
        # Using >=0.5 matches the contract wording.
        is_first = first_meas >= 0.5

        return tf.cond(
            tf.equal(self._sr_mode_code, self._SR_MODE_REPLAY),
            lambda: self._call_replay_branch(curr_meas, prev_meas, prev_out, is_first, inputs),
            lambda: self._call_stochastic_branch(curr_meas, prev_meas, prev_out, is_first),
        )