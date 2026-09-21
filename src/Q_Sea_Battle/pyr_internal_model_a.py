"""
pyr_internal_model_a.py

Trainable Pyramid internal model "A" (scaled/bit-aligned I/O adapters live outside).

This module implements the Pyramid architecture used by QSeaBattle's internal
trainable player model. The internal representation is *logits-only*:

- The model consumes a flattened field tensor and treats values as logits
  (float32, unbounded). Any conversion to/from bits or the "scaled" domain
  (commonly values in {-0.5, +0.5}) is handled by external adapters.
- The model outputs communication logits (no hardening/thresholding here).
  Logical bit values represented as logits are determined by logit sign.
- When used with PR-assisted shared resource (SR) in replay/training mode, the
  SR output logits can be teacher-forced. By construction, the model can be run
  so that out_a_logits_list[d] matches replay_out_a_logits_list[d] at each level.

Architecture per Pyramid level:
1) PyrMeasurementLayerA: field logits -> measurement logits
2) PRAssistedReplay: PR-assisted SR over measurement/outcome logits
3) PyrCombineLayerA: (field logits, outcome logits) -> next field logits

SR modes:
- "replay": deterministic replay behavior (per PRAssistedReplay implementation)
- "stochastic": stochastic follow behavior controlled by p_rule
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

import tensorflow as tf

from .pyr_measurement_layer_a import PyrMeasurementLayerA
from .pyr_combine_layer_a import PyrCombineLayerA
from .pr_assisted_replay import PRAssistedReplay


def _infer_n2_and_m(game_layout: Any) -> tuple[int, int]:
    """Infer ``(n2, m)`` from a GameLayout-like object.

    The codebase uses different layout objects across modules/tests. This helper
    supports common attribute names without importing a concrete type.

    Args:
        game_layout: Object that exposes either:
            - ``n2`` (flattened field length), or
            - ``field_size`` (board edge length; n2 = field_size ** 2),
            and either:
            - ``comms_size``, or
            - ``M`` (legacy).

    Returns:
        Tuple ``(n2, m)`` where:
            - n2 is the flattened board size.
            - m is the communication size.

    TODO(review): Consolidate layout attribute naming across the project.
    """
    if hasattr(game_layout, "n2"):
        n2 = int(getattr(game_layout, "n2"))
    else:
        field_size = int(getattr(game_layout, "field_size"))
        n2 = field_size * field_size
    m = int(getattr(game_layout, "comms_size", getattr(game_layout, "M", 1)))
    return n2, m


def _validate_power_of_two(n: int) -> int:
    """Validate that ``n`` is a power of two and return ``log2(n)``.

    Args:
        n: Positive integer.

    Returns:
        Integer ``k`` such that ``2**k == n``.

    Raises:
        ValueError: If ``n <= 0`` or ``n`` is not a power of two.
    """
    if n <= 0:
        raise ValueError("n2 must be positive.")
    k = int(round(math.log2(n)))
    if 2**k != n:
        raise ValueError(f"n2 must be a power of 2; got n2={n}.")
    return k


class PyrInternalModelA(tf.keras.Model):
    """Pyramid internal model A (logits-in, logits-out).

    This model is "contract-aligned" with the trainable-player interface:

    - Input is treated as field logits with shape ``(B, n2)``.
    - Output is communication logits with shape ``(B, 1)`` for Pyramid layouts.
    - The internal per-level tensors (measurements and SR outcomes) are kept as
      logits to support loss computation and teacher forcing.

    Attributes:
        n2: Flattened field length.
        M: Communication size (must be 1 for this Pyramid architecture).
        depth: Number of Pyramid levels (``log2(n2)``).
        measure_layers: Per-level measurement layers.
        combine_layers: Per-level combine layers.
        sr_layers: Per-level PR-assisted SR layers (PRAssistedReplay).
    """

    def __init__(
        self,
        game_layout: Any,
        *,
        sr_mode: str = "replay",        # {"replay","stochastic"}
        p_rule: float = 1.0,            # used only in stochastic mode inside PRAssistedReplay (second measurement)
        beta: float = 10.0,
        alpha: float = 5.0,
        seed: int | None = None,
        measure_layers: Optional[Sequence[tf.keras.layers.Layer]] = None,
        combine_layers: Optional[Sequence[tf.keras.layers.Layer]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the Pyramid internal model.

        Args:
            game_layout: GameLayout-like object used to infer ``n2`` and ``M``.
            sr_mode: Shared resource mode for each PRAssistedReplay layer.
                Expected values are {"replay", "stochastic"}.
            p_rule: Follow probability used by PRAssistedReplay in stochastic SR mode.
            beta: Logit magnitude used internally by PRAssistedReplay (see that class).
            alpha: PR gate sharpness used internally by PRAssistedReplay (see that class).
            seed: Optional seed forwarded to PRAssistedReplay layers.
            measure_layers: Optional explicit per-level measurement layers. If provided,
                must have length ``depth``.
            combine_layers: Optional explicit per-level combine layers. If provided,
                must have length ``depth``.
            name: Optional Keras model name.

        Raises:
            ValueError: If ``comms_size != 1`` or provided layer lists do not match
                the inferred Pyramid depth.
        """
        super().__init__(name=name)
        self.n2, self.M = _infer_n2_and_m(game_layout)
        if self.M != 1:
            raise ValueError(f"Pyr architecture requires comms_size==1; got m={self.M}.")
        self.depth = _validate_power_of_two(self.n2)

        # Per-level layers
        if measure_layers is None:
            self.measure_layers: List[tf.keras.layers.Layer] = [PyrMeasurementLayerA() for _ in range(self.depth)]
        else:
            if len(measure_layers) != self.depth:
                raise ValueError(f"measure_layers must have length depth={self.depth}; got {len(measure_layers)}.")
            self.measure_layers = list(measure_layers)

        if combine_layers is None:
            self.combine_layers: List[tf.keras.layers.Layer] = [PyrCombineLayerA() for _ in range(self.depth)]
        else:
            if len(combine_layers) != self.depth:
                raise ValueError(f"combine_layers must have length depth={self.depth}; got {len(combine_layers)}.")
            self.combine_layers = list(combine_layers)

        # Backward-compat aliases
        self.measure_layer = self.measure_layers[0]
        self.combine_layer = self.combine_layers[0]

        # SR layers (one per level), logits-only.
        self.sr_layers: List[PRAssistedReplay] = []
        active = self.n2
        for level in range(self.depth):
            k = active // 2
            self.sr_layers.append(
                PRAssistedReplay(
                    sr_mode=sr_mode,
                    p_rule=p_rule,
                    beta=beta,
                    alpha=alpha,
                    seed=seed,
                    name=f"pr_replay_a_{level}",
                )
            )
            active //= 2

    def set_alpha(self, alpha: float) -> None:
        """Set PR gate sharpness for all SR layers.

        This is a runtime-safe convenience wrapper around PRAssistedReplay.

        Args:
            alpha: New alpha value forwarded to each SR layer.

        Raises:
            AttributeError: If a configured SR layer does not implement
                ``set_alpha`` (indicates a version mismatch).
        """
        for sr in self.sr_layers:
            if hasattr(sr, "set_alpha"):
                sr.set_alpha(alpha)
            else:
                raise AttributeError("SR layer has no set_alpha(); update PRAssistedReplay first.")

    def set_p_rule(self, p_rule: float) -> None:
        """Set stochastic follow probability for all SR layers.

        Args:
            p_rule: New follow probability forwarded to each SR layer.

        Raises:
            AttributeError: If a configured SR layer does not implement
                ``set_p_rule`` (indicates a version mismatch).
        """
        for sr in self.sr_layers:
            if hasattr(sr, "set_p_rule"):
                sr.set_p_rule(p_rule)
            else:
                raise AttributeError("SR layer has no set_p_rule(); update PRAssistedReplay first.")

    def set_beta(self, beta: float) -> None:
        """Set hard-logit beta for all SR layers.

        Args:
            beta: New beta value forwarded to each SR layer.

        Raises:
            AttributeError: If a configured SR layer does not implement
                ``set_beta`` (indicates a version mismatch).
        """
        for sr in self.sr_layers:
            if hasattr(sr, "set_beta"):
                sr.set_beta(beta)
            else:
                raise AttributeError("SR layer has no set_beta(); update PRAssistedReplay first.")

    def set_sr_mode(self, sr_mode: str) -> None:
        """Set SR mode for all SR layers.

        Args:
            sr_mode: SR mode forwarded to each SR layer (e.g., "replay" or
                "stochastic").

        Raises:
            AttributeError: If a configured SR layer does not implement
                ``set_sr_mode`` (indicates a version mismatch).
        """
        for sr in self.sr_layers:
            if hasattr(sr, "set_sr_mode"):
                sr.set_sr_mode(sr_mode)
            else:
                raise AttributeError("SR layer has no set_sr_mode(); update PRAssistedReplay first.")

    def call(self, field_scaled: tf.Tensor, training: bool = False, **kwargs: Any) -> tf.Tensor:
        """Keras forward pass returning only the final communication logits.

        Note:
            The argument name ``field_scaled`` is retained for API compatibility.
            The implementation treats the input as float32 logits with shape
            ``(B, n2)``; any scaling/bit conversions occur outside this model.

        Args:
            field_scaled: Tensor interpreted as field logits.
            training: Forwarded to sublayers where supported.
            **kwargs: Unused; accepted for Keras compatibility.

        Returns:
            Communication logits tensor with shape ``(B, 1)`` for Pyramid layouts.
        """
        comm_logits, _, _ = self.compute_with_internal(field_scaled, replay_out_a_logits_list=None, training=training)
        return comm_logits

    def compute_with_internal(
        self,
        field_logits: tf.Tensor,
        replay_out_a_logits_list: Optional[Sequence[tf.Tensor]] = None,
        harden_between_levels: bool = False,
        beta_for_hardening: float = 10.0,
        training: bool = False,
    ) -> Tuple[tf.Tensor, List[tf.Tensor], List[tf.Tensor]]:
        """Compute a full forward pass and return per-level intermediate logits.

        All internal tensors are logits (float32). The per-level routine is:

        1) Measurement: ``meas_layer(state_logits) -> meas_logits``
        2) PR-assisted SR: ``PRAssistedReplay(inputs) -> out_logits``
           where SR inputs include:
             - current_measurement: meas_logits
             - previous_measurement: zeros (Model A uses no previous step here)
             - previous_outcome: zeros
             - first_measurement: ones (flag tensor of shape (B, 1))
           and may include:
             - replay_outcome_logits (teacher forcing)
        3) Combine: ``comb_layer(state_logits, out_logits) -> next_state_logits``

        Teacher forcing:
            If ``replay_out_a_logits_list`` is provided, element ``[d]`` is passed
            into the SR layer at depth ``d`` as ``replay_outcome_logits``. This
            enables deterministic per-level outcomes for supervised training,
            while still producing measurement and combined logits for losses.

        Args:
            field_logits: Field logits tensor of shape ``(B, n2)``.
            replay_out_a_logits_list: Optional sequence of length ``self.depth``.
                Element ``d`` must be broadcast-compatible with the SR output at
                that level (typically shape ``(B, k_d)``).
            harden_between_levels: If True, harden the *state logits* between
                levels to +/- ``beta_for_hardening`` based on sign. This is
                primarily a debugging/ablation option; it is not part of the
                standard logits-only contract.
            beta_for_hardening: Logit magnitude used when hardening is enabled.
            training: Forwarded to sublayers where supported.

        Returns:
            Tuple ``(comm_logits, meas_list, out_list)``:
              - comm_logits: Final communication logits (for Pyramid, shape (B, 1)).
              - meas_list: Python list of per-level measurement logits.
              - out_list: Python list of per-level SR outcome logits.

        Raises:
            ValueError: If ``field_logits`` rank/width does not match ``n2``, or
                if ``replay_out_a_logits_list`` length does not match ``depth``.
            TypeError: If ``replay_out_a_logits_list`` is provided but is not a
                list/tuple.
        """
        # ---- Input validation (backward compatible but explicit) ----
        x = tf.convert_to_tensor(field_logits, dtype=tf.float32)
        if x.shape.rank != 2:
            raise ValueError(f"field_logits must be rank-2 (B,n2); got {x.shape}.")
        if x.shape[-1] is not None and int(x.shape[-1]) != self.n2:
            raise ValueError(f"field_logits last dimension must be n2={self.n2}; got {x.shape[-1]}.")

        if replay_out_a_logits_list is not None:
            if not isinstance(replay_out_a_logits_list, (list, tuple)):
                raise TypeError("replay_out_a_logits_list must be a Python list/tuple of tensors or None.")
            if len(replay_out_a_logits_list) != self.depth:
                raise ValueError(
                    f"replay_out_a_logits_list must have length depth={self.depth}; got {len(replay_out_a_logits_list)}."
                )

        meas_list: List[tf.Tensor] = []
        out_list: List[tf.Tensor] = []

        state_logits = x
        last_field_logits: tf.Tensor | None = None

        # ---- Per-level forward pass ----
        for level in range(self.depth):
            meas_layer = self.measure_layers[level]
            comb_layer = self.combine_layers[level]
            sr = self.sr_layers[level]

            # Performance note: this loop is hot; keep Python-side branching minimal.
            # Optional hardening replaces logits with +/-beta based on sign.
            def harden_logits(logits, beta):
                logits = tf.cast(logits, tf.float32)
                return tf.where(logits >= 0.0, beta, -beta)

            if harden_between_levels:
                state_logits = harden_logits(state_logits, beta_for_hardening)

            # 1) Measurement (logits)
            # Keep try/except for compatibility with layers that do not accept `training=...`.
            try:
                meas_logits = tf.cast(meas_layer(state_logits, training=training), tf.float32)
            except TypeError:
                meas_logits = tf.cast(meas_layer(state_logits), tf.float32)

            # 2) PR-assisted SR
            # Model A uses "first measurement" semantics at every level (no previous step).
            zeros = tf.zeros_like(meas_logits)
            first_flag = tf.ones((tf.shape(meas_logits)[0], 1), dtype=tf.float32)

            sr_inputs = {
                "current_measurement": meas_logits,
                "previous_measurement": zeros,
                "previous_outcome": zeros,
                "first_measurement": first_flag,
            }

            # Optional teacher forcing: provide replay outcome logits (must match measurement width).
            if replay_out_a_logits_list is not None:
                replay_logits = tf.cast(tf.convert_to_tensor(replay_out_a_logits_list[level]), tf.float32)
                tf.debugging.assert_equal(
                    tf.shape(replay_logits)[-1],
                    tf.shape(meas_logits)[-1],
                    message=f"Replay outcome length mismatch at level {level}.",
                )
                sr_inputs["replay_outcome_logits"] = replay_logits

            out_logits = tf.cast(sr(sr_inputs, training=training), tf.float32)

            # 3) Combine A: (state_logits, out_logits) -> next field logits
            try:
                next_field_logits = tf.cast(comb_layer(state_logits, out_logits, training=training), tf.float32)
            except TypeError:
                next_field_logits = tf.cast(comb_layer(state_logits, out_logits), tf.float32)

            meas_list.append(meas_logits)
            out_list.append(out_logits)

            # Pure-logit feed-forward to the next Pyramid level.
            state_logits = next_field_logits
            last_field_logits = next_field_logits

        if last_field_logits is None:
            raise RuntimeError("Internal error: model depth produced no outputs.")

        # For Pyramid layouts, the final field logits are interpreted as comm logits (shape (B, 1)).
        comm_logits = tf.cast(last_field_logits, tf.float32)
        return comm_logits, meas_list, out_list

    # -------------------------------------------------
    # Weight utilities
    # -------------------------------------------------

    def _ensure_built(self) -> None:
        """Ensure variables are created before saving/loading weights.

        Keras variables may not exist until the first forward call. This helper
        runs a minimal forward pass (including SR teacher forcing tensors) to
        create all variables deterministically before weight IO.
        """
        if not self.built:
            dummy = tf.zeros((1, self.n2), dtype=tf.float32) - 0.5
            dummy_replay_out_a_logits_list = [
                tf.zeros((1, self.n2 // (2 ** (d + 1))), dtype=tf.float32) for d in range(self.depth)
            ]
            _ = self.compute_with_internal(dummy, replay_out_a_logits_list=dummy_replay_out_a_logits_list, training=False)
            self.built = True

    def save_weights_to(self, path: str) -> None:
        """Save model weights to a file.

        Args:
            path: Destination filepath understood by ``tf.keras.Model.save_weights``.
        """
        self._ensure_built()
        super().save_weights(path)

    def load_weights_from(self, path: str) -> None:
        """Load model weights from a file.

        Args:
            path: Source filepath understood by ``tf.keras.Model.load_weights``.
        """
        self._ensure_built()
        super().load_weights(path)