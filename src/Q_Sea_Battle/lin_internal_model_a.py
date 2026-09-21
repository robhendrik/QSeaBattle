"""Internal trainable Linear Model A (logits-only, contract-aligned).

This module defines a depth-1, trainable internal model used by QSeaBattle.
It consumes *field logits* and produces *communication logits* (comms). All
logical bits are represented as logits, where the bit value is determined by
the logit sign.

Architecture (fixed depth = 1):
1) Measurement layer: maps field logits -> measurement logits.
2) PR-assisted shared resource (SR): produces outcome logits from measurement
   logits, optionally conditioned on a provided replay outcome.
3) Combine layer: maps SR outcome logits -> communication logits.

Contract:
- ``depth`` is fixed to 1.
- Input field logits: float tensor of shape (B, n2).
- Output comm logits: float tensor of shape (B, m), where m >= 1.
- ``compute_with_internal`` returns ``(comm_logits, meas_list, out_list)``,
  where ``meas_list`` and ``out_list`` are length-1 lists containing the
  per-level tensors.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import tensorflow as tf

from .lin_measurement_layer_a import LinMeasurementLayerA
from .lin_combine_layer_a import LinCombineLayerA
from .pr_assisted_replay import PRAssistedReplay
from .pyr_internal_model_a import _infer_n2_and_m


class LinInternalModelA(tf.keras.Model):
    """Depth-1 linear internal Model A (logits-in, logits-out).

    The model is "contract-aligned" in the sense that it exposes the same
    internal tensors (measurement and SR outcome) as deeper internal models,
    but with depth fixed to 1.

    Attributes:
        n2: Number of field bits (flattened field size), inferred from
            ``game_layout``.
        M: Number of communication bits (comms size), inferred from
            ``game_layout``. Must be >= 1.
        depth: Fixed to 1.
        measure_layers: List of measurement layers; length is always 1.
        combine_layers: List of combine layers; length is always 1.
        sr_layers: List of PR-assisted SR layers; length is always 1.
    """

    def __init__(
        self,
        game_layout: Any,
        *,
        sr_mode: str = "replay",
        p_rule: float = 1.0,
        beta: float = 10.0,
        alpha: float = 5.0,
        seed: int | None = None,
        measure_layers: Optional[Sequence[tf.keras.layers.Layer]] = None,
        combine_layers: Optional[Sequence[tf.keras.layers.Layer]] = None,
        hidden_units_measure: int = 64,
        hidden_units_combine: int = 64,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the model.

        Args:
            game_layout: Game layout/config object used to infer ``n2`` and
                comms size ``M``.
            sr_mode: Shared resource mode; expected values are those supported
                by :class:`PRAssistedReplay` (e.g., ``"replay"`` or
                ``"stochastic"``).
            p_rule: Probability of using the rule-based component in the
                PR-assisted SR.
            beta: Temperature/sharpness parameter used by the PR-assisted SR.
            alpha: Mixing/strength parameter used by the PR-assisted SR.
            seed: Optional seed forwarded to the SR layer.
            measure_layers: Optional custom measurement layer sequence. If
                provided, must have length 1.
            combine_layers: Optional custom combine layer sequence. If
                provided, must have length 1.
            hidden_units_measure: Hidden units used when constructing the
                default measurement layer.
            hidden_units_combine: Hidden units used when constructing the
                default combine layer.
            name: Optional Keras model name.

        Raises:
            ValueError: If the inferred comms size is < 1, or if a provided
                layer list does not match the fixed depth.
        """
        super().__init__(name=name)
        self.n2, self.M = _infer_n2_and_m(game_layout)
        if self.M < 1:
            raise ValueError(f"Linear architecture requires comms_size>=1; got m={self.M}.")
        self.depth = 1

        if measure_layers is None:
            self.measure_layers: List[tf.keras.layers.Layer] = [
                LinMeasurementLayerA(hidden_units=hidden_units_measure)
            ]
        else:
            if len(measure_layers) != self.depth:
                raise ValueError(f"measure_layers must have length depth={self.depth}; got {len(measure_layers)}.")
            self.measure_layers = list(measure_layers)

        if combine_layers is None:
            self.combine_layers: List[tf.keras.layers.Layer] = [
                LinCombineLayerA(comms_size=self.M, hidden_units=hidden_units_combine)
            ]
        else:
            if len(combine_layers) != self.depth:
                raise ValueError(f"combine_layers must have length depth={self.depth}; got {len(combine_layers)}.")
            self.combine_layers = list(combine_layers)

        # Convenience aliases for depth=1 usage.
        self.measure_layer = self.measure_layers[0]
        self.combine_layer = self.combine_layers[0]

        self.sr_layers: List[PRAssistedReplay] = [
            PRAssistedReplay(
                sr_mode=sr_mode,
                p_rule=p_rule,
                beta=beta,
                alpha=alpha,
                seed=seed,
                name="pr_replay_a_0",
            )
        ]
        self.sr_layer = self.sr_layers[0]

    def set_alpha(self, alpha: float) -> None:
        """Set the PR-assisted SR ``alpha`` parameter for all SR layers."""
        for sr in self.sr_layers:
            if hasattr(sr, "set_alpha"):
                sr.set_alpha(alpha)
            else:
                raise AttributeError("SR layer has no set_alpha(); update PRAssistedReplay first.")

    def set_p_rule(self, p_rule: float) -> None:
        """Set the PR-assisted SR ``p_rule`` parameter for all SR layers."""
        for sr in self.sr_layers:
            if hasattr(sr, "set_p_rule"):
                sr.set_p_rule(p_rule)
            else:
                raise AttributeError("SR layer has no set_p_rule(); update PRAssistedReplay first.")

    def set_beta(self, beta: float) -> None:
        """Set the PR-assisted SR ``beta`` parameter for all SR layers."""
        for sr in self.sr_layers:
            if hasattr(sr, "set_beta"):
                sr.set_beta(beta)
            else:
                raise AttributeError("SR layer has no set_beta(); update PRAssistedReplay first.")

    def set_sr_mode(self, sr_mode: str) -> None:
        """Set the SR mode (e.g., ``replay`` or ``stochastic``) for all SR layers."""
        for sr in self.sr_layers:
            if hasattr(sr, "set_sr_mode"):
                sr.set_sr_mode(sr_mode)
            else:
                raise AttributeError("SR layer has no set_sr_mode(); update PRAssistedReplay first.")

    def call(self, field_scaled: tf.Tensor, training: bool = False, **kwargs: Any) -> tf.Tensor:
        """Keras forward pass.

        Args:
            field_scaled: Field logits, shaped (B, n2). The name reflects
                upstream code; this model treats the input as logits.
            training: Keras training flag.
            **kwargs: Unused extra Keras call kwargs.

        Returns:
            Communication logits of shape (B, M).
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
        """Compute comm logits and expose internal tensors.

        This method implements the contract-required interface used by training
        and evaluation code that expects per-level measurement and outcome
        tensors. Since this model is linear with fixed depth=1, both returned
        lists have length 1.

        Args:
            field_logits: Field logits tensor of shape (B, n2), float-compatible.
            replay_out_a_logits_list: Optional replay outcomes for SR in replay
                mode. If provided, must be a list/tuple of length 1 whose only
                element has trailing dimension equal to the measurement size.
            harden_between_levels: If True, replace logits by hard +/- beta
                based on sign. Included for API compatibility with deeper
                models; for depth=1 this hardens the input before measurement.
            beta_for_hardening: Magnitude used when hardening logits.
            training: Keras training flag.

        Returns:
            A tuple ``(comm_logits, meas_list, out_list)``:
            - comm_logits: float32 tensor of shape (B, M).
            - meas_list: length-1 list containing measurement logits.
            - out_list: length-1 list containing SR outcome logits.

        Raises:
            ValueError: If ``field_logits`` is not rank-2 or has mismatched
                trailing dimension, or if ``replay_out_a_logits_list`` has
                wrong length.
            TypeError: If ``replay_out_a_logits_list`` is not a list/tuple when
                provided.
        """
        x = tf.convert_to_tensor(field_logits, dtype=tf.float32)
        if x.shape.rank != 2:
            raise ValueError(f"field_logits must be rank-2 (B,n2); got {x.shape}.")
        if x.shape[-1] is not None and int(x.shape[-1]) != self.n2:
            raise ValueError(f"field_logits last dimension must be n2={self.n2}; got {x.shape[-1]}.")

        if replay_out_a_logits_list is not None:
            if not isinstance(replay_out_a_logits_list, (list, tuple)):
                raise TypeError("replay_out_a_logits_list must be a list/tuple of tensors or None.")
            if len(replay_out_a_logits_list) != self.depth:
                raise ValueError(
                    f"replay_out_a_logits_list must have length depth={self.depth}; got {len(replay_out_a_logits_list)}."
                )

        if harden_between_levels:
            # Convert logits to a hard sign-representation, preserving the
            # logit convention: positive => logical 1, negative => logical 0.
            x = tf.where(tf.cast(x, tf.float32) >= 0.0, beta_for_hardening, -beta_for_hardening)

        meas_layer = self.measure_layers[0]
        comb_layer = self.combine_layers[0]
        sr = self.sr_layers[0]

        # Some layers may not accept a `training=` kwarg; fall back for
        # compatibility with older/custom implementations.
        try:
            meas_logits = tf.cast(meas_layer(x, training=training), tf.float32)
        except TypeError:
            meas_logits = tf.cast(meas_layer(x), tf.float32)

        # Depth=1 SR expects "previous" tensors; provide zeros and mark this
        # as the first measurement.
        zeros = tf.zeros_like(meas_logits)
        first_flag = tf.ones((tf.shape(meas_logits)[0], 1), dtype=tf.float32)
        sr_inputs = {
            "current_measurement": meas_logits,
            "previous_measurement": zeros,
            "previous_outcome": zeros,
            "first_measurement": first_flag,
        }

        if replay_out_a_logits_list is not None:
            replay_logits = tf.cast(tf.convert_to_tensor(replay_out_a_logits_list[0]), tf.float32)
            tf.debugging.assert_equal(
                tf.shape(replay_logits)[-1],
                tf.shape(meas_logits)[-1],
                message="Replay outcome length mismatch at level 0.",
            )
            sr_inputs["replay_outcome_logits"] = replay_logits

        out_logits = tf.cast(sr(sr_inputs, training=training), tf.float32)

        try:
            comm_logits = tf.cast(comb_layer(out_logits, training=training), tf.float32)
        except TypeError:
            comm_logits = tf.cast(comb_layer(out_logits), tf.float32)

        return comm_logits, [meas_logits], [out_logits]

    def _ensure_built(self) -> None:
        """Materialize variables by running a dummy forward pass if needed."""
        if not self.built:
            dummy = tf.zeros((1, self.n2), dtype=tf.float32)
            dummy_replay = [tf.zeros((1, self.n2), dtype=tf.float32)]
            _ = self.compute_with_internal(dummy, replay_out_a_logits_list=dummy_replay, training=False)
            self.built = True

    def save_weights_to(self, path: str) -> None:
        """Save model weights to ``path`` after ensuring variables exist."""
        self._ensure_built()
        super().save_weights(path)

    def load_weights_from(self, path: str) -> None:
        """Load model weights from ``path`` after ensuring variables exist."""
        self._ensure_built()
        super().load_weights(path)