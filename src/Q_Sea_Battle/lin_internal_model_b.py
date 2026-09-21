"""Trainable linear internal model B (logits-only; contract-aligned).

This module implements a depth-1 internal model used by the Pyramid-compatible
training/inference stack. The model consumes *logits* (not probabilities). A
logical bit is represented by the sign of its corresponding logit.

Contract summary:
- Fixed depth: 1.
- Inputs:
  - `gun_logits`: rank-2 float tensor shaped (B, n2)
  - `comm_in_logits`: rank-2 float tensor shaped (B, m)
  - `prev_meas_list`: list/tuple of length 1; each element shaped (B, n2)
  - `prev_out_list`: list/tuple of length 1; each element shaped (B, n2)
- Output:
  - `shoot_logit`: rank-2 float tensor shaped (B, 1)

`compute_with_internal()` returns a Pyramid-compatible tuple:
  (shoot_logit, meas_b_list, out_b_list, comms_logits_list, gun_logits_list)

The trace lists are trivial depth-1 traces.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import tensorflow as tf

from .lin_measurement_layer_b import LinMeasurementLayerB
from .lin_combine_layer_b import LinCombineLayerB
from .pr_assisted_replay import PRAssistedReplay
from .pyr_internal_model_a import _infer_n2_and_m


class LinInternalModelB(tf.keras.Model):
    """Depth-1 linear internal model B (logits-in, logits-out).

    The model is composed of:
    - A measurement layer that maps `gun_logits` -> measurement logits.
    - A PR-assisted shared resource (SR) layer that produces an outcome logits
      vector using the current measurement and previous-step traces.
    - A combine layer that mixes SR outcome logits with incoming communication
      logits to produce a final `shoot_logit` (and an auxiliary flip logit).

    Attributes:
        n2: Number of gun bits (flattened board size).
        M: Communication channel width in bits/logits.
        depth: Fixed to 1 for this architecture.
        last_flip_logit: Most recent flip logit produced by the combine layer,
            or None if `compute_with_internal()` has not been called.
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
        """Initialize the internal model.

        Args:
            game_layout: Game layout object used to infer `n2` and `m`.
            sr_mode: SR mode passed to the PR-assisted SR implementation. The
                project uses "replay" and "stochastic".
            p_rule: Probability weight for the rule-based branch inside the
                PR-assisted SR (see `PRAssistedReplay`).
            beta: Logit scale used by the PR-assisted SR.
            alpha: Mixing/strength parameter used by the PR-assisted SR.
            seed: Optional seed forwarded to the PR-assisted SR.
            measure_layers: Optional custom measurement layers. Must have length
                equal to `depth` (1). If None, a default layer is created.
            combine_layers: Optional custom combine layers. Must have length
                equal to `depth` (1). If None, a default layer is created.
            hidden_units_measure: Hidden size for the default measurement layer.
            hidden_units_combine: Hidden size for the default combine layer.
            name: Optional Keras model name.

        Raises:
            ValueError: If `m < 1` or if provided layer lists do not match
                `depth`.
        """
        super().__init__(name=name)
        self.n2, self.M = _infer_n2_and_m(game_layout)
        if self.M < 1:
            raise ValueError(f"Linear architecture requires comms_size>=1; got m={self.M}.")
        self.depth = 1

        if measure_layers is None:
            self.measure_layers: List[tf.keras.layers.Layer] = [
                LinMeasurementLayerB(hidden_units=hidden_units_measure)
            ]
        else:
            if len(measure_layers) != self.depth:
                raise ValueError(f"measure_layers must have length depth={self.depth}; got {len(measure_layers)}.")
            self.measure_layers = list(measure_layers)

        if combine_layers is None:
            self.combine_layers: List[tf.keras.layers.Layer] = [
                LinCombineLayerB(comms_size=self.M, hidden_units=hidden_units_combine)
            ]
        else:
            if len(combine_layers) != self.depth:
                raise ValueError(f"combine_layers must have length depth={self.depth}; got {len(combine_layers)}.")
            self.combine_layers = list(combine_layers)

        # Convenience aliases for depth-1 usage.
        self.measure_layer = self.measure_layers[0]
        self.combine_layer = self.combine_layers[0]

        # PR-assisted shared resource (SR), configured as a depth-1 list to match
        # the multi-level interface used by other internal models.
        self.sr_layers: List[PRAssistedReplay] = [
            PRAssistedReplay(
                sr_mode=sr_mode,
                p_rule=p_rule,
                beta=beta,
                alpha=alpha,
                seed=seed,
                name="pr_replay_b_0",
            )
        ]
        self.sr_layer = self.sr_layers[0]
        self.last_flip_logit: tf.Tensor | None = None

    def set_alpha(self, alpha: float) -> None:
        """Update SR `alpha` for all SR layers."""
        for sr in self.sr_layers:
            if hasattr(sr, "set_alpha"):
                sr.set_alpha(alpha)
            else:
                raise AttributeError("SR layer has no set_alpha(); update PRAssistedReplay first.")

    def set_p_rule(self, p_rule: float) -> None:
        """Update SR `p_rule` for all SR layers."""
        for sr in self.sr_layers:
            if hasattr(sr, "set_p_rule"):
                sr.set_p_rule(p_rule)
            else:
                raise AttributeError("SR layer has no set_p_rule(); update PRAssistedReplay first.")

    def set_beta(self, beta: float) -> None:
        """Update SR `beta` for all SR layers."""
        for sr in self.sr_layers:
            if hasattr(sr, "set_beta"):
                sr.set_beta(beta)
            else:
                raise AttributeError("SR layer has no set_beta(); update PRAssistedReplay first.")

    def set_sr_mode(self, sr_mode: str) -> None:
        """Switch SR mode (e.g., "replay" vs "stochastic") for all SR layers."""
        for sr in self.sr_layers:
            if hasattr(sr, "set_sr_mode"):
                sr.set_sr_mode(sr_mode)
            else:
                raise AttributeError("SR layer has no set_sr_mode(); update PRAssistedReplay first.")

    def call(self, inputs, training: bool = False, **kwargs):
        """Keras forward pass.

        Accepts either:
        - `[gun, comm, prev_meas_list, prev_out_list]` where the last two items
          are lists/tuples; or
        - a flattened input list `[gun, comm, *prev_meas, *prev_out]` where the
          number of previous tensors matches `depth`.

        Args:
            inputs: Model inputs in one of the accepted formats.
            training: Keras training flag.
            **kwargs: Unused; present for Keras compatibility.

        Returns:
            Shoot logits as a float tensor shaped (B, 1).
        """
        if not isinstance(inputs, (list, tuple)):
            raise TypeError("inputs must be a list/tuple.")

        if len(inputs) == 4 and isinstance(inputs[2], (list, tuple)) and isinstance(inputs[3], (list, tuple)):
            gun, comm, prev_meas_list, prev_out_list = inputs
        else:
            if len(inputs) != 2 + 2 * self.depth:
                raise ValueError("Expected [gun, comm, *prev_meas, *prev_out].")
            gun, comm = inputs[0], inputs[1]
            prev_meas_list = list(inputs[2:2 + self.depth])
            prev_out_list = list(inputs[2 + self.depth:2 + 2 * self.depth])

        shoot_logit, *_ = self.compute_with_internal(
            gun, comm, prev_meas_list, prev_out_list, training=training
        )
        return shoot_logit

    def compute_with_internal(
        self,
        gun_logits: tf.Tensor,
        comm_in_logits: tf.Tensor,
        prev_meas_list: Sequence[tf.Tensor],
        prev_out_list: Sequence[tf.Tensor],
        harden_between_levels: bool = False,
        beta_for_hardening: float = 10.0,
        *,
        training: bool = False,
    ) -> tuple[tf.Tensor, list[tf.Tensor], list[tf.Tensor], list[tf.Tensor], list[tf.Tensor]]:
        """Compute outputs and return internal traces.

        This method follows the "internal model" contract used by the Pyramid
        stack: it produces the shoot logit and depth-indexed trace lists.

        Notes:
            - All values are logits. When `harden_between_levels` is enabled,
              logits are replaced with fixed-magnitude logits (+/- beta) based
              on sign. This is a non-differentiable straight-through-style
              discretization used to emulate bit boundaries between stages.

        Args:
            gun_logits: Gun logits, float tensor shaped (B, n2).
            comm_in_logits: Communication logits, float tensor shaped (B, m).
            prev_meas_list: Previous measurement trace list (length 1).
            prev_out_list: Previous SR outcome trace list (length 1).
            harden_between_levels: Whether to discretize logits between stages.
            beta_for_hardening: Magnitude used when hardening logits.
            training: Keras training flag.

        Returns:
            A 5-tuple `(shoot_logit, meas_b_logits_list, out_b_logits_list,
            comms_logits_list, gun_logits_list)` where:
              - shoot_logit: float tensor shaped (B, 1)
              - meas_b_logits_list: length-1 list of measurement logits
              - out_b_logits_list: length-1 list of SR outcome logits
              - comms_logits_list: `[comm_in_logits, shoot_logit]`
              - gun_logits_list: `[gun_logits, gun_logits]` (contract trace)

        Raises:
            ValueError: If input ranks or static last-dimension sizes are
                incompatible with `n2`/`m`, or if previous trace list lengths do
                not match `depth`.
            TypeError: If previous trace containers are not list/tuple.
        """
        gun_logits = tf.convert_to_tensor(gun_logits, dtype=tf.float32)
        comm = tf.convert_to_tensor(comm_in_logits, dtype=tf.float32)

        if gun_logits.shape.rank != 2:
            raise ValueError(f"gun_logits must be rank-2 (B,n2); got {gun_logits.shape}.")
        if gun_logits.shape[-1] is not None and int(gun_logits.shape[-1]) != self.n2:
            raise ValueError(f"gun_logits last dimension must be n2={self.n2}; got {gun_logits.shape[-1]}.")
        if comm.shape.rank != 2:
            raise ValueError(f"comm_in_logits must be rank-2 (B,m); got {comm.shape}.")
        if comm.shape[-1] is not None and int(comm.shape[-1]) != self.M:
            raise ValueError(f"comm_in_logits last dimension must be m={self.M}; got {comm.shape[-1]}.")

        if not isinstance(prev_meas_list, (list, tuple)) or not isinstance(prev_out_list, (list, tuple)):
            raise TypeError("prev_meas_list and prev_out_list must be lists/tuples of tensors.")
        if len(prev_meas_list) != self.depth or len(prev_out_list) != self.depth:
            raise ValueError(
                f"Previous lists must have length depth={self.depth}; got {len(prev_meas_list)} and {len(prev_out_list)}."
            )

        state_logits = gun_logits
        c_logits = comm

        def harden_logits(x: tf.Tensor, beta_val: float) -> tf.Tensor:
            """Map logits to fixed-magnitude logits by sign."""
            x = tf.cast(x, tf.float32)
            return tf.where(x >= 0.0, beta_val, -beta_val)

        if harden_between_levels:
            state_logits = harden_logits(state_logits, beta_for_hardening)
            c_logits = harden_logits(c_logits, beta_for_hardening)

        meas_layer = self.measure_layers[0]
        comb_layer = self.combine_layers[0]
        sr = self.sr_layers[0]

        meas_b_logits = tf.cast(meas_layer(state_logits, training=training), tf.float32)

        prev_meas = tf.cast(tf.convert_to_tensor(prev_meas_list[0]), tf.float32)
        prev_out = tf.cast(tf.convert_to_tensor(prev_out_list[0]), tf.float32)

        # Dynamic shape checks: enforce that traces match the measurement width.
        tf.debugging.assert_equal(
            tf.shape(prev_meas)[-1], tf.shape(meas_b_logits)[-1], message="prev_meas length mismatch at level 0."
        )
        tf.debugging.assert_equal(
            tf.shape(prev_out)[-1], tf.shape(meas_b_logits)[-1], message="prev_out length mismatch at level 0."
        )

        # Level-0 always uses "not first measurement" in this architecture.
        first_flag = tf.zeros((tf.shape(meas_b_logits)[0], 1), dtype=tf.float32)

        out_b_logits = tf.cast(
            sr(
                {
                    "current_measurement": meas_b_logits,
                    "previous_measurement": prev_meas,
                    "previous_outcome": prev_out,
                    "first_measurement": first_flag,
                },
                training=training,
            ),
            tf.float32,
        )
        if harden_between_levels:
            out_b_logits = harden_logits(out_b_logits, beta_for_hardening)

        shoot_logit_raw, flip_logit_raw = comb_layer(
            out_b_logits,
            c_logits,
            training=training,
            return_flip=True,
        )
        shoot_logit = tf.cast(shoot_logit_raw, tf.float32)
        self.last_flip_logit = tf.cast(flip_logit_raw, tf.float32)

        meas_b_logits_list = [meas_b_logits]
        out_b_logits_list = [out_b_logits]
        comms_logits_list = [c_logits, shoot_logit]
        gun_logits_list = [gun_logits, gun_logits]

        return shoot_logit, meas_b_logits_list, out_b_logits_list, comms_logits_list, gun_logits_list

    def _ensure_built(self) -> None:
        """Create variables by running a minimal forward pass if needed."""
        if not self.built:
            dummy_gun = tf.zeros((1, self.n2), dtype=tf.float32)
            dummy_comm = tf.zeros((1, self.M), dtype=tf.float32)
            dummy_prev_meas = [tf.zeros((1, self.n2), dtype=tf.float32)]
            dummy_prev_out = [tf.zeros((1, self.n2), dtype=tf.float32)]
            _ = self.compute_with_internal(
                dummy_gun,
                dummy_comm,
                dummy_prev_meas,
                dummy_prev_out,
                training=False,
            )
            self.built = True

    def save_weights_to(self, path: str) -> None:
        """Save Keras weights after ensuring variables are built."""
        self._ensure_built()
        super().save_weights(path)

    def load_weights_from(self, path: str) -> None:
        """Load Keras weights after ensuring variables are built."""
        self._ensure_built()
        super().load_weights(path)