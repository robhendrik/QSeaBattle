"""
pyr_internal_model_b_purelogit.py

Pure-logit internal Pyramid Model B.

This module implements the Player B internal pyramid used by the QSeaBattle
adapters. Unlike earlier variants that used scaled or centered representations,
this "pure-logit" architecture treats *all internal signals as logits* from end
to end. In this convention, logical bit values are represented by the sign of a
logit (positive vs. negative), and no `sigmoid(x) - 0.5` scaling is performed.

Boundary contracts (handled by adapters, not by this module)
------------------------------------------------------------
Game boundaries operate on *bits*, while internal models operate on *logits*.

- Player A samples `comm_bits` from `comm_logits` and transmits `comm_bits`
  through the game.
- AdapterB is responsible for converting boundary bits back into logits using a
  "hard-logit" mapping (typically with `beta_comm` / `beta_gun`):
    - `comm_bits -> comm_in_logits`
    - `gun_bits  -> gun_logits`

Internal interface
------------------
The main forward path is `compute_with_internal()`:

    shoot_logit, meas_b_list, out_b_list, comms_list, guns_list = model_b.compute_with_internal(
        gun_logits,          # (B, n2) float32 logits
        comm_in_logits,      # (B, 1)  float32 logits
        prev_meas_list,      # length=depth, each (B, k_d) float32 logits
        prev_out_list,       # length=depth, each (B, k_d) float32 logits
        training=False,
    )

For Player-facing compatibility, the model is also callable. `call()` must
delegate to `compute_with_internal()`.

Per-level computation (d = 0..depth-1)
--------------------------------------
1) `meas_b_logits = PyrMeasurementLayerB(state_logits)` (logits -> logits)
2) `out_b_logits  = PRAssistedReplay(...)`              (logits -> logits)
3) `(next_gun_logits, next_comm_logit) = PyrCombineLayerB(...)`
4) `state_logits <- next_gun_logits` (no sigmoid/scaling)
   `comm_logit   <- next_comm_logit`

Final output
------------
The returned `shoot_logit` is the final `comm_logit` (shape (B, 1), logits).

Notes
-----
- This module assumes its layers were trained on logit inputs and produce logit
  outputs.
- `PRAssistedReplay` already operates on logits, so it integrates directly.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import tensorflow as tf

from .pyr_measurement_layer_b import PyrMeasurementLayerB
from .pyr_combine_layer_b import PyrCombineLayerB
from .pr_assisted_replay import PRAssistedReplay
from .pyr_internal_model_a import _infer_n2_and_m, _validate_power_of_two


class PyrInternalModelB(tf.keras.Model):
    """Pure-logit Pyramid internal model for Player B.

    The model consumes logits at its inputs and emits logits at its output.
    Its per-level structure is:

    - Measurement (B): logits -> logits
    - PR-assisted shared resource (SR): logits -> logits
    - Combine (B): (state logits, SR output logits, comm logits) -> next logits

    Attributes:
        n2: Number of gun/state bits (width of the gun/state vector).
        M: Communication size (must be 1 for this architecture).
        depth: Number of pyramid levels, inferred from `n2`.
        measure_layers: Per-level measurement layers.
        combine_layers: Per-level combine layers.
        sr_layers: Per-level PR-assisted SR layers (`PRAssistedReplay`).
    """

    def __init__(
        self,
        game_layout: Any,
        *,
        sr_mode: str = "replay",        # {"replay","stochastic"}
        p_rule: float = 1.0,
        beta: float = 10.0,
        alpha: float = 5.0,
        seed: int | None = None,
        measure_layers: Optional[Sequence[tf.keras.layers.Layer]] = None,
        combine_layers: Optional[Sequence[tf.keras.layers.Layer]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the internal model and construct per-level sublayers.

        Args:
            game_layout: Game configuration object. Its structure is interpreted by
                `_infer_n2_and_m()`.
            sr_mode: SR mode passed to each `PRAssistedReplay` layer. Supported
                values are `replay` and `stochastic` (see that layer for details).
            p_rule: Probability parameter used by SR layers (interpretation depends
                on SR mode).
            beta: Hard-logit inverse-temperature used inside SR layers.
            alpha: Gate sharpness used by SR layers.
            seed: Optional random seed forwarded to SR layers.
            measure_layers: Optional sequence of pre-constructed measurement layers,
                one per pyramid level. If provided, its length must equal `depth`.
            combine_layers: Optional sequence of pre-constructed combine layers,
                one per pyramid level. If provided, its length must equal `depth`.
            name: Optional Keras model name.

        Raises:
            ValueError: If `comms_size != 1`, or if provided layer lists do not
                match the inferred `depth`.
        """
        super().__init__(name=name)

        self.n2, self.M = _infer_n2_and_m(game_layout)
        if self.M != 1:
            raise ValueError(f"Pyr architecture requires comms_size==1; got m={self.M}.")
        self.depth = _validate_power_of_two(self.n2)

        if measure_layers is None:
            self.measure_layers: List[tf.keras.layers.Layer] = [PyrMeasurementLayerB() for _ in range(self.depth)]
        else:
            if len(measure_layers) != self.depth:
                raise ValueError(f"measure_layers must have length depth={self.depth}; got {len(measure_layers)}.")
            self.measure_layers = list(measure_layers)

        if combine_layers is None:
            self.combine_layers: List[tf.keras.layers.Layer] = [PyrCombineLayerB() for _ in range(self.depth)]
        else:
            if len(combine_layers) != self.depth:
                raise ValueError(f"combine_layers must have length depth={self.depth}; got {len(combine_layers)}.")
            self.combine_layers = list(combine_layers)

        # One PR-assisted SR layer per pyramid level.
        self.sr_layers: List[PRAssistedReplay] = []
        active = self.n2
        for level in range(self.depth):
            _k = active // 2  # for readability; PRAssistedReplay doesn't require it explicitly here
            self.sr_layers.append(
                PRAssistedReplay(
                    sr_mode=sr_mode,
                    p_rule=p_rule,
                    beta=beta,
                    alpha=alpha,
                    seed=seed,
                    name=f"pr_replay_b_{level}",
                )
            )
            active //= 2

    def set_alpha(self, alpha: float) -> None:
        """Set PR gate sharpness for all PR-assisted SR layers."""
        for sr in self.sr_layers:
            if hasattr(sr, "set_alpha"):
                sr.set_alpha(alpha)
            else:
                raise AttributeError("SR layer has no set_alpha(); update PRAssistedReplay first.")

    def set_p_rule(self, p_rule: float) -> None:
        """Set `p_rule` for all PR-assisted SR layers."""
        for sr in self.sr_layers:
            if hasattr(sr, "set_p_rule"):
                sr.set_p_rule(p_rule)
            else:
                raise AttributeError("SR layer has no set_p_rule(); update PRAssistedReplay first.")

    def set_beta(self, beta: float) -> None:
        """Set hard-logit beta for all PR-assisted SR layers."""
        for sr in self.sr_layers:
            if hasattr(sr, "set_beta"):
                sr.set_beta(beta)
            else:
                raise AttributeError("SR layer has no set_beta(); update PRAssistedReplay first.")

    def set_sr_mode(self, sr_mode: str) -> None:
        """Set SR mode for all PR-assisted SR layers."""
        for sr in self.sr_layers:
            if hasattr(sr, "set_sr_mode"):
                sr.set_sr_mode(sr_mode)
            else:
                raise AttributeError("SR layer has no set_sr_mode(); update PRAssistedReplay first.")

    def call(self, inputs, training=False, **kwargs):
        """Keras/Player-facing forward call.

        This method is a compatibility wrapper and delegates to
        `compute_with_internal()`.

        Two input packings are accepted:

        - Flat:
          `[gun_logits, comm_in_logits, *prev_meas_list, *prev_out_list]`
        - Nested:
          `[gun_logits, comm_in_logits, prev_meas_list, prev_out_list]`

        Args:
            inputs: List/tuple containing gun logits, comm logits, and previous
                per-level tensors as described above.
            training: Whether to run sublayers in training mode.
            **kwargs: Unused; accepted for Keras compatibility.

        Returns:
            tf.Tensor: `shoot_logit` of shape (B, 1) with dtype float32.
        """
        # inputs: [gun, comm, *prev_meas(depth), *prev_out(depth)]
        if not isinstance(inputs, (list, tuple)):
            raise TypeError("inputs must be a list/tuple.")

        # Accept either:
        #  A) flat: [gun, comm, *prev_meas(depth), *prev_out(depth)]
        #  B) nested: [gun, comm, prev_meas_list, prev_out_list]
        if len(inputs) == 4 and isinstance(inputs[2], (list, tuple)) and isinstance(inputs[3], (list, tuple)):
            gun, comm, prev_meas_list, prev_out_list = inputs
        else:
            if len(inputs) != 2 + 2 * self.depth:
                raise ValueError("Expected [gun, comm, *prev_meas, *prev_out].")
            gun, comm = inputs[0], inputs[1]
            prev_meas_list = list(inputs[2:2 + self.depth])
            prev_out_list  = list(inputs[2 + self.depth:2 + 2 * self.depth])

        shoot_logit, *_ = self.compute_with_internal(gun, comm, prev_meas_list, prev_out_list, training=training)
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
        training: bool = False
    ) -> tuple[tf.Tensor, list[tf.Tensor], list[tf.Tensor], list[tf.Tensor], list[tf.Tensor], list[tf.Tensor]]:
        """Run the pure-logit internal forward pass and return per-level signals.

        Args:
            gun_logits: Gun/state logits of shape (B, n2), dtype float32.
            comm_in_logits: Incoming comm logit of shape (B, 1), dtype float32.
            prev_meas_list: Previous measurements, length `depth`, each a logit
                tensor shaped (B, k_d).
            prev_out_list: Previous SR outcomes, length `depth`, each a logit
                tensor shaped (B, k_d).
            harden_between_levels: If True, harden intermediate logits between
                pyramid levels to +/- `beta_for_hardening` based on sign. This is
                typically used to emulate deterministic bit-boundary behavior
                between stages while keeping a logit representation.
            beta_for_hardening: Magnitude used when hardening logits.
            training: Whether to run sublayers in training mode.

        Returns:
            A tuple:
                - shoot_logit: (B, 1) float32 logits (final comm logit).
                - meas_b_logits_list: length `depth`, per-level measurement logits.
                - out_b_logits_list: length `depth`, per-level SR output logits.
                - comms_logits_list: length `depth + 1`, including the input comm
                  logit at index 0 and per-level next comm logits thereafter.
                - gun_logits_list: length `depth + 1`, including the input gun
                  logits at index 0 and per-level next gun logits thereafter.

        Raises:
            ValueError: If tensor ranks/shapes are inconsistent with the expected
                (B, n2) and (B, 1) contracts, or if previous lists are not of
                length `depth`.
            TypeError: If `prev_meas_list` / `prev_out_list` are not list/tuple
                containers.

        Notes:
            A legacy implementation returned `gun_logits_list` and
            `comms_logits_list` in the opposite order. When comparing with older
            diagnostics, ensure the return order matches this function.
        """
        gun_logits = tf.convert_to_tensor(gun_logits, dtype=tf.float32)
        comm = tf.convert_to_tensor(comm_in_logits, dtype=tf.float32)

        if gun_logits.shape.rank != 2:
            raise ValueError(f"gun_logits must be rank-2 (B,n2); got {gun_logits.shape}.")
        if gun_logits.shape[-1] is not None and int(gun_logits.shape[-1]) != self.n2:
            raise ValueError(f"gun_logits last dimension must be n2={self.n2}; got {gun_logits.shape[-1]}.")
        if comm.shape.rank != 2 or (comm.shape[-1] is not None and int(comm.shape[-1]) != 1):
            raise ValueError(f"comm_in_logits must be shape (B,1); got {comm.shape}.")

        if not isinstance(prev_meas_list, (list, tuple)) or not isinstance(prev_out_list, (list, tuple)):
            raise TypeError("prev_meas_list and prev_out_list must be Python lists/tuples of tensors.")
        if len(prev_meas_list) != self.depth or len(prev_out_list) != self.depth:
            raise ValueError(
                f"Previous lists must have length depth={self.depth}; got {len(prev_meas_list)} and {len(prev_out_list)}."
            )

        state_logits = gun_logits
        c_logit = comm

        meas_b_logits_list: list[tf.Tensor] = []
        out_b_logits_list: list[tf.Tensor] = []
        comms_logits_list: list[tf.Tensor] = [c_logit]
        gun_logits_list: list[tf.Tensor] = [gun_logits]

        # NOTE: This loop is performance-sensitive; avoid unnecessary conversions
        # or Python-side logic inside it.
        def harden_logits(logits, beta):
            """Map logits to a hard +/-beta representation by sign."""
            logits = tf.cast(logits, tf.float32)
            return tf.where(logits >= 0.0, beta, -beta)

        for level in range(self.depth):
            meas_layer = self.measure_layers[level]
            comb_layer = self.combine_layers[level]
            sr = self.sr_layers[level]

            # Optional hardening to approximate bit-boundary determinism between
            # levels while remaining in logit space.
            if harden_between_levels:
                state_logits = harden_logits(state_logits, beta_for_hardening)
                c_logit = harden_logits(c_logit, beta_for_hardening)

            # Measurement B: logits -> logits
            meas_b_logits = tf.cast(meas_layer(state_logits, training=training), tf.float32)

            # PR-assisted SR: logits -> logits
            prev_meas = tf.cast(tf.convert_to_tensor(prev_meas_list[level]), tf.float32)
            prev_out  = tf.cast(tf.convert_to_tensor(prev_out_list[level]), tf.float32)

            tf.debugging.assert_equal(
                tf.shape(prev_meas)[-1],
                tf.shape(meas_b_logits)[-1],
                message=f"prev_meas length mismatch at level {level}.",
            )
            tf.debugging.assert_equal(
                tf.shape(prev_out)[-1],
                tf.shape(meas_b_logits)[-1],
                message=f"prev_out length mismatch at level {level}.",
            )

            # B-stage uses the SR as a "not first measurement" step; the SR layer
            # expects a (B, 1) flag tensor.
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

            # Combine B: produces next gun logits and next comm logit.
            next_gun_logits, next_comm_logit = comb_layer(
                state_logits,
                out_b_logits,
                c_logit,
                training=training,
            )

            next_gun_logits = tf.cast(next_gun_logits, tf.float32)
            next_comm_logit = tf.cast(next_comm_logit, tf.float32)

            meas_b_logits_list.append(meas_b_logits)
            out_b_logits_list.append(out_b_logits)
            comms_logits_list.append(next_comm_logit)
            gun_logits_list.append(next_gun_logits)

            state_logits = next_gun_logits
            c_logit = next_comm_logit

        shoot_logit = tf.cast(c_logit, tf.float32)
        return shoot_logit, meas_b_logits_list, out_b_logits_list, comms_logits_list, gun_logits_list
        #return shoot_logit, meas_b_logits_list, out_b_logits_list, gun_logits_list, comm_logits_list

    # -------------------------------------------------
    # Weight utilities--------

    def _ensure_built(self) -> None:
        """Ensure variables are created before saving/loading weights.

        Keras models may be unbuilt until they see input shapes. This helper runs
        a minimal dummy forward pass if needed, using correctly shaped per-level
        tensors inferred from `n2` and `depth`.
        """
        if not self.built:
            B = 1
            dummy_gun = tf.zeros((B, self.n2), dtype=tf.float32)
            dummy_comm = tf.zeros((B, 1), dtype=tf.float32)

            dummy_prev_meas = [
                tf.zeros((B, self.n2 // (2 ** (d + 1))), dtype=tf.float32)
                for d in range(self.depth)
            ]
            dummy_prev_out = [
                tf.zeros((B, self.n2 // (2 ** (d + 1))), dtype=tf.float32)
                for d in range(self.depth)
            ]

            flat_inputs = [dummy_gun, dummy_comm] + dummy_prev_meas + dummy_prev_out
            _ = self(flat_inputs, training=False)

    def save_weights_to(self, path: str) -> None:
        """Save model weights to a file.

        Args:
            path: Destination path understood by `tf.keras.Model.save_weights`.
        """
        self._ensure_built()
        super().save_weights(path)

    def load_weights_from(self, path: str) -> None:
        """Load model weights from a file.

        Args:
            path: Source path understood by `tf.keras.Model.load_weights`.
        """
        self._ensure_built()
        super().load_weights(path)

    # def compute_with_internal(
    #     self,
    #     gun_logits: tf.Tensor,
    #     comm_in_logits: tf.Tensor,
    #     prev_meas_list: list,
    #     prev_out_list: list,
    #     *,
    #     training: bool = False,
    #     **kwargs: Any,
    # ) -> Tuple[tf.Tensor, list[tf.Tensor], list[tf.Tensor]]:
    #     """
    #     Pure-logit internal forward pass with per-level outputs.
    #
    #     Inputs:
    #       gun_logits: (B,n2) float32 logits
    #       comm_in_logits: (B,1) float32 logits
    #       prev_meas_list: list length depth, each (B,k_d) float32 logits
    #       prev_out_list: list length depth, each (B,k_d) float32 logits
    #
    # -----------------------------------------
    #     Returns:
    #       shoot_logit: (B,1) float32 logits
    #       meas_b_logits_list: list length depth, each (B,k_d) float32 logits
    #       out_b_logits_list: list length depth, each (B,k_d) float32 logits
    #     """
    #     gun_logits = tf.convert_to_tensor(gun_logits, dtype=tf.float32)
    #     comm = tf.convert_to_tensor(comm_in_logits, dtype=tf.float32)
    #
    #     if gun_logits.shape.rank != 2:
    #         raise ValueError(f"gun_logits must be rank-2 (B,n2); got {gun_logits.shape}.")
    #     if gun_logits.shape[-1] is not None and int(gun_logits.shape[-1]) != self.n2:
    #         raise ValueError(f"gun_logits last dimension must be n2={self.n2}; got {gun_logits.shape[-1]}.")
    #     if comm.shape.rank != 2 or (comm.shape[-1] is not None and int(comm.shape[-1]) != 1):
    #         raise ValueError(f"comm_in_logits must be shape (B,1); got {comm.shape}.")
    #
    #     if not isinstance(prev_meas_list, (list, tuple)) or not isinstance(prev_out_list, (list, tuple)):
    #         raise TypeError("prev_meas_list and prev_out_list must be Python lists/tuples of tensors.")
    #     if len(prev_meas_list) != self.depth or len(prev_out_list) != self.depth:
    #         raise ValueError(
    #             f"Previous lists must have length depth={self.depth}; got {len(prev_meas_list)} and {len(prev_out_list)}."
    #         )
    #
    #     state_logits = gun_logits
    #     c_logit = comm
    #
    #     meas_b_logits_list: list[tf.Tensor] = []
    #     out_b_logits_list: list[tf.Tensor] = []
    #
    #     for level in range(self.depth):
    #         meas_layer = self.measure_layers[level]
    #         comb_layer = self.combine_layers[level]
    #         sr = self.sr_layers[level]
    #
    #         # Measurement B: logits -> logits
    #         try:
    #             meas_b_logits = tf.cast(meas_layer(state_logits, training=training), tf.float32)
    #         except TypeError:
    #             meas_b_logits = tf.cast(meas_layer(state_logits), tf.float32)
    #
    #         prev_meas = tf.cast(tf.convert_to_tensor(prev_meas_list[level]), tf.float32)
    #         prev_out = tf.cast(tf.convert_to_tensor(prev_out_list[level]), tf.float32)
    #
    #         tf.debugging.assert_equal(
    #             tf.shape(prev_meas)[-1], tf.shape(meas_b_logits)[-1],
    #             message=f"prev_meas length mismatch at level {level}."
    #         )
    #         tf.debugging.assert_equal(
    #             tf.shape(prev_out)[-1], tf.shape(meas_b_logits)[-1],
    #             message=f"prev_out length mismatch at level {level}."
    #         )
    #
    #         # In one-shot game, this is always "not first" for the replay stage during B.
    #         first_flag = tf.zeros((tf.shape(meas_b_logits)[0], 1), dtype=tf.float32)
    #
    #         # Assisted replay: logits -> logits
    #         out_b_logits = tf.cast(
    #             sr(
    #                 {
    #                     "current_measurement": meas_b_logits,
    #                     "previous_measurement": prev_meas,
    #                     "previous_outcome": prev_out,
    #                     "first_measurement": first_flag,
    #                 },
    #                 training=training,
    #             ),
    #             tf.float32,
    #         )
    #
    #         meas_b_logits_list.append(meas_b_logits)
    #         out_b_logits_list.append(out_b_logits)
    #
    #         # Combine B: logits + logits + logits -> (next gun logits, next comm logit)
    #         try:
    #             next_gun_logits, next_comm_logit = comb_layer(state_logits, out_b_logits, c_logit, training=training)
    #         except TypeError:
    #             next_gun_logits, next_comm_logit = comb_layer(state_logits, out_b_logits, c_logit)
    #
    #         state_logits = tf.cast(next_gun_logits, tf.float32)
    #         c_logit = tf.cast(next_comm_logit, tf.float32)
    #
    #     shoot_logit = tf.cast(c_logit, tf.float32)
    #     return shoot_logit, meas_b_logits_list, out_b_logits_list, None, None