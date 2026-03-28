"""
pyr_internal_model_b_purelogit.py

Pure-logit internal Pyramid Model B (contract-aligned with compute_with_internal).

This is an updated version of the existing `pyr_internal_model_b.py` (which used a
"scaled" gun/state representation internally). In the "pure-logit" architecture,
*all internal signals are logits*; no `sigmoid(x)-0.5` scaling is performed.

Contracts / expected boundary usage
-----------------------------------
Gameplay boundary (handled by adapters, not by this module):

- Player A samples `comm_bits` from `comm_logits` and passes `comm_bits` through the game.
- Therefore AdapterB must convert:
    comm_bits -> comm_in_logits (hard-logit with beta_comm)
    gun_bits  -> gun_logits     (hard-logit with beta_gun)

Internal model B inputs (pure-logit):
    shoot_logit, meas_b_list, out_b_list = model_b.compute_with_internal(
        gun_logits,          # (B, n2) float32 logits
        comm_in_logits,      # (B, 1)  float32 logits
        prev_meas_list,      # list length depth, each (B, k_d) float32 logits
        prev_out_list,       # list length depth, each (B, k_d) float32 logits
        training=False,
    )

The model is also callable for Player-compatibility:
    shoot_logit = model_b([gun_logits, comm_in_logits, prev_meas_list, prev_out_list])

and `call()` MUST delegate to `compute_with_internal()`.

Internal flow (per level)
-------------------------
For each level d=0..depth-1:
1) meas_b_logits = PyrMeasurementLayerB(state_logits)                       # logits -> logits
2) out_b_logits  = PRAssistedReplay(meas_b_logits, prev_meas, prev_out, ...) # logits -> logits
3) next_gun_logits, next_comm_logit = PyrCombineLayerB(state_logits, out_b_logits, comm_logit)
4) state_logits <- next_gun_logits (no sigmoid/scaling)
   comm_logit   <- next_comm_logit

Final:
    shoot_logit == comm_logit at the final level (shape (B,1), logits)

Notes
-----
- This module assumes your trained layers were trained with "hard_logit" inputs, i.e. they
  accept logits and emit logits.
- `PRAssistedReplay` already operates on logits, so it fits naturally.

"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import tensorflow as tf

from .pyr_measurement_layer_b import PyrMeasurementLayerB
from .pyr_combine_layer_b import PyrCombineLayerB
from .pr_assisted_replay import PRAssistedReplay
from .pyr_internal_model_a import _infer_n2_and_m, _validate_power_of_two


class PyrInternalModelB(tf.keras.Model):
    """Pure-logit internal Model B (logits-in, logits-out)."""

    def __init__(
        self,
        game_layout: Any,
        *,
        sr_mode: str = "replay",        # {"replay","stochastic"}
        p_high: float = 1.0,
        beta: float = 10.0,
        alpha: float = 5.0,
        seed: int | None = None,
        measure_layers: Optional[Sequence[tf.keras.layers.Layer]] = None,
        combine_layers: Optional[Sequence[tf.keras.layers.Layer]] = None,
        name: Optional[str] = None,
    ) -> None:
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

        # PRAssistedReplay per level
        self.sr_layers: List[PRAssistedReplay] = []
        active = self.n2
        for level in range(self.depth):
            _k = active // 2  # for readability; PRAssistedReplay doesn't require it explicitly here
            self.sr_layers.append(
                PRAssistedReplay(
                    sr_mode=sr_mode,
                    p_high=p_high,
                    beta=beta,
                    alpha=alpha,
                    seed=seed,
                    name=f"pr_replay_b_{level}",
                )
            )
            active //= 2

    def set_alpha(self, alpha: float) -> None:
        """Set PR gate sharpness for all SR layers (runtime-safe)."""
        for sr in self.sr_layers:
            if hasattr(sr, "set_alpha"):
                sr.set_alpha(alpha)
            else:
                raise AttributeError("SR layer has no set_alpha(); update PRAssistedReplay first.")

    def set_p_high(self, p_high: float) -> None:
        """Set stochastic follow probability for all SR layers (runtime-safe)."""
        for sr in self.sr_layers:
            if hasattr(sr, "set_p_high"):
                sr.set_p_high(p_high)
            else:
                raise AttributeError("SR layer has no set_p_high(); update PRAssistedReplay first.")

    def set_beta(self, beta: float) -> None:
        """Set hard-logit beta for all SR layers (runtime-safe)."""
        for sr in self.sr_layers:
            if hasattr(sr, "set_beta"):
                sr.set_beta(beta)
            else:
                raise AttributeError("SR layer has no set_beta(); update PRAssistedReplay first.")

    def set_sr_mode(self, sr_mode: str) -> None:
        """Set SR mode for all SR layers (runtime-safe)."""
        for sr in self.sr_layers:
            if hasattr(sr, "set_sr_mode"):
                sr.set_sr_mode(sr_mode)
            else:
                raise AttributeError("SR layer has no set_sr_mode(); update PRAssistedReplay first.")


    def call(self, inputs, training=False, **kwargs):
        """
        Player-facing call method, delegates to compute_with_internal.
        Expects inputs in the form:
        [gun_logits, comm_in_logits, *prev_meas_list, *prev_out_list]
        where prev_meas_list and prev_out_list are lists of tensors for each level.
        
        args:
        inputs: list or tuple of tensors, expected to be [gun_logits, comm_in_logits, *prev_meas_list, *prev_out_list]
        training: bool, whether in training mode (passed to internal layers)    
    
        returns:
        shoot_logit: (B,1) float32 logits for shooting decision
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
        """
            Pure-logit internal forward pass (Model B), contract-aligned.

            Inputs:
            gun_logits: (B,n2) float32 logits
            comm_in_logits: (B,1) float32 logits
            prev_meas_list: list length depth, each (B,k_d) float32 logits
            prev_out_list: list length depth, each (B,k_d) float32 logits

            Returns:
            shoot_logit: (B,1) float32 logits
            meas_b_logits_list: list length depth, each (B,k_d) float32 logits
            out_b_logits_list: list length depth, each (B,k_d) float32 logits
            comms_logits_list: list length depth+1, each (B,1) float32 logits
            gun_logits_list: list length depth+1, each (B,n2) float32

            NOTE: Harden logits between levels if self.harden_between_levels is True. Beta use is determined by self.beta_for_hardening.
            NOTE: A legacy version of this function had the return order of comms_logits_list and gun_logits_list swapped; be careful when comparing against older code or diagnostics.
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

        # NOTE: this loop is very performance-sensitive; avoid unnecessary conversions or Python-side logic inside it.
        def harden_logits(logits, beta):
            logits = tf.cast(logits, tf.float32)
            return tf.where(logits >= 0.0, beta, -beta)
        
        for level in range(self.depth):
            meas_layer = self.measure_layers[level]
            comb_layer = self.combine_layers[level]
            sr = self.sr_layers[level]

            # ======== Harden logits in gameplay mode to reflect deterministic measurements and outcomes ========
            if harden_between_levels:
                state_logits = harden_logits(state_logits, beta_for_hardening)
                c_logit = harden_logits(c_logit, beta_for_hardening)

            # --------------------------------------
            # Measurement B: logits -> logits
            # --------------------------------------

            meas_b_logits = tf.cast(meas_layer(state_logits, training=training), tf.float32)

            # ------------------------------------------------------------
            # 2) Assisted Replay
            # ------------------------------------------------------------
            prev_meas = tf.cast(tf.convert_to_tensor(prev_meas_list[level]), tf.float32)
            prev_out  = tf.cast(tf.convert_to_tensor(prev_out_list[level]), tf.float32)

            tf.debugging.assert_equal(tf.shape(prev_meas)[-1], tf.shape(meas_b_logits)[-1],
                                    message=f"prev_meas length mismatch at level {level}.")
            tf.debugging.assert_equal(tf.shape(prev_out)[-1], tf.shape(meas_b_logits)[-1],
                                    message=f"prev_out length mismatch at level {level}.")

            # Treat B-stage as "not first measurement"
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

            # ------------------------------------------------------------
            # 3) Combine
            # ------------------------------------------------------------
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

            # Feed forward
            state_logits = next_gun_logits
            c_logit = next_comm_logit

        shoot_logit = tf.cast(c_logit, tf.float32)
        return shoot_logit, meas_b_logits_list, out_b_logits_list, comms_logits_list, gun_logits_list
        #return shoot_logit, meas_b_logits_list, out_b_logits_list, gun_logits_list, comm_logits_list

    # -------------------------------------------------
    # Weight utilities--------

    def _ensure_built(self) -> None:
        """
        Ensure variables are created before loading weights.
        Runs a minimal dummy forward pass if needed.
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
        """
        Save model weights to file.
        Example:
            model_b.save_weights_to("model_b_weights.h5")
        """
        self._ensure_built()
        super().save_weights(path)

    def load_weights_from(self, path: str) -> None:
        """
        Load model weights from file.
        Example:
            model_b.load_weights_from("model_b_weights.h5")
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

    #     Inputs:
    #       gun_logits: (B,n2) float32 logits
    #       comm_in_logits: (B,1) float32 logits
    #       prev_meas_list: list length depth, each (B,k_d) float32 logits
    #       prev_out_list: list length depth, each (B,k_d) float32 logits

    # -----------------------------------------
    #     Returns:
    #       shoot_logit: (B,1) float32 logits
    #       meas_b_logits_list: list length depth, each (B,k_d) float32 logits
    #       out_b_logits_list: list length depth, each (B,k_d) float32 logits
    #     """
    #     gun_logits = tf.convert_to_tensor(gun_logits, dtype=tf.float32)
    #     comm = tf.convert_to_tensor(comm_in_logits, dtype=tf.float32)

    #     if gun_logits.shape.rank != 2:
    #         raise ValueError(f"gun_logits must be rank-2 (B,n2); got {gun_logits.shape}.")
    #     if gun_logits.shape[-1] is not None and int(gun_logits.shape[-1]) != self.n2:
    #         raise ValueError(f"gun_logits last dimension must be n2={self.n2}; got {gun_logits.shape[-1]}.")
    #     if comm.shape.rank != 2 or (comm.shape[-1] is not None and int(comm.shape[-1]) != 1):
    #         raise ValueError(f"comm_in_logits must be shape (B,1); got {comm.shape}.")

    #     if not isinstance(prev_meas_list, (list, tuple)) or not isinstance(prev_out_list, (list, tuple)):
    #         raise TypeError("prev_meas_list and prev_out_list must be Python lists/tuples of tensors.")
    #     if len(prev_meas_list) != self.depth or len(prev_out_list) != self.depth:
    #         raise ValueError(
    #             f"Previous lists must have length depth={self.depth}; got {len(prev_meas_list)} and {len(prev_out_list)}."
    #         )

    #     state_logits = gun_logits
    #     c_logit = comm

    #     meas_b_logits_list: list[tf.Tensor] = []
    #     out_b_logits_list: list[tf.Tensor] = []

    #     for level in range(self.depth):
    #         meas_layer = self.measure_layers[level]
    #         comb_layer = self.combine_layers[level]
    #         sr = self.sr_layers[level]

    #         # Measurement B: logits -> logits
    #         try:
    #             meas_b_logits = tf.cast(meas_layer(state_logits, training=training), tf.float32)
    #         except TypeError:
    #             meas_b_logits = tf.cast(meas_layer(state_logits), tf.float32)

    #         prev_meas = tf.cast(tf.convert_to_tensor(prev_meas_list[level]), tf.float32)
    #         prev_out = tf.cast(tf.convert_to_tensor(prev_out_list[level]), tf.float32)

    #         tf.debugging.assert_equal(
    #             tf.shape(prev_meas)[-1], tf.shape(meas_b_logits)[-1],
    #             message=f"prev_meas length mismatch at level {level}."
    #         )
    #         tf.debugging.assert_equal(
    #             tf.shape(prev_out)[-1], tf.shape(meas_b_logits)[-1],
    #             message=f"prev_out length mismatch at level {level}."
    #         )

    #         # In one-shot game, this is always "not first" for the replay stage during B.
    #         first_flag = tf.zeros((tf.shape(meas_b_logits)[0], 1), dtype=tf.float32)

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

    #         meas_b_logits_list.append(meas_b_logits)
    #         out_b_logits_list.append(out_b_logits)

    #         # Combine B: logits + logits + logits -> (next gun logits, next comm logit)
    #         try:
    #             next_gun_logits, next_comm_logit = comb_layer(state_logits, out_b_logits, c_logit, training=training)
    #         except TypeError:
    #             next_gun_logits, next_comm_logit = comb_layer(state_logits, out_b_logits, c_logit)

    #         state_logits = tf.cast(next_gun_logits, tf.float32)
    #         c_logit = tf.cast(next_comm_logit, tf.float32)

    #     shoot_logit = tf.cast(c_logit, tf.float32)
    #     return shoot_logit, meas_b_logits_list, out_b_logits_list, None, None