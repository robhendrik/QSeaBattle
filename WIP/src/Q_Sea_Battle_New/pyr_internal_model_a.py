"""
pyr_internal_model_a.py

Internal trainable Pyramid Model A (logits-only, contract-aligned).

Key contract points:
- Input field_scaled values in {-0.5,+0.5} (scaled domain)
- Outputs are logits (no hardening)
- In replay/training mode, out_a_logits_list[d] equals replay_out_a_logits_list[d] by construction.

This model uses:
- PyrMeasurementLayerA (scaled -> meas logits)
- PRAssistedReplay (logits-only shared resource)
- PyrCombineLayerA (scaled + outcome logits -> next field logits)
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

import tensorflow as tf

from .pyr_measurement_layer_a import PyrMeasurementLayerA
from .pyr_combine_layer_a import PyrCombineLayerA
from .pr_assisted_replay import PRAssistedReplay


def _infer_n2_and_m(game_layout: Any) -> tuple[int, int]:
    """Infer (n2, m) from a GameLayout-like object."""
    if hasattr(game_layout, "n2"):
        n2 = int(getattr(game_layout, "n2"))
    else:
        field_size = int(getattr(game_layout, "field_size"))
        n2 = field_size * field_size
    m = int(getattr(game_layout, "comms_size", getattr(game_layout, "M", 1)))
    return n2, m


def _validate_power_of_two(n: int) -> int:
    """Return log2(n) if n is a power of two, else raise ValueError."""
    if n <= 0:
        raise ValueError("n2 must be positive.")
    k = int(round(math.log2(n)))
    if 2**k != n:
        raise ValueError(f"n2 must be a power of 2; got n2={n}.")
    return k


class PyrInternalModelA(tf.keras.Model):
    """Contract-aligned internal Model A (scaled-in, logits-out)."""

    def __init__(
        self,
        game_layout: Any,
        *,
        sr_mode: str = "replay",        # {"replay","stochastic"}
        p_high: float = 1.0,            # used only in stochastic mode inside PRAssistedReplay (second measurement)
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

        # SR layers (one per level), logits-only
        self.sr_layers: List[PRAssistedReplay] = []
        active = self.n2
        for level in range(self.depth):
            k = active // 2
            self.sr_layers.append(
                PRAssistedReplay(
                    sr_mode=sr_mode,
                    p_high=p_high,
                    beta=beta,
                    alpha=alpha,
                    seed=seed,
                    name=f"pr_replay_a_{level}",
                )
            )
            active //= 2

    def call(self, field_scaled: tf.Tensor, training: bool = False, **kwargs: Any) -> tf.Tensor:
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
        """
        Compute Model A forward pass using the *pure-logit* internal representation.

        This method implements the gameplay/training-forward used by the TrainableAssistedPlayer
        contract: internally, all tensors are logits (unbounded float32), and any conversion
        from bits/scaled representations happens outside the internal model (adapters/converters).

        Per level (depth):
        1) Measurement: meas_layer(state_logits) -> meas_logits (B, k_d)
        2) Assisted replay SR: sr({
                current_measurement=meas_logits,
                previous_measurement=zeros,
                previous_outcome=zeros,
                first_measurement=ones,
                [optional replay_outcome_logits=teacher forcing]
            }) -> out_logits (B, k_d)
        3) Combine A: comb_layer(state_logits, out_logits) -> next_state_logits (B, k_{d+1} or 1)

        Teacher forcing (supervised training):
        If replay_out_a_logits_list is provided, at each level d it is passed to SR as
        "replay_outcome_logits". This forces SR to emit the provided replay outcome logits
        (depending on SR implementation) while still allowing the model to compute consistent
        intermediate tensors for loss computation.

        Args
        ----
        field_logits:
            tf.Tensor float32, shape (B, n2). Logits representation of the field state.
            For Pyr models, n2 is the flattened board size (e.g. 16 for 4x4).
        replay_out_a_logits_list:
            Optional sequence of length self.depth. Each element is a tensor of shape (B, k_d)
            containing teacher-forced replay outcome logits for SR at level d.
            If None, SR operates in its normal mode (e.g., replay/stochastic depending on SR config).
        training:
            Boolean forwarded to Keras layers where supported.

        Returns
        -------
        comm_logits:
            tf.Tensor float32, shape (B, 1) for Pyr. Final communication logits produced by the
            last combine layer (i.e., last field logits).
        meas_list:
            Python list length self.depth; meas_list[d] has shape (B, k_d).
        out_list:
            Python list length self.depth; out_list[d] has shape (B, k_d).

        Raises
        ------
        ValueError:
            If input shapes are incompatible or replay_out_a_logits_list length mismatches depth.
        TypeError:
            If replay_out_a_logits_list is not a list/tuple when provided.
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
            
            # NOTE: this loop is very performance-sensitive; avoid unnecessary conversions or Python-side logic inside it.
            def harden_logits(logits, beta):
                logits = tf.cast(logits, tf.float32)
                return tf.where(logits >= 0.0, beta, -beta)
            if harden_between_levels:
                state_logits = harden_logits(state_logits, beta_for_hardening)
            
            # 1) Measurement (logits)
            # Keep try/except for backward compatibility with layers that don't accept `training=...`.
            try:
                meas_logits = tf.cast(meas_layer(state_logits, training=training), tf.float32)
            except TypeError:
                meas_logits = tf.cast(meas_layer(state_logits), tf.float32)

            # 2) Assisted replay SR
            # For Model A, this is always the "first measurement" in the SR sense (no previous).
            zeros = tf.zeros_like(meas_logits)
            first_flag = tf.ones((tf.shape(meas_logits)[0], 1), dtype=tf.float32)

            sr_inputs = {
                "current_measurement": meas_logits,
                "previous_measurement": zeros,
                "previous_outcome": zeros,
                "first_measurement": first_flag,
            }

            # Optional teacher forcing: provide replay outcome logits (must match meas dimension).
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

            # Pure-logit feed-forward
            state_logits = next_field_logits
            last_field_logits = next_field_logits

        if last_field_logits is None:
            raise RuntimeError("Internal error: model depth produced no outputs.")

        # For Pyr models, the final field logits are the comm logits (shape (B,1)).
        comm_logits = tf.cast(last_field_logits, tf.float32)
        return comm_logits, meas_list, out_list
    # -------------------------------------------------
    # Weight utilities
    # -------------------------------------------------

    def _ensure_built(self) -> None:
        """
        Ensure variables are created before loading weights.
        Runs a minimal dummy forward pass if needed.
        """
        if not self.built:
            dummy = tf.zeros((1, self.n2), dtype=tf.float32)-0.5
            dummy_replay_out_a_logits_list = [tf.zeros((1, self.n2 // (2 ** (d + 1))), dtype=tf.float32) for d in range(self.depth)]    
            _ = self.compute_with_internal(dummy, replay_out_a_logits_list=dummy_replay_out_a_logits_list, training=False)
            self.built = True
            
    def save_weights_to(self, path: str) -> None:
        """
        Save model weights to file.
        Example:
            model_a.save_weights_to("model_a_weights.h5")
        """
        self._ensure_built()
        super().save_weights(path)

    def load_weights_from(self, path: str) -> None:
        """
        Load model weights from file.
        Example:
            model_a.load_weights_from("model_a_weights.h5")
        """
        self._ensure_built()
        super().load_weights(path)
