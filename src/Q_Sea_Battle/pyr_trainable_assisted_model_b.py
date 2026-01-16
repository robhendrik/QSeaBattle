"""Pyramid Trainable Assisted Model B.

This module defines :class:`~Q_Sea_Battle.pyr_trainable_assisted_model_b.PyrTrainableAssistedModelB`,
the Player-B model for the pyramid (Pyr) assisted architecture.

(This file is a small robustness patch over the existing version: it accepts
the Keras `training` kwarg in `call` and passes it through to sublayers when possible.)
"""

from __future__ import annotations

import tensorflow as tf
from typing import Any, List, Optional, Sequence

from .pyr_measurement_layer_b import PyrMeasurementLayerB
from .pyr_combine_layer_b import PyrCombineLayerB
from .pr_assisted_layer import PRAssistedLayer
from .pyr_trainable_assisted_model_a import _infer_n2_and_m, _validate_power_of_two


class PyrTrainableAssistedModelB(tf.keras.Model):
    """Player-B pyramid assisted model (per-level layers)."""

    def __init__(
        self,
        game_layout: Any,
        p_high: float = 0.9,
        sr_mode: str = "sample",
        measure_layers: Optional[Sequence[tf.keras.layers.Layer]] = None,
        combine_layers: Optional[Sequence[tf.keras.layers.Layer]] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.n2, self.M = _infer_n2_and_m(game_layout)
        if self.M != 1:
            raise ValueError(f"Pyr architecture requires comms_size==1; got {self.M}.")
        self.depth = _validate_power_of_two(self.n2)

        if measure_layers is None:
            self.measure_layers: List[tf.keras.layers.Layer] = [PyrMeasurementLayerB() for _ in range(self.depth)]
        else:
            if len(measure_layers) != self.depth:
                raise ValueError(f"measure_layers must have length K={self.depth}; got {len(measure_layers)}.")
            self.measure_layers = list(measure_layers)

        if combine_layers is None:
            self.combine_layers: List[tf.keras.layers.Layer] = [PyrCombineLayerB() for _ in range(self.depth)]
        else:
            if len(combine_layers) != self.depth:
                raise ValueError(f"combine_layers must have length K={self.depth}; got {len(combine_layers)}.")
            self.combine_layers = list(combine_layers)

        # Backward-compat aliases.
        self.measure_layer = self.measure_layers[0]
        self.combine_layer = self.combine_layers[0]

        self.sr_layers: List[PRAssistedLayer] = []
        active = self.n2
        for level in range(self.depth):
            length = active // 2
            self.sr_layers.append(
                PRAssistedLayer(length=length, p_high=p_high, mode=sr_mode, resource_index=level)
            )
            active //= 2

    def call(self, inputs: list, training: bool = False, **kwargs: Any) -> tf.Tensor:
        """Forward pass.

        inputs:
            [gun_batch, comm_batch, prev_measurements, prev_outcomes]

        Returns:
            shoot_logit: Tensor (B, 1)
        """
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 4:
            raise ValueError(
                "PyrTrainableAssistedModelB.call expects 4 inputs: "
                "[gun, comm, prev_measurements, prev_outcomes]."
            )
        gun_batch, comm_batch, prev_measurements, prev_outcomes = inputs

        gun = tf.cast(tf.convert_to_tensor(gun_batch), tf.float32)
        comm = tf.cast(tf.convert_to_tensor(comm_batch), tf.float32)

        if gun.shape.rank != 2:
            raise ValueError(f"gun_batch must be rank-2 (B,n2); got shape {gun.shape}.")
        if comm.shape.rank != 2 or comm.shape[-1] != 1:
            raise ValueError(f"comm_batch must be (B,1); got shape {comm.shape}.")

        if not isinstance(prev_measurements, (list, tuple)) or not isinstance(prev_outcomes, (list, tuple)):
            raise TypeError("prev_measurements and prev_outcomes must be Python lists/tuples of tensors.")
        if len(prev_measurements) != self.depth or len(prev_outcomes) != self.depth:
            raise ValueError(
                f"Previous lists must have length K={self.depth}; got "
                f"{len(prev_measurements)} and {len(prev_outcomes)}."
            )

        state = gun
        c = tf.clip_by_value(comm, 0.0, 1.0)

        for level, sr in enumerate(self.sr_layers):
            meas_layer = self.measure_layers[level]
            comb_layer = self.combine_layers[level]

            # measurement layer may or may not accept training kwarg
            try:
                meas_b = tf.cast(meas_layer(state, training=training), tf.float32)
            except TypeError:
                meas_b = tf.cast(meas_layer(state), tf.float32)

            prev_meas = tf.cast(tf.convert_to_tensor(prev_measurements[level]), tf.float32)
            prev_out = tf.cast(tf.convert_to_tensor(prev_outcomes[level]), tf.float32)

            if prev_out.shape.rank != 2 or meas_b.shape.rank != 2:
                raise ValueError("Per-level tensors must be rank-2 (B, length).")
            if prev_out.shape[-1] != meas_b.shape[-1] or prev_meas.shape[-1] != meas_b.shape[-1]:
                raise ValueError(
                    f"Per-level length mismatch at level {level}: "
                    f"meas_b={meas_b.shape[-1]}, prev_meas={prev_meas.shape[-1]}, prev_out={prev_out.shape[-1]}"
                )

            first_flag = tf.zeros((tf.shape(meas_b)[0], 1), dtype=tf.float32)
            out_b = sr(
                {
                    "current_measurement": meas_b,
                    "previous_measurement": prev_meas,
                    "previous_outcome": prev_out,
                    "first_measurement": first_flag,
                }
            )

            # combine layer may or may not accept training kwarg
            try:
                state, c = comb_layer(state, out_b, c, training=training)
            except TypeError:
                state, c = comb_layer(state, out_b, c)

        c = tf.clip_by_value(c, 0.0, 1.0)
        shoot_logit = (c * 2.0 - 1.0) * 10.0
        return shoot_logit
