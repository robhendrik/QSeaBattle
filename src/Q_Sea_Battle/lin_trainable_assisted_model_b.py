"""LinTrainableAssistedModelB.

Player B model for the linear trainable-assisted baseline.

Consumes tensors produced by Player A's ``compute_with_internal``:
- prev_meas_list[0]: measurement used by A (B, n2)
- prev_out_list[0]: SR outcome produced by A (B, n2)

Pipeline:
    gun -> LinMeasurementLayerB -> meas_probs_b
    meas_probs_b + prev tensors -> SharedRandomnessLayer (second measurement)
    (outcomes_b, comm) -> LinCombineLayerB -> shoot_logit

Spec (Step 5):
- call([gun, comm, prev_meas_list, prev_out_list]) -> (B, 1)
- Must use SR with ``first_measurement=0`` and consume prev tensors.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import tensorflow as tf

from .lin_measurement_layer_b import LinMeasurementLayerB
from .lin_combine_layer_b import LinCombineLayerB
from .shared_randomness_layer import SharedRandomnessLayer


class LinTrainableAssistedModelB(tf.keras.Model):
    """Linear trainable-assisted model for Player B."""

    def __init__(
        self,
        field_size: int,
        comms_size: int,
        *,
        sr_mode: str = "expected",
        seed: int | None = 0,
        p_high: float = 0.9,
        resource_index: int = 0,
        hidden_units_meas: Sequence[int] = (64,),
        hidden_units_combine: int | Sequence[int] = (64, 64),
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name or "LinTrainableAssistedModelB", **kwargs)
        self.field_size = int(field_size)
        self.comms_size = int(comms_size)
        self.n2 = self.field_size * self.field_size

        self.measurement = LinMeasurementLayerB(n2=self.n2, hidden_units=hidden_units_meas)
        self.shared_randomness = SharedRandomnessLayer(
            length=self.n2,
            p_high=float(p_high),
            mode=str(sr_mode),
            resource_index=int(resource_index),
            seed=seed,
            name="SharedRandomnessLayerB",
        )
        self.combine = LinCombineLayerB(comms_size=self.comms_size, hidden_units=hidden_units_combine)
        # Backwards-compatible aliases (match Model A)
        self.measure_layer = self.measurement
        self.sr_layer = self.shared_randomness
        self.combine_layer = self.combine
        
    @staticmethod
    def _ensure_batched(x: tf.Tensor) -> Tuple[tf.Tensor, bool]:
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        if x.shape.rank == 1:
            return tf.expand_dims(x, axis=0), False
        return x, True

    def call(self, inputs: Any, training: bool = False) -> tf.Tensor:
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 4:
            raise ValueError(
                "LinTrainableAssistedModelB.call expects inputs=[gun, comm, prev_meas_list, prev_out_list]."
            )

        gun, comm, prev_meas_list, prev_out_list = inputs

        if not isinstance(prev_meas_list, (list, tuple)) or len(prev_meas_list) < 1:
            raise ValueError("prev_meas_list must be a list/tuple with at least one tensor.")
        if not isinstance(prev_out_list, (list, tuple)) or len(prev_out_list) < 1:
            raise ValueError("prev_out_list must be a list/tuple with at least one tensor.")

        gun_b, _ = self._ensure_batched(gun)
        comm_b, _ = self._ensure_batched(comm)

        prev_meas0, _ = self._ensure_batched(prev_meas_list[0])
        prev_out0, _ = self._ensure_batched(prev_out_list[0])

        meas_probs_b = self.measurement(gun_b, training=training)

        # In sample mode, second measurement expects binary current measurement.
        if getattr(self.shared_randomness, "mode", "expected") == "sample":
            meas_for_sr = tf.cast(meas_probs_b >= 0.5, tf.float32)
        else:
            meas_for_sr = meas_probs_b

        bsz = tf.shape(meas_for_sr)[0]
        second_flag = tf.zeros((bsz, 1), dtype=tf.float32)

        outcomes_b = self.shared_randomness(
            {
                "current_measurement": meas_for_sr,
                "previous_measurement": prev_meas0,
                "previous_outcome": prev_out0,
                "first_measurement": second_flag,
            }
        )

        shoot_logit = self.combine(outcomes_b, comm_b, training=training)
        return shoot_logit
