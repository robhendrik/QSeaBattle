"""LinTrainableAssistedModelA.

Player A model for the linear trainable-assisted baseline.

Pipeline:
    field -> LinMeasurementLayerA -> PRAssistedLayer (first measurement)
          -> LinCombineLayerA -> comm_logits

Public API:
    call(field_batch) -> comm_logits
    compute_with_internal(field_batch) -> (comm_logits, [meas], [out])
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import tensorflow as tf

from .lin_measurement_layer_a import LinMeasurementLayerA
from .lin_combine_layer_a import LinCombineLayerA
from .pr_assisted_layer import PRAssistedLayer


class LinTrainableAssistedModelA(tf.keras.Model):
    """Linear trainable-assisted model for Player A."""

    def __init__(
        self,
        field_size: int,
        comms_size: int,
        *,
        sr_mode: str = "expected",  # Shared resource mode (kept for backward-compatible naming).
        seed: int | None = 0,
        p_high: float = 0.9,
        resource_index: int = 0,
        hidden_units_meas: Sequence[int] = (64,),
        hidden_units_combine: int | Sequence[int] = (64, 64),
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name or "LinTrainableAssistedModelA", **kwargs)
        self.field_size = int(field_size)
        self.comms_size = int(comms_size)
        self.n2 = self.field_size * self.field_size

        self.measurement = LinMeasurementLayerA(n2=self.n2, hidden_units=hidden_units_meas)
        self.pr_assisted = PRAssistedLayer(
            length=self.n2,
            p_high=float(p_high),
            mode=str(sr_mode),
            resource_index=int(resource_index),
            seed=seed,
            name="PRAssistedLayerA",
        )
        self.combine = LinCombineLayerA(comms_size=self.comms_size, hidden_units=hidden_units_combine)

        # Backwards-compatible aliases.
        self.measure_layer = self.measurement
        self.sr_layer = self.pr_assisted
        self.combine_layer = self.combine

    @staticmethod
    def _ensure_batched(x: tf.Tensor) -> Tuple[tf.Tensor, bool]:
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        if x.shape.rank == 1:
            return tf.expand_dims(x, axis=0), False
        return x, True

    def compute_with_internal(
        self, field_batch: tf.Tensor, training: bool = False
    ) -> Tuple[tf.Tensor, List[tf.Tensor], List[tf.Tensor]]:
        field_batch, _ = self._ensure_batched(field_batch)

        meas_probs = self.measurement(field_batch, training=training)

        # In sample mode we must provide binary measurements.
        if getattr(self.pr_assisted, "mode", "expected") == "sample":
            meas_for_resource = tf.cast(meas_probs >= 0.5, tf.float32)
        else:
            meas_for_resource = meas_probs

        bsz = tf.shape(meas_for_resource)[0]
        zeros = tf.zeros_like(meas_for_resource)
        first_flag = tf.ones((bsz, 1), dtype=tf.float32)

        outcomes = self.pr_assisted(
            {
                "current_measurement": meas_for_resource,
                "previous_measurement": zeros,
                "previous_outcome": zeros,
                "first_measurement": first_flag,
            }
        )

        comm_logits = self.combine(outcomes, training=training)
        return comm_logits, [meas_for_resource], [outcomes]

    def call(self, field_batch: tf.Tensor, training: bool = False) -> tf.Tensor:
        comm_logits, _, _ = self.compute_with_internal(field_batch, training=training)
        return comm_logits