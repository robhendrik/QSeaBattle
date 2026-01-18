"""LinTrainableAssistedModelB.

Player B model for the linear trainable-assisted baseline.

Consumes tensors produced by Player A's ``compute_with_internal``:
- prev_meas_list[0]: measurement used by A (B, n2)
- prev_out_list[0]: PR-assisted outcome produced by A (B, n2)

Pipeline:
    gun -> LinMeasurementLayerB -> meas_probs_b
    meas_probs_b + prev tensors -> PRAssistedLayer (second measurement)
    (outcomes_b, comm) -> LinCombineLayerB -> shoot_logit

Spec (Step 5):
- call([gun, comm, prev_meas_list, prev_out_list]) -> (B, 1)
- Must use PR-assisted resource with ``first_measurement=0`` and consume prev tensors.

Author: Rob Hendriks
Package: Q_Sea_Battle
Version: 0.1
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import tensorflow as tf

from .lin_measurement_layer_b import LinMeasurementLayerB
from .lin_combine_layer_b import LinCombineLayerB
from .pr_assisted_layer import PRAssistedLayer


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
        self.pr_assisted = PRAssistedLayer(
            length=self.n2,
            p_high=float(p_high),
            mode=str(sr_mode),
            resource_index=int(resource_index),
            seed=seed,
            name="PRAssistedLayerB",
        )
        self.combine = LinCombineLayerB(comms_size=self.comms_size, hidden_units=hidden_units_combine)

        # Backwards-compatible aliases
        self.measure_layer = self.measurement
        self.pr_layer = self.pr_assisted
        self.resource_layer = self.pr_assisted
        self.sr_layer = self.pr_assisted  # deprecated alias
        self.combine_layer = self.combine

    @staticmethod
    def _ensure_batched(x: tf.Tensor) -> Tuple[tf.Tensor, bool]:
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        if x.shape.rank == 1:
            return tf.expand_dims(x, axis=0), False
        return x, True

    def compute_with_internal(
        self,
        gun_batch: tf.Tensor,
        comm_batch: tf.Tensor,
        prev_meas_list: List[tf.Tensor],
        prev_out_list: List[tf.Tensor],
        training: bool = False,
    ) -> Tuple[tf.Tensor, List[tf.Tensor], List[tf.Tensor]]:
        """Compute shoot logits and return intermediate tensors.

        Returns:
            shoot_logit: (B, 1)
            meas_list: [meas_for_resource_b] where (B, n2)
            out_list: [outcomes_b] where (B, n2)
        """
        gun_batch, _ = self._ensure_batched(gun_batch)
        comm_batch, _ = self._ensure_batched(comm_batch)

        # Normalize comm shape to (B, comms_size)
        comm_batch = tf.cast(comm_batch, tf.float32)
        if comm_batch.shape.rank != 2:
            raise ValueError(f"comm_batch must be rank-2; got shape {comm_batch.shape}.")
        if comm_batch.shape[-1] != self.comms_size:
            raise ValueError(
                f"comm_batch last dim must be comms_size={self.comms_size}; got {comm_batch.shape[-1]}."
            )

        # Ensure gun shape is (B, n2)
        gun_batch = tf.cast(gun_batch, tf.float32)
        if gun_batch.shape.rank != 2:
            raise ValueError(f"gun_batch must be rank-2; got shape {gun_batch.shape}.")
        if gun_batch.shape[-1] != self.n2:
            raise ValueError(f"gun_batch last dim must be n2={self.n2}; got {gun_batch.shape[-1]}.")

        # Accept "prev_*" as list/tuple or a single tensor; normalize to list of len 1.
        if not isinstance(prev_meas_list, (list, tuple)):
            prev_meas_list = [prev_meas_list]  # type: ignore[list-item]
        if not isinstance(prev_out_list, (list, tuple)):
            prev_out_list = [prev_out_list]  # type: ignore[list-item]
        if len(prev_meas_list) < 1 or len(prev_out_list) < 1:
            raise ValueError("prev_meas_list and prev_out_list must be non-empty lists.")

        prev_meas = tf.cast(tf.convert_to_tensor(prev_meas_list[0]), tf.float32)
        prev_out = tf.cast(tf.convert_to_tensor(prev_out_list[0]), tf.float32)

        # Measurement for B (probabilities in [0,1])
        meas_probs_b = self.measurement(gun_batch, training=training)

        # In sample mode the PR-assisted resource expects binary measurements.
        if getattr(self.pr_assisted, "mode", "expected") == "sample":
            meas_for_resource_b = tf.cast(meas_probs_b >= 0.5, tf.float32)
        else:
            meas_for_resource_b = tf.cast(meas_probs_b, tf.float32)

        # Basic shape checks against previous tensors
        if prev_meas.shape.rank != 2 or prev_out.shape.rank != 2:
            raise ValueError("prev_meas and prev_out must be rank-2 (B, n2).")
        if prev_meas.shape[-1] != self.n2 or prev_out.shape[-1] != self.n2:
            raise ValueError("prev_meas and prev_out must have last dim n2.")
        if meas_for_resource_b.shape.rank != 2 or meas_for_resource_b.shape[-1] != self.n2:
            raise ValueError("meas_for_resource_b must have shape (B, n2).")

        bsz = tf.shape(meas_for_resource_b)[0]
        first_flag = tf.zeros((bsz, 1), dtype=tf.float32)

        outcomes_b = self.pr_assisted(
            {
                "current_measurement": meas_for_resource_b,
                "previous_measurement": prev_meas,
                "previous_outcome": prev_out,
                "first_measurement": first_flag,
            }
        )

        # Combine outcomes with comm to produce shoot logit.
        # Different versions of LinCombineLayerB may accept (outcomes, comm) or a dict.
        try:
            shoot_logit = self.combine(outcomes_b, comm_batch, training=training)
        except TypeError:
            shoot_logit = self.combine({"outcomes": outcomes_b, "comm": comm_batch}, training=training)

        # Ensure shape (B, 1)
        shoot_logit = tf.convert_to_tensor(shoot_logit, dtype=tf.float32)
        if shoot_logit.shape.rank == 1:
            shoot_logit = tf.expand_dims(shoot_logit, axis=-1)

        return shoot_logit, [meas_for_resource_b], [outcomes_b]

    def call(self, inputs: list, training: bool = False, **kwargs: Any) -> tf.Tensor:
        """Forward pass.

        Args:
            inputs: [gun_batch, comm_batch, prev_meas_list, prev_out_list]

        Returns:
            shoot_logit: Tensor (B, 1)
        """
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 4:
            raise ValueError(
                "LinTrainableAssistedModelB.call expects 4 inputs: "
                "[gun, comm, prev_meas_list, prev_out_list]."
            )

        gun_batch, comm_batch, prev_meas_list, prev_out_list = inputs
        shoot_logit, _, _ = self.compute_with_internal(
            gun_batch=gun_batch,
            comm_batch=comm_batch,
            prev_meas_list=prev_meas_list,  # type: ignore[arg-type]
            prev_out_list=prev_out_list,    # type: ignore[arg-type]
            training=training,
        )
        return shoot_logit
