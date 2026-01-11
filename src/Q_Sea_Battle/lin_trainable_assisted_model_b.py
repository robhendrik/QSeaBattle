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
