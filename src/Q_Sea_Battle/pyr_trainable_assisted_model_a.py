"""Pyr trainable assisted model A (placeholder)

Version: 0.1 (placeholders)

Notes:
    Interfaces only. Methods intentionally raise NotImplementedError.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import tensorflow as tf

from Q_Sea_Battle.shared_randomness_layer import SharedRandomnessLayer
from Q_Sea_Battle.pyr_measurement_layer_a import PyrMeasurementLayerA
from Q_Sea_Battle.pyr_combine_layer_a import PyrCombineLayerA


class PyrTrainableAssistedModelA(tf.keras.Model):
    """Pyr Model A interface (may use multiple SR calls)."""

    def __init__(self, game_layout, p_high: float = 0.9, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.n2 = int(game_layout.field_size ** 2)
        self.M = int(game_layout.comms_size)

        self.measure_layer = PyrMeasurementLayerA(self.n2)
        self.sr_layer = SharedRandomnessLayer(length=self.n2, p_high=float(p_high))
        self.combine_layer = PyrCombineLayerA(self.n2, self.M)

    def call(self, field_batch: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def compute_with_internal(
        self, field_batch: tf.Tensor
    ) -> Tuple[tf.Tensor, List[tf.Tensor], List[tf.Tensor]]:
        raise NotImplementedError
