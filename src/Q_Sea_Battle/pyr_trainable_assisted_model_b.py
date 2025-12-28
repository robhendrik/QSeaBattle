"""Pyr trainable assisted model B (placeholder)

Version: 0.1 (placeholders)

Notes:
    Interfaces only. Methods intentionally raise NotImplementedError.
"""
from __future__ import annotations
from typing import Optional
import tensorflow as tf

from Q_Sea_Battle.shared_randomness_layer import SharedRandomnessLayer
from Q_Sea_Battle.pyr_measurement_layer_b import PyrMeasurementLayerB
from Q_Sea_Battle.pyr_combine_layer_b import PyrCombineLayerB


class PyrTrainableAssistedModelB(tf.keras.Model):
    """Pyr Model B interface (may use multiple SR calls)."""

    def __init__(self, game_layout, p_high: float = 0.9, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.n2 = int(game_layout.field_size ** 2)
        self.M = int(game_layout.comms_size)

        self.measure_layer = PyrMeasurementLayerB(self.n2)
        self.sr_layer = SharedRandomnessLayer(length=self.n2, p_high=float(p_high))
        self.combine_layer = PyrCombineLayerB(self.n2, self.M)

    def call(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        raise NotImplementedError
