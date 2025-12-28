"""Pyr measurement layer B (placeholder)

Version: 0.1 (placeholders)

Notes:
    Interfaces only. Methods intentionally raise NotImplementedError.
"""
from __future__ import annotations
from typing import Optional
import tensorflow as tf


class PyrMeasurementLayerB(tf.keras.layers.Layer):
    """Pyr measurement mapping for B (interface)."""

    def __init__(self, n2: int, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.n2 = int(n2)

    def call(self, gun_batch: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError
