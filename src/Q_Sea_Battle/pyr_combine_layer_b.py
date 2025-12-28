"""Pyr combine layer B (placeholder)

Version: 0.1 (placeholders)

Notes:
    Interfaces only. Methods intentionally raise NotImplementedError.
"""
from __future__ import annotations
from typing import Optional
import tensorflow as tf


class PyrCombineLayerB(tf.keras.layers.Layer):
    """Pyr combine mapping for B (interface)."""

    def __init__(self, n2: int, m: int, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.n2 = int(n2)
        self.m = int(m)

    def call(self, outcomes_b: tf.Tensor, comm_batch: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError
