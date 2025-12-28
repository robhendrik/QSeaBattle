"""Pyr combine layer A (placeholder)

Version: 0.1 (placeholders)

Notes:
    Interfaces only. Methods intentionally raise NotImplementedError.
"""
from __future__ import annotations
from typing import Optional
import tensorflow as tf


class PyrCombineLayerA(tf.keras.layers.Layer):
    """Pyr combine mapping for A (interface)."""

    def __init__(self, n2: int, m: int, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.n2 = int(n2)
        self.m = int(m)

    def call(self, outcomes_a: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError
