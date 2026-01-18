"""Linear (Lin) teacher layer primitives.

This module re-exports the linear layer primitives used to build reference
(teacher) logic and trainable assisted models.

Terminology
- SR: shared resource. Any pre-shared auxiliary resource available to both
  players without communication. ``PRAssistedLayer`` (defined in
  ``pr_assisted_layer.py``) is a specific type of SR.
- The term "shared randomness" is not used in this project.

Exports
- LinMeasurementLayerA
- LinMeasurementLayerB
- LinCombineLayerA
- LinCombineLayerB

Notes
- The exported linear primitives are implemented as ``tf.keras.layers.Layer``.
  Some of them are learnable; the word "teacher" here refers to their role as
  low-level primitives in the reference pipeline and dataset specifications.

Author: Rob Hendriks
Package: Q_Sea_Battle
Version: 0.1
"""

from .lin_measurement_layer_a import LinMeasurementLayerA
from .lin_measurement_layer_b import LinMeasurementLayerB
from .lin_combine_layer_a import LinCombineLayerA
from .lin_combine_layer_b import LinCombineLayerB

__all__ = [
    "LinMeasurementLayerA",
    "LinMeasurementLayerB",
    "LinCombineLayerA",
    "LinCombineLayerB",
]
