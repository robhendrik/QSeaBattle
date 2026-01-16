"""Pyramid (Pyr) teacher layer primitives.

This module re-exports the non-trainable pyramid layer primitives that define
the teacher rules used for pyramid imitation-learning dataset generation.

Terminology
- SR: shared resource. Any pre-shared auxiliary resource available to both
  players without communication. PRAssistedLayer is a specific type of SR.
- "Shared randomness" is not used in this project terminology.

Exports
- PyrMeasurementLayerA
- PyrMeasurementLayerB
- PyrCombineLayerA
- PyrCombineLayerB

Author: Rob Hendriks
"""

from .pyr_measurement_layer_a import PyrMeasurementLayerA
from .pyr_measurement_layer_b import PyrMeasurementLayerB
from .pyr_combine_layer_a import PyrCombineLayerA
from .pyr_combine_layer_b import PyrCombineLayerB

__all__ = [
    "PyrMeasurementLayerA",
    "PyrMeasurementLayerB",
    "PyrCombineLayerA",
    "PyrCombineLayerB",
]
