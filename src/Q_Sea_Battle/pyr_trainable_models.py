"""Pyramid (Pyr) trainable assisted models.

This module re-exports the trainable pyramid models for Player A and Player B.

Terminology
- SR: shared resource. Any pre-shared auxiliary resource available to both
  players without communication. PRAssistedLayer is a specific type of SR.
- "Shared randomness" is not used in this project terminology.

Exports
- PyrTrainableAssistedModelA
- PyrTrainableAssistedModelB

Author: Rob Hendriks
"""

from .pyr_trainable_assisted_model_a import PyrTrainableAssistedModelA
from .pyr_trainable_assisted_model_b import PyrTrainableAssistedModelB

__all__ = ["PyrTrainableAssistedModelA", "PyrTrainableAssistedModelB"]
