"""Linear (Lin) trainable assisted models.

This module re-exports the trainable linear assisted models for Player A and
Player B.

Terminology
- SR: shared resource. Any pre-shared auxiliary resource available to both
  players without communication. ``PRAssistedLayer`` (defined in
  ``pr_assisted_layer.py``) is a specific type of SR.
- The term "shared randomness" is not used in this project.

Exports
- LinTrainableAssistedModelA
- LinTrainableAssistedModelB
"""

from .lin_trainable_assisted_model_a import LinTrainableAssistedModelA
from .lin_trainable_assisted_model_b import LinTrainableAssistedModelB

__all__ = ["LinTrainableAssistedModelA", "LinTrainableAssistedModelB"]
