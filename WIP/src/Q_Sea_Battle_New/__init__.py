"""
Q_Sea_Battle_New

Work-in-progress implementations of next-generation QSeaBattle
layers, internal models, and gameplay adapters.

This package is intentionally isolated from the stable Q_Sea_Battle
package while interfaces and architectures are being finalized.
"""

from __future__ import annotations

# --- Core shared resources ---
from .pr_assisted_replay import PRAssistedReplay


# --- Dataset utilities ---
from .pyr_dataset_generation_utilities import generate_pyr_dataset
from .pyr_dataset_generation_utilities import save_npz



from Q_Sea_Battle_New.pyr_dataset_conversion_utilities import (
    convert_layer_measure_a,
    convert_layer_combine_a,
    convert_layer_measure_b,
    convert_layer_combine_b,
    convert_internal_model_a,
    convert_internal_model_b,
    convert_full_system,
)

    
# --- Pyramid internal models ---
from .pyr_internal_model_a import PyrInternalModelA
from .pyr_internal_model_b import PyrInternalModelB



__all__ = [
    "PRAssistedReplay",
    "PyrInternalModelA",
    "PyrInternalModelB",
    "GameplayPyrModelAAdapter",
    "GameplayPyrModelBAdapter",
]
