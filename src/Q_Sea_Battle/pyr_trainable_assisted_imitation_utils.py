"""Pyr imitation utils (placeholder)

Version: 0.1 (placeholders)

Notes:
    Interfaces only. Methods intentionally raise NotImplementedError.
"""
from __future__ import annotations


def generate_measurement_dataset_a(layout, num_samples: int, seed: int | None = None):
    """Synthetic dataset for PyrMeasurementLayerA (interface)."""
    raise NotImplementedError


def generate_measurement_dataset_b(layout, num_samples: int, seed: int | None = None):
    """Synthetic dataset for PyrMeasurementLayerB (interface)."""
    raise NotImplementedError


def generate_combine_dataset_a(layout, num_samples: int, seed: int | None = None):
    """Synthetic dataset for PyrCombineLayerA (interface)."""
    raise NotImplementedError


def generate_combine_dataset_b(layout, num_samples: int, seed: int | None = None):
    """Synthetic dataset for PyrCombineLayerB (interface)."""
    raise NotImplementedError
