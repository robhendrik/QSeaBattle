# Module pyr_teacher_layers

## Overview
This module defines the **teacher-layer primitives** for the Pyramid (Pyr) architecture.
It re-exports non-trainable measurement and combine layers that implement the
rule-based teacher strategy used for imitation-learning dataset generation.

The module contains **no trainable parameters** and serves as the algorithmic
reference for pyramid-based assisted play.

## Terminology
- **SR (shared resource)**: Any pre-shared auxiliary resource available to both
  players without communication.
- **PRAssistedLayer**: A specific type of SR.

## Exports
- `PyrMeasurementLayerA`
- `PyrMeasurementLayerB`
- `PyrCombineLayerA`
- `PyrCombineLayerB`

## Preconditions
- `field_size ** 2 = n2` is a power of two.
- `comms_size = 1` for pyramid architectures.
- GameLayout-derived constraints are respected.

## Postconditions
- Measurement layers reduce active dimension from `n2` to `n2 / 2`.
- Combine layers produce a valid next-level field representation.

## Errors
- `ValueError` if input shapes violate pyramid constraints.

## Examples
```python
from Q_Sea_Battle.pyr_teacher_layers import PyrMeasurementLayerA

layer = PyrMeasurementLayerA()
y = layer(x)  # x: np.ndarray, dtype int {0,1}, shape (n2,)
```

## Testing Hooks
- Output shape halves input shape at each level.
- Deterministic outputs for identical inputs.

## Changelog
- 2026-01-16 (Rob Hendriks): Initial MkDocs module page.
