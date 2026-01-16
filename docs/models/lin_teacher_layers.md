# Module lin_teacher_layers

## Overview
This module defines the **teacher-layer primitives** for the Linear (Lin)
architecture. It re-exports the low-level measurement and combine layers that
are used as reference components in linear assisted strategies and dataset
specifications.

The term *teacher* refers to the role of these layers in defining reference
behavior; individual layers may still contain trainable parameters.

## Terminology
- **SR (shared resource)**: Any pre-shared auxiliary resource available to both
  players without communication.
- **PRAssistedLayer**: A specific type of SR.
- The term *shared randomness* is not used in this project.

## Module Import Path
`Q_Sea_Battle.lin_teacher_layers`

## Exports
- `LinMeasurementLayerA`
- `LinMeasurementLayerB`
- `LinCombineLayerA`
- `LinCombineLayerB`

## Preconditions
- `field_size ** 2 = n2`.
- `comms_size = m` with `m | n2`.
- Inputs conform to the active linear layout derived from `GameLayout`.

## Postconditions
- Measurement layers emit representations consistent with linear partitioning.
- Combine layers emit valid next-stage representations of size `n2 / m`.

## Errors
- `ValueError` if input shapes or layout-derived constraints are violated.

## Examples
```python
from Q_Sea_Battle.lin_teacher_layers import LinMeasurementLayerA

layer = LinMeasurementLayerA()
y = layer(x)  # x: tf.Tensor, dtype float32, shape (B, n2)
```

## Testing Hooks
- Output shapes match the expected linear reduction rules.
- Identical inputs produce identical outputs when layers are in inference mode.

## Changelog
- 2026-01-16 (Rob Hendriks): Initial MkDocs module page.
