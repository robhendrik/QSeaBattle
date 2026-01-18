# Q_Sea_Battle.lin_teacher_layers

> Role: Re-export hub for linear (Lin) teacher layer primitives used to build reference (teacher) logic and trainable assisted models.

Location: `Q_Sea_Battle.lin_teacher_layers`

## Overview

This module provides a single import surface for the linear layer primitives used in the project’s “teacher” (reference) pipeline and dataset specifications. It defines no layers itself; instead, it re-exports four `tf.keras.layers.Layer` implementations from sibling modules and declares them in `__all__`.

Terminology used by the module documentation: SR (shared resource) refers to any pre-shared auxiliary resource available to both players without communication; the term “shared randomness” is explicitly not used in this project.

## Public API

### Functions

Not specified.

### Constants

- `__all__`: `list[str]` (export list) containing: `"LinMeasurementLayerA"`, `"LinMeasurementLayerB"`, `"LinCombineLayerA"`, `"LinCombineLayerB"`.

### Types

Not specified.

## Dependencies

- Imports (relative): `.lin_measurement_layer_a.LinMeasurementLayerA`, `.lin_measurement_layer_b.LinMeasurementLayerB`, `.lin_combine_layer_a.LinCombineLayerA`, `.lin_combine_layer_b.LinCombineLayerB`
- External dependencies: Not specified in this module (the docstring mentions `tf.keras.layers.Layer` as the implementation base class for the exported primitives, but TensorFlow is not imported here).

## Planned (design-spec)

Not specified.

## Deviations

Not specified.

## Notes for Contributors

- This module is intended to remain a thin re-export layer; add new exports by importing them and updating `__all__` in the same order they should appear for consumers.
- Avoid adding side effects here; keep imports and export-list maintenance only.

## Related

- `Q_Sea_Battle.lin_measurement_layer_a` (source of `LinMeasurementLayerA`)
- `Q_Sea_Battle.lin_measurement_layer_b` (source of `LinMeasurementLayerB`)
- `Q_Sea_Battle.lin_combine_layer_a` (source of `LinCombineLayerA`)
- `Q_Sea_Battle.lin_combine_layer_b` (source of `LinCombineLayerB`)
- `pr_assisted_layer.py` (mentioned in docstring as defining `PRAssistedLayer`; not part of this module’s code)

## Changelog

- 0.1: Initial module providing re-exports for linear teacher layer primitives (per module docstring).