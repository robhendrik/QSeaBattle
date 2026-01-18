# Q_Sea_Battle.pyr_teacher_layers

> Role: Re-export module for pyramid (Pyr) teacher layer primitives used as non-trainable teacher rules for pyramid imitation-learning dataset generation.

Location: `Q_Sea_Battle.pyr_teacher_layers`

## Overview

This module provides a single import surface for the pyramid teacher layer primitives used to define teacher rules for pyramid imitation-learning dataset generation. It re-exports the following symbols from sibling modules: `PyrMeasurementLayerA`, `PyrMeasurementLayerB`, `PyrCombineLayerA`, and `PyrCombineLayerB`.

Terminology (as defined by this module docstring): SR refers to a shared resource available to both players without communication; "Shared randomness" is not used in this project's terminology.

## Public API

### Functions

Not specified.

### Constants

- `__all__`: List of public re-exports: `["PyrMeasurementLayerA", "PyrMeasurementLayerB", "PyrCombineLayerA", "PyrCombineLayerB"]`.

### Types

Not specified.

## Dependencies

- Imports (relative): `.pyr_measurement_layer_a.PyrMeasurementLayerA`, `.pyr_measurement_layer_b.PyrMeasurementLayerB`, `.pyr_combine_layer_a.PyrCombineLayerA`, `.pyr_combine_layer_b.PyrCombineLayerB`.

## Planned (design-spec)

Not specified.

## Deviations

Not specified.

## Notes for Contributors

- This module is a re-export layer; changes here should generally be limited to updating imports and `__all__` to reflect the intended public surface.  
- The implementation details, signatures, and behavior of the exported classes are defined in their respective modules (not shown here).

## Related

- `Q_Sea_Battle.pyr_measurement_layer_a`
- `Q_Sea_Battle.pyr_measurement_layer_b`
- `Q_Sea_Battle.pyr_combine_layer_a`
- `Q_Sea_Battle.pyr_combine_layer_b`

## Changelog

- Version: 0.1 (as stated in module docstring).