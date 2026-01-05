# Utility module: Data generation

## Purpose
Provide deterministic, spec-compliant utilities for generating datasets used in training and evaluation,
without introducing any additional information channels.

## Scope
These utilities are used exclusively for **offline data generation**.
They MUST NOT be used at runtime during actual games or tournaments.

## Core utilities

### `generate_measurement_dataset_a(layout, num_samples, seed)`
Generate training samples for Player A measurement layers.

**Inputs**
- `layout: GameLayout`
- `num_samples: int`
- `seed: int`

**Outputs**
- Dataset of `(field, meas_target)`

**Contract**
- Targets MUST be generated using classical AssistedPlayerA
- No gun information is accessible


### `generate_measurement_dataset_b(layout, num_samples, seed)`
Generate training samples for Player B measurement layers.

**Contract**
- Targets MUST be generated using classical AssistedPlayerB
- Field information MUST NOT be accessible

## Invariants
- Dataset shapes MUST match model interfaces
- Random seeds MUST make generation reproducible