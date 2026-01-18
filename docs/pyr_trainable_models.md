# Q_Sea_Battle.pyr_trainable_models

> Role: Re-export hub for Pyramid (Pyr) trainable assisted models for Player A and Player B.

Location: `Q_Sea_Battle.pyr_trainable_models`

## Overview

This module re-exports the trainable pyramid assisted models for Player A and Player B. It also defines project terminology used in the surrounding package: SR (shared resource) refers to any pre-shared auxiliary resource available to both players without communication; "shared randomness" is explicitly not used in this project’s terminology.

## Public API

### Functions

Not specified.

### Constants

- `__all__`: `["PyrTrainableAssistedModelA", "PyrTrainableAssistedModelB"]`  
  Purpose: Declares the public re-exports of this module.

### Types

Not specified.

## Dependencies

- `Q_Sea_Battle.pyr_trainable_assisted_model_a.PyrTrainableAssistedModelA`
- `Q_Sea_Battle.pyr_trainable_assisted_model_b.PyrTrainableAssistedModelB`

## Planned (design-spec)

Not specified.

## Deviations

Not specified.

## Notes for Contributors

- This module is a thin re-export layer; keep it free of implementation logic and restrict changes to import wiring and export surface (`__all__`) unless the package structure changes.

## Related

- `Q_Sea_Battle.pyr_trainable_assisted_model_a`
- `Q_Sea_Battle.pyr_trainable_assisted_model_b`

## Changelog

- 0.1: Initial re-export module for `PyrTrainableAssistedModelA` and `PyrTrainableAssistedModelB`.