# Q_Sea_Battle.lin_trainable_models

> Role: Re-export module for trainable linear (Lin) assisted models for Player A and Player B.

Location: `Q_Sea_Battle.lin_trainable_models`

## Overview

This module provides a consolidated import location for trainable linear assisted models used by two different players (A and B). It defines project terminology around shared resources (SR) and explicitly avoids the term "shared randomness" within this project.

Terminology (as specified by the module docstring): SR (shared resource) is any pre-shared auxiliary resource available to both players without communication; `PRAssistedLayer` (defined in `pr_assisted_layer.py`) is a specific type of SR.

## Public API

### Functions

Not specified.

### Constants

- `__all__`: `["LinTrainableAssistedModelA", "LinTrainableAssistedModelB"]`  
  Purpose: Declares the public symbols re-exported by this module.

### Types

Not specified.

## Dependencies

- `Q_Sea_Battle.lin_trainable_assisted_model_a.LinTrainableAssistedModelA` (imported and re-exported)
- `Q_Sea_Battle.lin_trainable_assisted_model_b.LinTrainableAssistedModelB` (imported and re-exported)

## Planned (design-spec)

Not specified.

## Deviations

Not specified.

## Notes for Contributors

- This module is a re-export layer; functional behavior and implementation details are expected to live in the imported modules.
- Keep `__all__` aligned with the intended public surface for stable imports.

## Related

- `Q_Sea_Battle.lin_trainable_assisted_model_a` (source of `LinTrainableAssistedModelA`)
- `Q_Sea_Battle.lin_trainable_assisted_model_b` (source of `LinTrainableAssistedModelB`)
- `Q_Sea_Battle.pr_assisted_layer` (mentioned as defining `PRAssistedLayer`; not imported here)

## Changelog

- Version: 0.1 (as specified in the module docstring)