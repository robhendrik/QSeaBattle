# Module lin_trainable_models

## Overview
This module defines the **trainable linear assisted models** for Player A and
Player B. These models combine linear teacher primitives with trainable
components to learn assisted strategies under linear communication constraints.

## Terminology
- **SR (shared resource)**: Any pre-shared auxiliary resource available to both
  players without communication.
- **PRAssistedLayer**: A specific type of SR.
- The term *shared randomness* is not used in this project.

## Module Import Path
`Q_Sea_Battle.lin_trainable_models`

## Exports
- `LinTrainableAssistedModelA`
- `LinTrainableAssistedModelB`

## Preconditions
- `field_size ** 2 = n2`.
- `comms_size = m` with `m | n2`.
- Training data respects linear layout constraints.

## Postconditions
- Player A emits a communication vector of size `m`.
- Player B emits a valid guess over `n2` positions.

## Errors
- `ValueError` if model construction violates layout constraints.
- `RuntimeError` if Player A / Player B configurations are inconsistent.

## Examples
```python
from Q_Sea_Battle.lin_trainable_models import LinTrainableAssistedModelB

model = LinTrainableAssistedModelB(layout)
y = model(x)  # x: tf.Tensor, dtype float32, shape (B, n2)
```

## Testing Hooks
- Output shapes are invariant under batch size.
- On noiseless teacher data, training converges to reference performance.

## Changelog
- 2026-01-16 (Rob Hendriks): Initial MkDocs module page.
