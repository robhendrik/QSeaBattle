# Module pyr_trainable_models

## Overview
This module defines the **trainable pyramid models** for Player A and Player B.
These models learn to approximate the teacher strategy defined by
`pyr_teacher_layers` using imitation learning.

## Terminology
- **SR (shared resource)**: Any pre-shared auxiliary resource available to both
  players without communication.
- **PRAssistedLayer**: A specific type of SR.

## Exports
- `PyrTrainableAssistedModelA`
- `PyrTrainableAssistedModelB`

## Preconditions
- `field_size ** 2 = n2` is a power of two.
- `comms_size = 1`.
- Teacher-generated datasets respect pyramid structure.

## Postconditions
- Player A emits a single communication bit.
- Player B emits a valid guess consistent with pyramid constraints.

## Errors
- `ValueError` if model is constructed with invalid layout.
- `RuntimeError` if level structure between players mismatches.

## Examples
```python
from Q_Sea_Battle.pyr_trainable_models import PyrTrainableAssistedModelB

model = PyrTrainableAssistedModelB(layout)
y = model(x)  # x: tf.Tensor, dtype float32, shape (B, n2)
```

## Testing Hooks
- Model output shape is invariant under batch size.
- Learned outputs converge to teacher outputs on noiseless data.

## Changelog
- 2026-01-16 (Rob Hendriks): Initial MkDocs module page.
