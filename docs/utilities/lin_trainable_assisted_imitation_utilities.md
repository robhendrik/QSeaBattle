# Module lin_trainable_assisted_imitation_utilities

## Overview
Utilities for generating supervised imitation datasets and training helpers
for linear trainable-assisted models.

## Module Import Path
`Q_Sea_Battle.lin_trainable_assisted_imitation_utilities`

## Preconditions
- `field_size >= 1`.
- `1 <= comms_size <= n2`.

## Postconditions
- Generated targets implement parity-based linear teacher rules.

## Errors
- `ValueError` on inconsistent dataset shapes.

## Examples
```python
from Q_Sea_Battle.lin_trainable_assisted_imitation_utilities import generate_measurement_dataset_a
ds = generate_measurement_dataset_a(layout, 128)
```

## Testing Hooks
- Parity targets equal XOR reduction of inputs.

## Changelog
- 2026-01-16 (Rob Hendriks): Initial utilities module page.
