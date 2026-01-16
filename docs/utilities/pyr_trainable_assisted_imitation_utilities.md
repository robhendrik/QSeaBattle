# Module pyr_trainable_assisted_imitation_utilities

## Overview
Per-level dataset generation and training helpers for pyramid
trainable-assisted models.

## Module Import Path
`Q_Sea_Battle.pyr_trainable_assisted_imitation_utilities`

## Preconditions
- `field_size` per level is a power of two.
- `L >= 2` for all levels.

## Postconditions
- Generated datasets respect pyramid halving structure.

## Errors
- `ValueError` on invalid level sizes.

## Examples
```python
from Q_Sea_Battle.pyr_trainable_assisted_imitation_utilities import pyramid_levels
levels = pyramid_levels(16)
```

## Testing Hooks
- `pyramid_levels(16) == [16, 8, 4, 2]`

## Changelog
- 2026-01-16 (Rob Hendriks): Initial utilities module page.
