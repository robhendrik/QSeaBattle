# Module neural_net_imitation_utilities

## Overview
Dataset-generation utilities for imitation training of NeuralNetPlayers
based on majority strategies.

## Module Import Path
`Q_Sea_Battle.neural_net_imitation_utilities`

## Preconditions
- `field_size >= 1`.
- `1 <= comms_size <= n2`.

## Postconditions
- Generated datasets match NeuralNetPlayers input expectations.

## Errors
- `ValueError` on invalid layout or dataset sizes.

## Examples
```python
from Q_Sea_Battle.neural_net_imitation_utilities import make_segments
segments = make_segments(layout)
```

## Testing Hooks
- Segments fully cover `[0, n2)` without overlap.

## Changelog
- 2026-01-16 (Rob Hendriks): Initial utilities module page.
