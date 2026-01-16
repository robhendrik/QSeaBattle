# Module dru_utilities

## Overview
Utilities implementing the Discretize / Regularize Unit (DRU) used in
DIAL-style training of communicating agents.

## Module Import Path
`Q_Sea_Battle.dru_utilities`

## Terminology
- DRU: deterministic transformation of message logits.
- No trainable parameters.

## Public Interface
Functions:
- `dru_train`
- `dru_execute`

## Preconditions
- `sigma >= 0.0`.
- Inputs are logits over `m` communication dimensions.

## Postconditions
- Training output lies in `(0.0, 1.0)`.
- Execution output lies in `{0,1}`.

## Errors
- `ValueError` if `sigma < 0`.

## Examples
```python
from Q_Sea_Battle.dru_utilities import dru_execute
bits = dru_execute([1.0, -1.0])
```

## Testing Hooks
- `dru_execute(0.0) == 0`
- `dru_train(0.0, sigma=0.0) == 0.5`

## Changelog
- 2026-01-16 (Rob Hendriks): Initial utilities module page.
