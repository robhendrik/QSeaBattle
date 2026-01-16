# Module logit_utilities

## Overview
Numerically stable utilities for converting between logits, probabilities,
and log-probabilities for Bernoulli variables.

## Module Import Path
`Q_Sea_Battle.logit_utilities`

## Public Interface
Functions:
- `logit_to_prob`
- `logit_to_logprob`

## Preconditions
- Inputs must be broadcastable to a common shape.
- Action values must lie in {0,1}.

## Postconditions
- Returned probabilities lie in `[0.0, 1.0]`.

## Errors
- `ValueError` if broadcasting fails or invalid actions are provided.

## Examples
```python
from Q_Sea_Battle.logit_utilities import logit_to_prob
p = logit_to_prob(0.0)
```

## Testing Hooks
- `logit_to_prob(0.0) == 0.5`
- `logit_to_logprob(0.0, 1) < 0.0`

## Changelog
- 2026-01-16 (Rob Hendriks): Initial utilities module page.
