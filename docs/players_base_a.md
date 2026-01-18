# PlayerA

> Role: Baseline A-side player that returns a random binary communication vector.

Location: `Q_Sea_Battle.players_base_a.PlayerA`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | GameLayout, constraints: not specified, shape: N/A | Game configuration for this player; stored as `self.game_layout`. |

Preconditions: `game_layout` must be an instance compatible with `GameLayout` and must provide attribute `comms_size` used by `decide` (type/constraints not specified in this module).  
Postconditions: `self.game_layout` is set to the provided `game_layout`.  
Errors: Not specified.  
Example:

```python
from Q_Sea_Battle.players_base_a import PlayerA
from Q_Sea_Battle.game_layout import GameLayout

game_layout = GameLayout(...)  # Not specified here
player = PlayerA(game_layout)
```

## Public Methods

### decide(field, supp=None)

Decide on a communication vector given the field; the base implementation ignores inputs and returns a random binary vector of length $m = \mathrm{comms\_size}$.

Parameters:

- `field`: np.ndarray, dtype int {0,1}, shape (field_size,), flattened field array containing 0/1 values; content is unused in the base implementation.
- `supp`: Optional[Any], constraints: may be None, shape: N/A, optional supporting information (unused in base class).

Returns:

- np.ndarray, dtype int {0,1}, shape (m,), a one-dimensional communication array of length $m = \mathrm{self.game\_layout.comms\_size}$.

Preconditions: `self.game_layout.comms_size` must be an integer $m \ge 0$ (exact constraints not specified in this module).  
Postconditions: Returns a newly generated random vector; no object state is mutated.  
Errors: Not specified.  
Example:

```python
import numpy as np
from Q_Sea_Battle.players_base_a import PlayerA
from Q_Sea_Battle.game_layout import GameLayout

game_layout = GameLayout(...)  # Must provide .comms_size
player = PlayerA(game_layout)

field_size = 100
field = np.zeros((field_size,), dtype=int)
comms = player.decide(field)
```

## Data & State

- `game_layout`: GameLayout, constraints: not specified, shape: N/A, shared configuration object stored at construction time; used to read `comms_size` during `decide`.

## Planned (design-spec)

Not specified.

## Deviations

None identified.

## Notes for Contributors

- `decide` uses `np.random.randint(0, 2, size=m, dtype=int)`; changes to randomness, seeding, or output dtype/shape should preserve the contract: np.ndarray of dtype int with values in {0,1} and shape (m,).
- The module does not validate `field` dtype/values/shape or `game_layout.comms_size`; add validation only if coordinated with upstream expectations.

## Related

- `Q_Sea_Battle.game_layout.GameLayout`

## Changelog

- 0.2: Initial baseline implementation with random communication strategy.