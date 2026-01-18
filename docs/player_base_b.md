# PlayerB

> Role: Baseline B-side player that decides whether to shoot; default strategy is random.

Location: `Q_Sea_Battle.player_base_b.PlayerB`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | `GameLayout`, not specified, shape (not applicable) | Game configuration for this player; stored on the instance as `game_layout`. |

Preconditions

- `game_layout` is provided (not `None`); further constraints are not specified.

Postconditions

- `self.game_layout` is set to the provided `game_layout`.

Errors

- Not specified.

Example

```python
from Q_Sea_Battle.player_base_b import PlayerB
from Q_Sea_Battle.game_layout import GameLayout

game_layout = GameLayout()  # arguments not specified in this module
player_b = PlayerB(game_layout=game_layout)
```

## Public Methods

### decide(gun, comm, supp=None)

Decide whether to shoot based on gun position and message; the base implementation ignores inputs and returns a random decision in {0, 1}.

Parameters

- `gun`: `np.ndarray`, dtype not specified, flattened one-hot encoding, shape (n2,).
- `comm`: `np.ndarray`, dtype not specified, communication vector, shape (comms_size,).
- `supp`: `Optional[Any]`, constraint: may be `None`, shape (not applicable).

Returns

- `int`, constraint: value in {0, 1}, shape (not applicable).

Errors

- Not specified.

Example

```python
import numpy as np
from Q_Sea_Battle.player_base_b import PlayerB
from Q_Sea_Battle.game_layout import GameLayout

player_b = PlayerB(game_layout=GameLayout())
gun = np.zeros((10,), dtype=int)   # n2 is not specified in this module
comm = np.zeros((4,), dtype=float) # comms_size is not specified in this module
action = player_b.decide(gun=gun, comm=comm)
assert action in (0, 1)
```

## Data & State

- `game_layout`: `GameLayout`, not specified, shape (not applicable); shared configuration from the Players factory.

## Planned (design-spec)

- Not specified.

## Deviations

- No design notes were provided; no deviations can be derived.

## Notes for Contributors

- The current implementation of `decide` is intentionally input-agnostic and uses `np.random.randint(0, 2)`; subclasses may override `decide` to implement non-random strategies.

## Related

- `Q_Sea_Battle.game_layout.GameLayout`

## Changelog

- Not specified in module text.