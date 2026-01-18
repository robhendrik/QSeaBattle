# PlayerA

> Role: Baseline A-side player that emits a random binary communication vector of length $m=\mathrm{comms\_size}$, independent of inputs.
Location: `Q_Sea_Battle.player_base_a.PlayerA`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | GameLayout, constraints: not specified, shape: not applicable | Game configuration for this player; stored as `self.game_layout`. |

Preconditions

- `game_layout` must provide attribute `comms_size` (type and constraints not specified in this module).

Postconditions

- `self.game_layout` is set to the provided `game_layout`.

Errors

- Not specified.

!!! example "Example"
    ```python
    from Q_Sea_Battle.player_base_a import PlayerA
    from Q_Sea_Battle.game_layout import GameLayout
    
    layout = GameLayout(...)  # Not specified in this module
    player = PlayerA(game_layout=layout)
    ```

## Public Methods

### decide(field, supp=None)

Decide on a communication vector given the field; the base implementation ignores both inputs and returns a random binary vector of length $m=\mathrm{comms\_size}$.

Parameters

- `field`: np.ndarray, dtype int {0,1}, shape (n2,); flattened field array containing 0/1 values; content is unused in the base implementation.
- `supp`: Optional[Any], constraints: may be None, shape: not applicable; optional supporting information (unused in base class).

Returns

- np.ndarray, dtype int {0,1}, shape (m,); a one-dimensional communication array with entries in {0, 1}, where $m=\mathrm{game\_layout.comms\_size}$.

Preconditions

- `self.game_layout.comms_size` is defined and is usable as the `size` argument to `np.random.randint` (exact type constraints not specified).

Postconditions

- No state changes are specified.

Errors

- Not specified.

!!! example "Example"
    ```python
    import numpy as np
    from Q_Sea_Battle.player_base_a import PlayerA
    from Q_Sea_Battle.game_layout import GameLayout
    
    layout = GameLayout(...)  # Not specified in this module
    player = PlayerA(layout)
    
    n2 = 100
    field = np.zeros((n2,), dtype=int)
    comms = player.decide(field)
    ```

## Data & State

- `game_layout`: GameLayout, constraints: not specified, shape: not applicable; shared configuration from the Players factory; used for `comms_size`.

## Planned (design-spec)

- None specified.

## Deviations

- None identified.

## Notes for Contributors

- This baseline uses `np.random.randint(0, 2, size=m, dtype=int)`; any changes that affect reproducibility (e.g., RNG seeding) are not specified in this module and should be documented explicitly if introduced.
- Ensure `field` remains a flattened 0/1 vector of shape `(n2,)` if downstream implementations start depending on it; this base class currently ignores `field` and `supp`.

## Related

- `Q_Sea_Battle.game_layout.GameLayout` (provides `comms_size`).

## Changelog

- Version in module docstring: 0.2.