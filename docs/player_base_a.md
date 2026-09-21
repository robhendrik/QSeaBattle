# PlayerA

> Role: Baseline A-side player that produces a random binary communication vector for a given game layout.
Location: `Q_Sea_Battle.player_base_a.PlayerA`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | `GameLayout`, constraints: not specified, shape: N/A | Game configuration for this player. |

Preconditions

- `game_layout`: `GameLayout`, constraints: not specified, shape: N/A.

Postconditions

- `self.game_layout` is set to the provided `game_layout`.

Errors

- Not specified.

!!! example "Example"
    ```python
    from Q_Sea_Battle.player_base_a import PlayerA
    from Q_Sea_Battle.game_layout import GameLayout
    
    game_layout = GameLayout()  # constructor signature not specified here
    player = PlayerA(game_layout=game_layout)
    ```

## Public Methods

### decide

Return a communication vector based on the current field; the base implementation ignores inputs and returns a random 0/1 vector of length $m = \text{game\_layout.comms\_size}$.

Parameters

- `field`: `np.ndarray`, dtype: not specified, constraints: flattened field array, shape: not specified.
- `supp`: `Optional[Any]`, constraints: optional supporting information (unused), shape: N/A.

Returns

- `np.ndarray`, dtype `int`, constraints: values in `{0,1}`, shape `(m,)` where $m = \text{game\_layout.comms\_size}$.

Errors

- Not specified.

!!! example "Example"
    ```python
    import numpy as np
    from Q_Sea_Battle.player_base_a import PlayerA
    
    # Assume `game_layout` exists and provides `comms_size`.
    player = PlayerA(game_layout)
    
    field = np.zeros((10 * 10,), dtype=int)  # shape is illustrative; not enforced by PlayerA
    comms = player.decide(field=field)
    assert comms.shape == (game_layout.comms_size,)
    assert set(np.unique(comms)).issubset({0, 1})
    ```

## Data & State

- `game_layout`: `GameLayout`, constraints: not specified, shape: N/A; game configuration provided at construction time.

## Planned (design-spec)

- None specified.

## Deviations

- None identified.

## Notes for Contributors

- Subclasses typically override `decide` to implement learned or rule-based strategies; the baseline implementation does not use `field` or `supp`.

## Related

- `Q_Sea_Battle.game_layout.GameLayout`

## Changelog

- Not specified.