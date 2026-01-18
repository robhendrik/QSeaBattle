# PlayerB

> Role: Baseline B-side player that decides whether to shoot; default implementation returns a random binary decision.

Location: `Q_Sea_Battle.players_base_b.PlayerB`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | GameLayout, constraints: not specified, shape: N/A | Shared game configuration for this player; stored on the instance as `game_layout`. |

Preconditions

- `game_layout` is a `GameLayout` instance (additional constraints not specified).

Postconditions

- `self.game_layout` is set to the provided `game_layout`.

Errors

- Not specified.

!!! example "Example"
    ```python
    from Q_Sea_Battle.players_base_b import PlayerB
    from Q_Sea_Battle.game_layout import GameLayout

    game_layout = GameLayout()  # construction details not specified here
    player_b = PlayerB(game_layout=game_layout)
    ```

## Public Methods

### decide(gun, comm, supp=None)

Decide whether to shoot based on gun position and a communication vector; the base implementation ignores inputs and returns a random decision in {0, 1}.

Parameters

- `gun`: np.ndarray, dtype not specified, constraints: "flattened one-hot" (implies binary entries), shape not specified.
- `comm`: np.ndarray, dtype not specified, constraints not specified, shape not specified.
- `supp`: Any | None, constraints: optional, shape: N/A; unused in the base class.

Returns

- `int`, constraints: in {0, 1}, shape: scalar.

Errors

- Not specified.

!!! note "Behavior"
    The implementation returns `int(np.random.randint(0, 2))`, ignoring `gun`, `comm`, and `supp`.

!!! example "Example"
    ```python
    import numpy as np
    from Q_Sea_Battle.players_base_b import PlayerB
    from Q_Sea_Battle.game_layout import GameLayout

    player_b = PlayerB(GameLayout())
    gun = np.zeros(10, dtype=int)   # shape/dimensions are game-dependent
    comm = np.zeros(5, dtype=int)   # shape/dimensions are game-dependent
    action = player_b.decide(gun=gun, comm=comm)
    assert action in (0, 1)
    ```

## Data & State

- `game_layout`: GameLayout, constraints: not specified, shape: N/A; shared configuration provided at construction time.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- This is a baseline implementation; `decide` currently ignores all inputs and uses NumPy RNG to return a random binary action.

## Related

- `Q_Sea_Battle.game_layout.GameLayout`

## Changelog

- 0.2: Initial baseline B-side player interface with random shooting decision (as indicated by module docstring version).