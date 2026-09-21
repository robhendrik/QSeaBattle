# PlayerB

> Role: Baseline B-side player policy that returns a random binary shoot/no-shoot decision.

Location: `Q_Sea_Battle.player_base_b.PlayerB`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | GameLayout, constraints Not specified, shape Not applicable | Shared game configuration for this player instance; stored as `self.game_layout`. |

Preconditions

- `game_layout` is a `GameLayout` instance.

Postconditions

- `self.game_layout` is set to the provided `game_layout`.

Errors

- Not specified.

!!! example "Example"
    ```python
    from Q_Sea_Battle.player_base_b import PlayerB
    from Q_Sea_Battle.game_layout import GameLayout

    layout = GameLayout()  # construction args not specified here
    player_b = PlayerB(game_layout=layout)
    ```

## Public Methods

### decide

Return Player B's shoot / no-shoot decision by sampling a uniform random action in $\{0, 1\}$; all inputs are ignored in this baseline implementation.

Parameters

- `gun`: np.ndarray, dtype Not specified, constraints Not specified, shape Not specified; gun position encoding (typically a flattened one-hot array); ignored.
- `comm`: np.ndarray, dtype Not specified, constraints Not specified, shape Not specified; communication vector from Player A; ignored.
- `supp`: Optional[Any], constraints Not specified, shape Not applicable; optional supporting information; unused.

Returns

- int, constraints in {0,1}, shape scalar; `0` for "do not shoot" or `1` for "shoot".

Errors

- Not specified.

!!! example "Example"
    ```python
    import numpy as np
    from Q_Sea_Battle.player_base_b import PlayerB
    from Q_Sea_Battle.game_layout import GameLayout

    player_b = PlayerB(GameLayout())
    gun = np.zeros((10,), dtype=int)
    comm = np.zeros((5,), dtype=float)

    action = player_b.decide(gun=gun, comm=comm)
    assert action in (0, 1)
    ```

## Data & State

- `game_layout`: GameLayout, constraints Not specified, shape Not applicable; shared game configuration provided at construction time.

## Planned (design-spec)

- Not specified.

## Deviations

- None detected between code and provided design notes.

## Notes for Contributors

- `decide` currently ignores `gun`, `comm`, and `supp` and uses `np.random.randint(0, 2)`; changing this behavior will affect baseline reproducibility expectations if any external tests assume randomness.
- Consider injecting a RNG or seeding strategy if deterministic behavior becomes necessary; no such mechanism exists in the current code.

## Related

- `Q_Sea_Battle.players_base` (mentioned as a legacy import context in the module docstring).
- `Q_Sea_Battle.game_layout.GameLayout`

## Changelog

- Not specified.