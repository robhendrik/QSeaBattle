# AssistedPlayerB

> Role: Player B implementation that decides to shoot using hierarchical shared randomness outcomes plus a communication bit.

Location: `Q_Sea_Battle.Archive.assisted_player_b.AssistedPlayerB`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | GameLayout, constraints: not specified, shape: not applicable | Game configuration passed to the base `PlayerB` constructor. |
| parent | AssistedPlayers, constraints: must be an instance of `AssistedPlayers`, shape: not applicable | Owning factory providing access to shared randomness boxes; validated with `isinstance`. |

Preconditions: `parent` must be an instance of `AssistedPlayers`.  
Postconditions: `self.parent` is set to `parent`; `PlayerB` base initialization is completed with `game_layout`.  
Errors: Raises `TypeError` if `parent` is not an `AssistedPlayers` instance.  
Example:

!!! example "Construct an AssistedPlayerB"
    ```python
    from Q_Sea_Battle.Archive.assisted_player_b import AssistedPlayerB
    # from Q_Sea_Battle.Archive.assisted_players import AssistedPlayers
    # from Q_Sea_Battle.Archive.game_layout import GameLayout
    # game_layout = GameLayout(...)
    # parent = AssistedPlayers(...)
    # player_b = AssistedPlayerB(game_layout=game_layout, parent=parent)
    ```

## Public Methods

### decide(gun, comm, supp=None)

Decide whether to shoot by tracing a one-hot `gun` vector through successive pairwise reductions, querying one shared randomness box per level, collecting one outcome bit per level at the active pair index, then XOR-ing (parity) all collected bits with the communication bit.

Parameters:

- `gun`: np.ndarray, dtype int {0,1}, shape (n2,), constraints: must be 1D; length must equal $n2 = \mathrm{field\_size}^2$; must be one-hot (sum equals 1).
- `comm`: np.ndarray, dtype int {0,1}, shape (1,), constraints: must be 1D; length must equal 1.
- `supp`: Any | None, constraints: unused (deleted), shape: not applicable.

Returns:

- int, constraints: in {0,1}; 1 means shoot, 0 means do not shoot; shape: scalar.

Errors:

- Raises `ValueError` if `gun` is not 1D of length `n2`, contains values not in {0,1}, or is not one-hot.
- Raises `ValueError` if `comm` is not 1D of length 1 or contains values not in {0,1}.
- Raises `ValueError` during processing if `intermediate_gun` has odd length at any level, ceases to be one-hot, has zero or more than one active pair (pair equal to (0,1) or (1,0)), or if `measurement` (a.k.a. measurement string) has sum not in {0,1}.

Example:

!!! example "Decide with one-hot gun and one-bit comm"
    ```python
    import numpy as np
    # player_b = AssistedPlayerB(game_layout=..., parent=...)
    # n2 = player_b.game_layout.field_size ** 2
    # gun = np.zeros(n2, dtype=int)
    # gun[0] = 1
    # comm = np.array([1], dtype=int)
    # shoot = player_b.decide(gun=gun, comm=comm)
    # assert shoot in (0, 1)
    ```

## Data & State

- `parent`: AssistedPlayers, constraints: set at construction and must satisfy `isinstance(parent, AssistedPlayers)` at initialization time; shape: not applicable.
- `game_layout`: GameLayout, constraints: provided via base class `PlayerB`; shape: not applicable (exact storage and invariants in `PlayerB` are not specified in this module).

## Planned (design-spec)

Not specified.

## Deviations

Not specified.

## Notes for Contributors

- This module performs a local import of `AssistedPlayers` inside `__init__` to avoid import cycles; if refactoring imports, preserve cycle-free initialization behavior.
- The method enforces strict one-hot invariants at every reduction level; any changes to the shared randomness protocol must keep these checks aligned with the intended specification.

## Related

- `Q_Sea_Battle.Archive.game_layout.GameLayout`
- `Q_Sea_Battle.Archive.players_base.PlayerB`
- `Q_Sea_Battle.Archive.assisted_players.AssistedPlayers` (imported locally inside `__init__`)

## Changelog

- 0.1: Initial version (per module docstring).