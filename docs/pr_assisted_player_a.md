# PRAssistedPlayerA

> Role: Player A implementation using PR-assisted resources to iteratively compress a binary field into a single communication bit.
Location: `Q_Sea_Battle.pr_assisted_player_a.PRAssistedPlayerA`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | GameLayout, constraints: instance of `Q_Sea_Battle.game_layout.GameLayout`, shape: N/A | Game configuration, used to derive $n2 = \text{field\_size}^2$ via `game_layout.field_size`. |
| parent | PRAssistedPlayers, constraints: instance of `Q_Sea_Battle.pr_assisted_players.PRAssistedPlayers`, shape: N/A | Owning factory providing access to PR-assisted boxes via `parent.pr_assisted(level)`. |

Preconditions

- `parent` is an instance of `Q_Sea_Battle.pr_assisted_players.PRAssistedPlayers`.

Postconditions

- `self.game_layout` is initialized by `PlayerA.__init__(game_layout)` (exact state not specified in this module).
- `self.parent` is set to the provided `parent`.

Errors

- `TypeError`: raised if `parent` is not a `PRAssistedPlayers` instance.

!!! example "Example"
    ```python
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.pr_assisted_players import PRAssistedPlayers
    from Q_Sea_Battle.pr_assisted_player_a import PRAssistedPlayerA

    game_layout = GameLayout(...)  # not specified in this module
    parent = PRAssistedPlayers(...)  # not specified in this module
    player_a = PRAssistedPlayerA(game_layout=game_layout, parent=parent)
    ```

## Public Methods

### decide(field, supp=None)

Compute the communication bit from the field by repeatedly compressing the 1D binary field using PR-assisted resources at increasing `level` until a single bit remains.

Parameters

- field: np.ndarray, dtype int {0,1}, shape (n2,) where $n2 = \text{field\_size}^2$; constraints: `field.ndim == 1`, `field.shape[0] == n2`, and values are only 0/1 (checked after `np.asarray(field, dtype=int)` coercion).
- supp: Any | None, constraints: unused and ignored, shape: N/A.

Returns

- np.ndarray, dtype int {0,1}, shape (1,); constraints: a 1D NumPy array of length 1 containing the computed communication bit.

Preconditions

- `self.game_layout.field_size` is defined and supports exponentiation and integer multiplication such that $n2 = \text{field\_size}^2$ is a valid integer.
- The PR-assisted resources returned by `self.parent.pr_assisted(level)` provide a method `measurement_a(measurement)` that accepts `measurement: np.ndarray, dtype int {0,1}, shape (m,)` and returns an array indexable as `outcome_a[k]` for `k in [0, m)`; exact type/shape constraints are not specified in this module but must be compatible with the algorithm.

Postconditions

- The returned array contains `comm_bit = int(intermediate_field[0])` where `intermediate_field` is the final compressed field of length 1.
- `supp` has no effect on the output.

Errors

- `ValueError`: raised if `field` is not 1D or its length is not $n2$.
- `ValueError`: raised if `field` contains values other than 0/1 (after conversion to `dtype=int`).
- `ValueError`: raised if an intermediate compression level produces an odd-length `intermediate_field` (i.e., `intermediate_field.size % 2 != 0`).
- `RuntimeError`: raised if the final `intermediate_field` does not have length 1 (defensive check).

!!! example "Example"
    ```python
    import numpy as np
    from Q_Sea_Battle.pr_assisted_player_a import PRAssistedPlayerA

    # player_a constructed elsewhere
    field_size = player_a.game_layout.field_size
    n2 = field_size ** 2
    field = np.zeros((n2,), dtype=int)
    comm = player_a.decide(field)
    assert comm.shape == (1,)
    assert comm.dtype == int
    ```

## Data & State

- parent: PRAssistedPlayers, constraints: instance of `Q_Sea_Battle.pr_assisted_players.PRAssistedPlayers`, shape: N/A; set in the constructor and used by `decide()` to access PR-assisted boxes via `self.parent.pr_assisted(level)`.
- game_layout: GameLayout, constraints: instance of `Q_Sea_Battle.game_layout.GameLayout`, shape: N/A; inherited from `PlayerA` and used by `decide()` to compute $n2 = \text{field\_size}^2$ via `self.game_layout.field_size`.

## Planned (design-spec)

- Not specified (no design notes provided).

## Deviations

- Not specified (no design notes provided).

## Notes for Contributors

- `decide()` coerces `field` using `np.asarray(field, dtype=int)` before validating values; callers passing non-integer numeric types may have their inputs truncated before validation.
- The compression loop requires that the current length be even at every level; since the initial length is $n2 = \text{field\_size}^2$, `field_size` must be such that repeated halving eventually reaches 1 without producing an odd length; this property is not validated up front.
- The PR-assisted box interface is used implicitly (`measurement_a`); changes to `PRAssistedPlayers.pr_assisted()` or the returned box object must preserve compatibility with this call pattern.

## Related

- `Q_Sea_Battle.players_base.PlayerA`
- `Q_Sea_Battle.game_layout.GameLayout`
- `Q_Sea_Battle.pr_assisted_players.PRAssistedPlayers`

## Changelog

- 0.1: Initial version (module docstring indicates PR-assisted naming update and internal rename from "shared randomness" to "PR-assisted").