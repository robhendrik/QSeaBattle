# MajorityPlayerA

> Role: Player A that encodes segment-wise majority information from a flattened binary field into a deterministic communication vector.

Location: `Q_Sea_Battle.majority_player_a.MajorityPlayerA`

## Derived constraints

- Let field_size be `self.game_layout.field_size` (int, not specified), then $n2 = \text{field\_size}^2$ (int).
- Let comms_size be `self.game_layout.comms_size` (int, not specified), then $m = \text{comms\_size}$ (int).
- Segment length is $\text{segment\_len} = n2 // m$ (int); the implementation assumes $m$ divides $n2$ (comment: enforced by `GameLayout`).

## Constructor

| Parameter | Type | Description |
|---|---|---|
| game_layout | GameLayout, constraints: not specified, shape: N/A | Game configuration for this player; stored/used via the `PlayerA` base class and accessed as `self.game_layout`. |

Preconditions

- `game_layout` is a `GameLayout` instance (not validated here).
- Additional constraints are not specified in this class.

Postconditions

- The instance is initialized by delegating to `PlayerA.__init__(game_layout)`.

Errors

- Not specified (any exceptions would come from `PlayerA.__init__` or invalid `game_layout` usage elsewhere).

!!! example "Example"
    ```python
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.majority_player_a import MajorityPlayerA
    
    layout = GameLayout(field_size=4, comms_size=4)  # exact signature not specified here
    player = MajorityPlayerA(layout)
    ```

## Public Methods

### decide(field, supp=None)

Encode majority statistics in the communication vector.

Parameters

- field: np.ndarray, dtype: any (converted to int), constraints: interpreted as 0/1 values by documentation but not validated, shape: any (flattened internally to 1D via `.ravel()`).
- supp: Optional[Any], constraints: unused, shape: N/A.

Returns

- np.ndarray, dtype int, constraints: values in {0, 1}, shape (m,), where $m = \text{comms\_size}$.

Behavior

- Converts `field` to `flat_field = np.asarray(field, dtype=int).ravel()`.
- Computes $n2 = \text{field\_size}^2$ and $m = \text{comms\_size}$ from `self.game_layout`.
- Sets `segment_len = n2 // m`.
- For each segment $i \in [0, m)$, slices `flat_field[start:end]` where `start = i * segment_len` and `end = start + segment_len`, then sets `comm[i] = 1` if `ones >= zeros` else `0`, with `ones = int(segment.sum())` and `zeros = segment_len - ones`.
- Returns the resulting `comm`.

Preconditions

- `self.game_layout.field_size` and `self.game_layout.comms_size` exist and are usable as integers.
- The implementation assumes $m$ divides $n2$.
- `field` must contain at least `n2` elements after flattening; otherwise segment slices may be shorter than `segment_len`, affecting majority computation (not checked).

Postconditions

- Returns a newly allocated vector `comm` of length `m`.

Errors

- May raise exceptions from `np.asarray(..., dtype=int)` for non-coercible inputs.
- May raise `ZeroDivisionError` if `m == 0` (not prevented here).
- May raise attribute errors if `self.game_layout` is missing required attributes.

!!! example "Example"
    ```python
    import numpy as np
    from Q_Sea_Battle.majority_player_a import MajorityPlayerA
    from Q_Sea_Battle.game_layout import GameLayout
    
    layout = GameLayout(field_size=4, comms_size=4)  # exact signature not specified here
    player = MajorityPlayerA(layout)
    
    field = np.array([
        1, 0, 1, 0,
        0, 0, 1, 1,
        1, 1, 0, 0,
        0, 1, 0, 1,
    ])
    comm = player.decide(field)
    ```

## Data & State

- Inherited state from `PlayerA` (not defined in this module).
- Uses `self.game_layout` (type: GameLayout, constraints: not specified, shape: N/A) as established by the base class.
- No additional attributes are defined by `MajorityPlayerA`.

## Planned (design-spec)

- Not specified (no design notes provided).

## Deviations

- No deviations identified (no design notes provided to compare against code).

## Notes for Contributors

- The method `decide` computes `n2` from `field_size` rather than from the actual length of `field`; if `field` does not match the expected size, behavior is not validated and may silently produce incorrect segment statistics.
- Majority tie-breaking is implemented as `1` when `ones >= zeros`.

## Related

- `Q_Sea_Battle.players_base.PlayerA` (base class; behavior not documented here).
- `Q_Sea_Battle.game_layout.GameLayout` (provides `field_size` and `comms_size`).

## Changelog

- 0.1: Initial implementation (module docstring version).