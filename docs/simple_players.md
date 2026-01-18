# SimplePlayers

> Role: Factory that produces a matched `(SimplePlayerA, SimplePlayerB)` pair sharing a `GameLayout`.
Location: `Q_Sea_Battle.simple_players.SimplePlayers`

## Constructor

| Parameter | Type | Description |
|---|---|---|
| game_layout | GameLayout \| None, constraint: may be `None` | Optional shared configuration; if `None`, a default `GameLayout` is created by the base class. |

Preconditions

- Not specified.

Postconditions

- `self.game_layout` is initialized via the base `Players` constructor (exact initialization behavior is defined by `Players`).

Errors

- Not specified.

!!! example "Example"
    ```python
    from Q_Sea_Battle.simple_players import SimplePlayers

    factory = SimplePlayers()
    player_a, player_b = factory.players()
    ```

## Public Methods

### players

Create a `(SimplePlayerA, SimplePlayerB)` pair that shares the same `GameLayout`.

Returns

- `Tuple[PlayerA, PlayerB]`, constraint: length exactly `2`, shape: `(2,)` as a fixed-size 2-tuple of `(player_a, player_b)`.

Preconditions

- Not specified.

Postconditions

- Returns two newly created player objects, both constructed with `self.game_layout`.

Errors

- Not specified.

!!! example "Example"
    ```python
    factory = SimplePlayers()
    player_a, player_b = factory.players()
    ```

## Data & State

- `game_layout`: `GameLayout`, constraints: not specified in this module; provided/managed by the base class `Players`.

## Planned (design-spec)

- None specified.

## Deviations

- None identified (no design notes provided beyond empty placeholder).

## Notes for Contributors

- This class delegates `GameLayout` initialization to `Players.__init__`; changes to default layout behavior should be implemented in `Players`, not here.
- The concrete player types are hard-coded as `SimplePlayerA` and `SimplePlayerB` within `players()`.

## Related

- `Q_Sea_Battle.players_base.Players`
- `Q_Sea_Battle.players_base.PlayerA`
- `Q_Sea_Battle.players_base.PlayerB`
- `Q_Sea_Battle.game_layout.GameLayout`
- `Q_Sea_Battle.simple_player_a.SimplePlayerA`
- `Q_Sea_Battle.simple_player_b.SimplePlayerB`

## Changelog

- 0.1: Initial version (module docstring).