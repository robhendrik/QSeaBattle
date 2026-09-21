# SimplePlayers

> Role: Factory that produces a paired `(PlayerA, PlayerB)` set as `(SimplePlayerA, SimplePlayerB)` sharing a single `GameLayout`.
Location: `Q_Sea_Battle.simple_players.SimplePlayers`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | GameLayout \| None, optional | Optional shared game configuration; if `None`, the base `Players` class creates a default `GameLayout`. |

Preconditions

- Not specified.

Postconditions

- The instance is initialized via `Players.__init__(game_layout)` and has a `game_layout` associated with it (exact state shape/ownership not specified in this module).

Errors

- Not specified.

Example

```python
from Q_Sea_Battle.simple_players import SimplePlayers

factory = SimplePlayers()
player_a, player_b = factory.players()
```

## Public Methods

### players

Instantiate and return the concrete player pair.

Returns

- Tuple[PlayerA, PlayerB], shape (2,): A tuple `(player_a, player_b)` containing `SimplePlayerA` and `SimplePlayerB` instances that share `self.game_layout`.

Preconditions

- Not specified.

Postconditions

- Both returned players reference the same `GameLayout` instance via `self.game_layout`.

Errors

- Not specified.

Example

```python
factory = SimplePlayers()
player_a, player_b = factory.players()
```

## Data & State

- Inherited state from `Players` (not specified in this module), including `self.game_layout: GameLayout` as implied by usage.

## Planned (design-spec)

- None specified.

## Deviations

- None identified between code and provided design notes.

## Notes for Contributors

- The key invariant is that both `SimplePlayerA` and `SimplePlayerB` must be constructed with the same `self.game_layout` object (shared instance), not merely equivalent configurations.

## Related

- `Q_Sea_Battle.players_base.Players`
- `Q_Sea_Battle.simple_player_a.SimplePlayerA`
- `Q_Sea_Battle.simple_player_b.SimplePlayerB`
- `Q_Sea_Battle.game_layout.GameLayout`

## Changelog

- Not specified.