# Players

> Role: Factory and container for a pair of QSeaBattle players that share a common `GameLayout`.
Location: `Q_Sea_Battle.players_base.Players`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | Optional[GameLayout], constraints: may be None, shape: scalar object | Optional shared configuration. If None, a default `GameLayout` is created. |

Preconditions

- None specified.

Postconditions

- `self.game_layout` is set to the provided `game_layout`, or to a newly created default `GameLayout` if `game_layout` is None.

Errors

- None specified.

Example

```python
from Q_Sea_Battle.players_base import Players

players_factory = Players()
player_a, player_b = players_factory.players()
```

## Public Methods

### players

Create the concrete Player A and Player B instances using the shared `GameLayout`.

Returns

- Tuple["PlayerA", "PlayerB"], constraints: length 2 tuple, shape: (2,) of objects: A tuple `(player_a, player_b)` created as `_PlayerA(self.game_layout)` and `_PlayerB(self.game_layout)`.

Errors

- Not specified.

Example

```python
from Q_Sea_Battle.players_base import Players

players_factory = Players()
player_a, player_b = players_factory.players()
```

### reset

Reset any internal state across both players.

Returns

- None, constraints: always None, shape: scalar.

Errors

- Not specified.

Example

```python
from Q_Sea_Battle.players_base import Players

players_factory = Players()
players_factory.reset()
```

## Data & State

- `game_layout`: GameLayout, constraints: non-None after construction, shape: scalar object; shared configuration used by both players.

## Planned (design-spec)

- Not specified.

## Deviations

- The module exposes deprecated attribute names `PlayerA` and `PlayerB` via `__getattr__` for backward compatibility, but this class specification documents only `Players` as the single top-level public class as required.

## Notes for Contributors

- `Players.players()` currently instantiates `_PlayerA` and `_PlayerB` imported from `Q_Sea_Battle.players_base_a` and `Q_Sea_Battle.players_base_b`; subclasses may override `players()` to supply alternative implementations while reusing `self.game_layout`.
- `reset()` is intentionally a no-op in the base implementation to support compatibility with subclasses that maintain state.

## Related

- `Q_Sea_Battle.players_base_a.PlayerA` (concrete baseline implementation, preferred import path)
- `Q_Sea_Battle.players_base_b.PlayerB` (concrete baseline implementation, preferred import path)
- `Q_Sea_Battle.game_layout.GameLayout`

## Changelog

- 0.2: `Players` provided as the stable public facade for constructing a pair of players sharing a `GameLayout`; `PlayerA`/`PlayerB` names retained in the module via deprecation mechanism.