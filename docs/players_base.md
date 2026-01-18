# Players

> Role: Factory and container for a pair of QSeaBattle players.
Location: `Q_Sea_Battle.players_base.Players`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | Optional[GameLayout], constraint: None or instance of `Q_Sea_Battle.game_layout.GameLayout`, shape: scalar | Optional shared configuration; if None, a default `GameLayout` is created. |

Preconditions

- Not specified.

Postconditions

- `self.game_layout` is set to the provided `game_layout` if not None, otherwise to a newly created `GameLayout`.

Errors

- Not specified.

Example

```python
from Q_Sea_Battle.players_base import Players

players_facade = Players()
player_a, player_b = players_facade.players()
```

## Public Methods

### players

Create the concrete Player A and Player B instances using the shared `GameLayout`.

Parameters

- None.

Returns

- Tuple["PlayerA", "PlayerB"], constraint: 2-tuple of player instances, shape: (2,).

Errors

- Not specified.

Example

```python
from Q_Sea_Battle.players_base import Players

p = Players()
player_a, player_b = p.players()
```

### reset

Reset any internal state across both players.

Parameters

- None.

Returns

- None, constraint: always `None`, shape: scalar.

Errors

- Not specified.

Example

```python
from Q_Sea_Battle.players_base import Players

p = Players()
p.reset()
```

## Data & State

- game_layout: GameLayout, constraint: instance of `Q_Sea_Battle.game_layout.GameLayout`, shape: scalar; shared configuration used by both players.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- This module also provides deprecated attribute access for `PlayerA` and `PlayerB` via module-level `__getattr__`, emitting `DeprecationWarning` and caching the resolved symbols in `globals()`.

## Related

- `Q_Sea_Battle.player_base_a.PlayerA` (concrete baseline implementation imported as `_PlayerA` internally)
- `Q_Sea_Battle.player_base_b.PlayerB` (concrete baseline implementation imported as `_PlayerB` internally)
- `Q_Sea_Battle.game_layout.GameLayout`

## Changelog

- Not specified.