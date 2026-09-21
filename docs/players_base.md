# Players

> Role: Factory/container that holds a shared `GameLayout` and constructs paired Player A and Player B instances.

Location: `Q_Sea_Battle.players_base.Players`

## Constructor

Parameter | Type | Description
--- | --- | ---
game_layout | Optional[GameLayout], shape () | Shared configuration for both players; if `None`, a default `GameLayout` is created.

Preconditions

- `game_layout` is `None` or an instance of `Q_Sea_Battle.game_layout.GameLayout`, shape ().

Postconditions

- `self.game_layout` is a `GameLayout`, shape (), equal to `game_layout` if provided; otherwise a newly created default `GameLayout`.

Errors

- Not specified.

Example

```python
from Q_Sea_Battle.players_base import Players
from Q_Sea_Battle.game_layout import GameLayout

players = Players()  # uses default GameLayout
custom = Players(GameLayout())
```

## Public Methods

### players

Create Player A and Player B instances sharing the container's `game_layout`.

Parameters

- None.

Returns

- Tuple["PlayerA", "PlayerB"], shape (2,): Tuple `(player_a, player_b)` constructed as `_PlayerA(self.game_layout)` and `_PlayerB(self.game_layout)`.

Errors

- Not specified.

Example

```python
from Q_Sea_Battle.players_base import Players

container = Players()
player_a, player_b = container.players()
```

### reset

Reset any container-level state.

Parameters

- None.

Returns

- None, shape (): Always returns `None`.

Errors

- Not specified.

Example

```python
from Q_Sea_Battle.players_base import Players

container = Players()
container.reset()
```

## Data & State

- `game_layout`: GameLayout, shape (): Shared configuration used by both players.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

!!! note "Deprecated legacy names are handled at module level"
    This module also defines a module-level `__getattr__(name: str) -> Any` that provides deprecated access to `PlayerA` and `PlayerB` with a `DeprecationWarning` and caches the resolved symbol in `globals()`. This is not part of the `Players` class API but affects public imports from `Q_Sea_Battle.players_base`.

## Related

- `Q_Sea_Battle.game_layout.GameLayout`
- `Q_Sea_Battle.player_base_a.PlayerA` (imported internally as `_PlayerA`)
- `Q_Sea_Battle.player_base_b.PlayerB` (imported internally as `_PlayerB`)
- Module-level deprecated accessors: `Q_Sea_Battle.players_base.PlayerA`, `Q_Sea_Battle.players_base.PlayerB` via `__getattr__`

## Changelog

- Not specified.