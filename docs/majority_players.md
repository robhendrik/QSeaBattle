# MajorityPlayers

> Role: Factory that creates a paired `(MajorityPlayerA, MajorityPlayerB)` sharing a single `GameLayout` instance.
Location: `Q_Sea_Battle.majority_players.MajorityPlayers`

## Constructor

Parameter | Type | Description
--- | --- | ---
game_layout | `GameLayout | None`, constraints: if not `None` must be a valid `GameLayout` instance, shape: N/A | Shared game configuration; if `None`, the base class creates a default `GameLayout`.

Preconditions

- Not specified.

Postconditions

- `self.game_layout` is set (created by the base class if `game_layout is None`).

Errors

- Not specified.

Example

```python
from Q_Sea_Battle.majority_players import MajorityPlayers

factory = MajorityPlayers()
player_a, player_b = factory.players()
```

## Public Methods

### players

Creates and returns a `(player_a, player_b)` pair that share this factory's `GameLayout`.

Signature

- `players(self) -> Tuple[PlayerA, PlayerB]`

Parameters

- None.

Returns

- `Tuple[PlayerA, PlayerB]`, constraints: 2-tuple `(player_a, player_b)`, shape: `(2,)` where `player_a` is an instance of `MajorityPlayerA` (a `PlayerA`) and `player_b` is an instance of `MajorityPlayerB` (a `PlayerB`).

Preconditions

- `self.game_layout` exists and is a `GameLayout` instance (established by the base class constructor).

Postconditions

- Returns two newly constructed player objects.
- Both returned players reference the exact same `GameLayout` instance (`self.game_layout`).

Errors

- Not specified.

Example

```python
factory = MajorityPlayers()
player_a, player_b = factory.players()

assert player_a.game_layout is player_b.game_layout
```

## Data & State

- `game_layout`: `GameLayout`, constraints: not `None` after construction, shape: N/A; provided by the `Players` base class and shared by all created players.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- The `players()` method must preserve identity sharing: both players must be constructed with the same `self.game_layout` object (not a copy).

## Related

- `Q_Sea_Battle.players_base.Players`
- `Q_Sea_Battle.majority_player_a.MajorityPlayerA`
- `Q_Sea_Battle.majority_player_b.MajorityPlayerB`
- `Q_Sea_Battle.game_layout.GameLayout`

## Changelog

- Not specified.