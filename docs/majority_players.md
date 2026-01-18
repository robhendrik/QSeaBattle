# MajorityPlayers

> Role: Factory that produces a `(MajorityPlayerA, MajorityPlayerB)` pair sharing a common `GameLayout`.
Location: `Q_Sea_Battle.majority_players.MajorityPlayers`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | GameLayout \| None, optional | Shared configuration for both players; if `None`, a default `GameLayout` is created by the base class. |

Preconditions

- Not specified.

Postconditions

- `self.game_layout` is available for subsequent `players()` calls (exact initialization behavior is delegated to `Players.__init__`).

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

Create a `(MajorityPlayerA, MajorityPlayerB)` pair that shares the factory's `GameLayout`.

**Signature:** `players(self) -> Tuple[PlayerA, PlayerB]`

Returns

- Tuple[PlayerA, PlayerB], length 2, shape (2,): `(player_a, player_b)` where `player_a` is a `MajorityPlayerA` and `player_b` is a `MajorityPlayerB`; both are constructed with the same `self.game_layout`.

Preconditions

- `self.game_layout` is set (provided by the `Players` base class).

Postconditions

- Two new player instances are created and returned.

Errors

- Not specified.

Example

```python
from Q_Sea_Battle.majority_players import MajorityPlayers
from Q_Sea_Battle.game_layout import GameLayout

layout = GameLayout()
factory = MajorityPlayers(game_layout=layout)
a, b = factory.players()
```

## Data & State

- `game_layout`: GameLayout, constraints unknown, shape N/A; stored on the instance via `Players.__init__` and passed into constructed players in `players()`.

## Planned (design-spec)

- None specified.

## Deviations

- None identified.

## Notes for Contributors

- `MajorityPlayers` delegates `game_layout` initialization to `Players.__init__`; update this documentation if base-class behavior changes (e.g., if defaults or validation are added there).
- `players()` currently always constructs `MajorityPlayerA` and `MajorityPlayerB` with `self.game_layout`; any future support for alternative player types should be reflected here.

## Related

- `Q_Sea_Battle.players_base.Players`
- `Q_Sea_Battle.majority_player_a.MajorityPlayerA`
- `Q_Sea_Battle.majority_player_b.MajorityPlayerB`
- `Q_Sea_Battle.game_layout.GameLayout`

## Changelog

- 0.1: Initial version (module docstring).