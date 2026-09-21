# Tournament

> Role: Run a multi-game QSeaBattle tournament by repeatedly executing games and appending per-game artifacts and metadata into a `TournamentLog`.
Location: `Q_Sea_Battle.tournament.Tournament`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_env | `GameEnv`, constraints: instance compatible with `Game(self.game_env, self.players)`; shape: N/A | Game environment used to execute games. |
| players | `Players`, constraints: instance compatible with `Game(self.game_env, self.players)` and optionally exposing attributes `has_log_probs: bool` and/or `has_prev: bool`; shape: N/A | Factory/container providing player A and player B instances. |
| game_layout | `GameLayout`, constraints: must provide attribute `number_of_games_in_tournament: int`; shape: N/A | Layout/configuration specifying the number of games and other tournament-level settings. |

Preconditions

- `game_layout.number_of_games_in_tournament` is an `int` and is iterable via `range(n_games)`.
- `Game(game_env, players)` is constructible.
- `Game.play()` returns a 5-tuple `(reward, field, gun, comm, shoot)` compatible with downstream indexing and logging.
- `field` and `gun` support boolean masking `field[gun == 1]` and yield at least one element; the implementation assumes exactly one gun cell is marked with `1`.

Postconditions

- `self.game_env`, `self.players`, and `self.game_layout` reference the constructor arguments without modification.
- The constructed instance is ready to execute `tournament()`.

Errors

- Not specified; any exceptions raised by `Game`, `TournamentLog`, player methods (`get_log_prob`, `get_prev`), indexing (`field[gun == 1][0]`), or log update methods may propagate.

Example

```python
from Q_Sea_Battle.tournament import Tournament
from Q_Sea_Battle.game_env import GameEnv
from Q_Sea_Battle.players_base import Players
from Q_Sea_Battle.game_layout import GameLayout

t = Tournament(game_env=GameEnv(), players=Players(), game_layout=GameLayout())
log = t.tournament()
```

## Public Methods

### tournament

Execute the tournament and return the accumulated log.

For each game, the runner calls `Game.play()`, computes `cell_value = int(field[gun == 1][0])`, and records the main outputs via `TournamentLog.update(...)`. Optional player-provided metadata is recorded when available using feature flags on `self.players`.

Parameters

- None.

Returns

- `TournamentLog`, constraints: instance returned from `TournamentLog(self.game_layout)` updated for each game; shape: N/A.

Side effects

- Instantiates a `TournamentLog` and a `Game`.
- Mutates the `TournamentLog` via calls to: `update`, optionally `update_log_probs`, optionally `update_log_prev`, and `update_indicators`.

Optional metadata behavior

- If `getattr(self.players, "has_log_probs", False)` is truthy: calls `player_a.get_log_prob()` and `player_b.get_log_prob()` (where `player_a, player_b = self.players.players()`) and passes results to `log.update_log_probs(logprob_comm, logprob_shoot)`.
- If `getattr(self.players, "has_prev", False)` is truthy: calls `player_a.get_prev()` (where `player_a, _ = self.players.players()`); if non-`None`, expects a pair `(prev_meas, prev_out)` and passes it to `log.update_log_prev(prev_meas, prev_out)`.

Errors

- `IndexError` if `field[gun == 1]` is empty (e.g., no gun cell marked with `1`).
- Type/attribute errors if `players.players()`, `get_log_prob()`, `get_prev()`, or `TournamentLog` update methods are missing or return incompatible values.
- Any exceptions raised by `Game.play()` or downstream log methods may propagate.

Example

```python
log = t.tournament()
```

## Data & State

- `game_env`: `GameEnv`, constraints: stored reference; shape: N/A.
- `players`: `Players`, constraints: stored reference; shape: N/A.
- `game_layout`: `GameLayout`, constraints: stored reference; shape: N/A.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- `cell_value` is derived via `int(field[gun == 1][0])`; if the representation of `gun` changes (e.g., multiple active cells), this selection rule must be revisited alongside the logging schema.
- The feature flags `players.has_log_probs` and `players.has_prev` are accessed via `getattr(..., False)`; this intentionally tolerates `Players` implementations that do not define these attributes.

## Related

- `Q_Sea_Battle.game.Game`
- `Q_Sea_Battle.game_env.GameEnv`
- `Q_Sea_Battle.game_layout.GameLayout`
- `Q_Sea_Battle.players_base.Players`
- `Q_Sea_Battle.tournament_log.TournamentLog`

## Changelog

- Not specified.