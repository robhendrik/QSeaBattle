# Tournament

> Role: Run a multi-game QSeaBattle tournament by repeatedly executing `Game.play()` and recording outcomes in a `TournamentLog`.
Location: `Q_Sea_Battle.tournament.Tournament`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_env | `GameEnv`, constraints: instance of `Q_Sea_Battle.game_env.GameEnv`, shape: N/A | Game environment instance reused across games. |
| players | `Players`, constraints: instance of `Q_Sea_Battle.players_base.Players`, shape: N/A | Players factory/provider for player A and B; reused across games. |
| game_layout | `GameLayout`, constraints: instance of `Q_Sea_Battle.game_layout.GameLayout`, shape: N/A | Configuration specifying tournament length (used to determine number of games). |

Preconditions: `game_env`, `players`, and `game_layout` are non-`None` objects compatible with downstream calls in `tournament()` (e.g., `Game(self.game_env, self.players)` and `TournamentLog(self.game_layout)` must be constructible).

Postconditions: `self.game_env`, `self.players`, and `self.game_layout` are set to the provided instances.

Errors: Not specified (any exceptions raised by attribute access or object construction will propagate).

!!! example "Example"
    ```python
    from Q_Sea_Battle.tournament import Tournament
    from Q_Sea_Battle.game_env import GameEnv
    from Q_Sea_Battle.players_base import Players
    from Q_Sea_Battle.game_layout import GameLayout
    
    game_env = GameEnv()
    players = Players()
    game_layout = GameLayout()
    
    t = Tournament(game_env=game_env, players=players, game_layout=game_layout)
    log = t.tournament()
    ```

## Public Methods

### tournament

Execute a full tournament and return its log.

Parameter: None.

Returns: `TournamentLog`, constraints: instance of `Q_Sea_Battle.tournament_log.TournamentLog`, shape: N/A; contains results for all games played in the tournament.

Side effects: Constructs `TournamentLog` and `Game`; repeatedly calls `Game.play()`; updates `TournamentLog` with per-game results and optional extra data when available from `players`.

Preconditions: `self.game_layout.number_of_games_in_tournament` exists and is usable as the `range()` bound (i.e., an `int` or `__index__`-compatible type). `Game(self.game_env, self.players).play()` returns a 5-tuple `(reward, field, gun, comm, shoot)` such that `gun == 1` is a valid boolean mask over `field` and `field[gun == 1][0]` exists. `TournamentLog(self.game_layout)` supports `update(...)`, `update_indicators(...)`, and optionally `update_log_probs(...)` and `update_log_prev(...)` depending on `players` capabilities.

Postconditions: The returned `TournamentLog` has been updated once per game with: base game outcome (via `update`), identifiers (via `update_indicators`), and optionally log-probabilities (via `update_log_probs`) and previous measurement/outcome data (via `update_log_prev`).

Errors: Not specified; exceptions from `Game.play()`, numpy-like indexing operations, `players` methods (e.g., `players.players()`, `get_log_prob()`, `get_prev()`), or `TournamentLog` update methods may propagate.

## Data & State

- `game_env`: `GameEnv`, constraints: instance of `Q_Sea_Battle.game_env.GameEnv`, shape: N/A; stored reference used for constructing a `Game`.
- `players`: `Players`, constraints: instance of `Q_Sea_Battle.players_base.Players`, shape: N/A; stored reference used for constructing a `Game` and optional per-game logging (`has_log_probs`, `has_prev`).
- `game_layout`: `GameLayout`, constraints: instance of `Q_Sea_Battle.game_layout.GameLayout`, shape: N/A; stored reference used to create `TournamentLog` and determine `n_games` via `number_of_games_in_tournament`.

## Planned (design-spec)

Not specified.

## Deviations

Not specified.

## Notes for Contributors

- The implementation uses fixed identifiers `tournament_id = 0` and `meta_id = 0` for all games; comments indicate these may be extended later.
- Optional logging is enabled via `getattr(self.players, "has_log_probs", False)` and `getattr(self.players, "has_prev", False)`; when present, the code assumes child players implement `get_log_prob()` (for A and B) and `get_prev()` (for A).
- `cell_value` is derived as `int(field[gun == 1][0])`; ensure `gun` contains at least one element equal to `1` and that the masking semantics are valid for the `field`/`gun` types returned by `Game.play()`.

## Related

- `Q_Sea_Battle.game.Game`
- `Q_Sea_Battle.tournament_log.TournamentLog`
- `Q_Sea_Battle.game_env.GameEnv`
- `Q_Sea_Battle.game_layout.GameLayout`
- `Q_Sea_Battle.players_base.Players`

## Changelog

- 0.1: Initial version (module header indicates Version: 0.1).