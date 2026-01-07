# Tournament

> **Role**: Runs a multi-game QSeaBattle tournament by repeatedly executing `Game.play()` and logging results.

**Location**: `Q_Sea_Battle.tournament.Tournament`

!!! note "Derived constraints from GameLayout"
    Let `field_size = n` and `n2 = n**2`. Let `comms_size = m`.

    - `n2` is a power of two.
    - `m | n2` (i.e. `n2 % m == 0`).

    These constraints are validated by `GameLayout` during construction.

## Constructor

| Parameter | Type | Description |
|---|---|---|
| game_env | `GameEnv`, scalar | Environment instance reused across games.|
| players | `Players`, scalar | Player factory reused across games.|
| game_layout | `GameLayout`, scalar | Configuration used to determine tournament length.|

**Preconditions**

- `game_env` is a `GameEnv`, scalar.
- `players` is a `Players`, scalar.
- `game_layout` is a `GameLayout`, scalar.
- `game_layout.number_of_games_in_tournament` is an `int` and `> 0`.

**Postconditions**

- `self.game_env`, `self.players`, and `self.game_layout` reference the provided objects.

**Errors**

- No explicit exceptions are raised by the constructor.
- Attribute errors may propagate if invalid objects are passed.

!!! example "Example"
    ```python
    from Q_Sea_Battle.game_env import GameEnv
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.players_base import Players
    from Q_Sea_Battle.tournament import Tournament

    layout = GameLayout(field_size=8, comms_size=4, number_of_games_in_tournament=100)
    env = GameEnv(layout)
    players = Players(layout)

    t = Tournament(game_env=env, players=players, game_layout=layout)
    log = t.tournament()
    ```

## Public Methods

### tournament

**Signature**

- `tournament() -> TournamentLog`
**Purpose**

Execute `number_of_games_in_tournament` games, update a `TournamentLog` each round, and return the log.

**Arguments**

- None.

**Returns**

- `log`: `TournamentLog`, scalar.
  - Contains one entry per game with core outcomes and optional auxiliary fields.

**Preconditions**

- `Game(self.game_env, self.players)` construction succeeds.
- `Game.play()` returns `(reward, field, gun, comm, shoot)` with:
  - `reward`: `float` {0.0, 1.0}, scalar.
  - `field`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)`.
  - `gun`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)`, one-hot.
  - `comm`: `np.ndarray`, dtype `int` {0,1}, shape `(m,)`.
  - `shoot`: `int` {0,1}, scalar.

- `TournamentLog` supports the update methods used:
  - `update(field, gun, comm, shoot, cell_value, reward) -> None`.
  - `update_log_probs(logprob_comm, logprob_shoot) -> None` (optional path).
  - `update_log_prev(prev_meas, prev_out) -> None` (optional path).
  - `update_indicators(game_id, tournament_id, meta_id) -> None`.

**Postconditions**

- A `TournamentLog` is created with `game_layout` and updated exactly `n_games` times.
- For each `game_id` in `0..n_games-1`:
  - One call to `Game.play()` is executed.
  - `cell_value` is derived from `field` at the one-hot `gun` index:
    - `cell_value`: `int` {0,1}, scalar.
  - `TournamentLog.update_indicators(...)` is called with:
    - `game_id`: `int`, scalar.
    - `tournament_id`: `int`, scalar (currently fixed to `0`).
    - `meta_id`: `int`, scalar (currently fixed to `0`).

- Optional logging paths:
  - If `getattr(self.players, "has_log_probs", False)` is true:
    - calls `player_a.get_log_prob() -> float`, scalar.
    - calls `player_b.get_log_prob() -> float`, scalar.
    - records them via `TournamentLog.update_log_probs(...)`.
  - If `getattr(self.players, "has_prev", False)` is true:
    - calls `player_a.get_prev() -> tuple[prev_meas, prev_out] | None`.
    - if not `None`, records via `TournamentLog.update_log_prev(prev_meas, prev_out)`.

**Errors**

- Any exception raised by:
  - `TournamentLog(...)` construction or any of its update methods.
  - `Game(...)` construction or `Game.play()` execution.
  - Optional player hooks (`get_log_prob`, `get_prev`) if the flags are enabled.

!!! tip "Optional hooks are convention-based"
    `has_log_probs`, `has_prev`, `get_log_prob()`, and `get_prev()` are accessed dynamically via `getattr` and are
    not enforced by the base `Players` interface.

## Data & State

- Attributes (public):
  - `game_env`: `GameEnv`, scalar.
  - `players`: `Players`, scalar.
  - `game_layout`: `GameLayout`, scalar.

- Side effects:
  - Runs `n_games` game rounds.
  - Mutates `game_env` and player internal state via their `reset`/`decide` logic (indirectly through `Game.play()`).
  - Mutates the returned `TournamentLog` via repeated updates.

- Thread-safety:
  - Not thread-safe (reuses mutable environment, players, and log in a loop).

## Planned (design-spec)

- Tournament identifiers:
  - The implementation currently sets `tournament_id = 0` and `meta_id = 0` for all games.
  - Design may plan richer identifier handling (e.g., multiple tournaments / meta experiments).

!!! note "Design-spec alignment"
    This page documents only APIs present in the code. Any additional tournament controls present in the design
    document but not implemented should be added here under this section once identified.

## Deviations

- Configuration ownership:
  - Implementation requires `game_layout` as an explicit constructor argument and also carries `game_env`, which itself
    holds a `game_layout` internally.
  - A design might treat `GameEnv.game_layout` as the single source of truth.

- Optional logging interfaces:
  - Implementation uses dynamic attributes (`has_log_probs`, `has_prev`) and assumes methods exist on concrete players
    (`get_log_prob`, `get_prev`).
  - The base `Players` interface does not declare these hooks explicitly.

## Notes for Contributors

- Keep the tournament loop minimal and deterministic with respect to control flow.
- If you introduce new per-game metadata (e.g., unique `tournament_id` or `meta_id`), ensure the values are logged
  consistently and document their type and semantics.
- Consider making optional logging hooks explicit via a protocol or base-class methods to reduce reliance on `getattr`.
- Avoid allocating large intermediate arrays per game; update the log incrementally as currently implemented.

## Related

- See also: `Game` (single-round orchestration), `TournamentLog` (result accumulator), `GameEnv` (environment).

## Changelog

- 2026-01-07 — Author: Rob Hendriks — Initial specification page for `Tournament`.
