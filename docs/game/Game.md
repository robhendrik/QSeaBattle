# Game

> **Role**: Orchestrates a single QSeaBattle game round between Player A and Player B.

**Location**: `Q_Sea_Battle.game.Game`

!!! note "Derived constraints from GameLayout"
    Let `field_size = n` and `n2 = n**2`. Let `comms_size = m`.

    - `n2` is a power of two.
    - `m | n2` (i.e. `n2 % m == 0`).

    These constraints are enforced by `GameLayout`, which is held by `GameEnv`.

## Constructor

| Parameter | Type | Description |
|---|---|---|
| game_env | GameEnv, scalar | Environment providing field generation, channel noise, and reward evaluation. |
| players | Players, scalar | Players factory providing Player A and Player B instances. |

**Preconditions**

- `game_env` is an instance of `GameEnv`.
- `players` is an instance of `Players`.
- Both objects are fully initialised and ready to be reset.

**Postconditions**

- `self.game_env` references the provided environment.
- `self.players` references the provided players factory.

**Errors**

- No explicit exceptions are raised by the constructor.
- Attribute errors may propagate if invalid objects are passed.

!!! example "Example"
    ```python
    from Q_Sea_Battle.game import Game
    from Q_Sea_Battle.game_env import GameEnv
    from Q_Sea_Battle.players_base import Players

    game = Game(game_env=GameEnv(), players=Players(...))
    ```

## Public Methods

### play

**Signature**

- `play() -> tuple[float, np.ndarray, np.ndarray, np.ndarray, int]`

**Purpose**

Run a single game round following the fixed QSeaBattle protocol:
reset, communicate, shoot, and evaluate.

**Arguments**

- None.

**Returns**

- `(reward, field, gun, comm, shoot)` where:
  - `reward`: `float` {0.0, 1.0}, scalar.
  - `field`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)`.
  - `gun`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)`, one-hot.
  - `comm`: `np.ndarray`, dtype `int` {0,1}, shape `(m,)` (after channel noise).
  - `shoot`: `int` {0,1}, scalar.

**Preconditions**

- `self.game_env.reset()` succeeds.
- `self.players.reset()` succeeds.
- `self.players.players()` returns `(player_a, player_b)` with compatible interfaces:
  - `player_a.decide(field, supp=None) -> np.ndarray, shape (m,)`.
  - `player_b.decide(gun, comm, supp=None) -> int {0,1}`.

**Postconditions**

- Exactly one full game round has been executed.
- Internal state of `game_env` reflects the last reset state.
- Player internal state has advanced by one round.

**Errors**

- Any exception raised by:
  - `GameEnv.reset`, `provide`, `apply_channel_noise`, or `evaluate`.
  - `Players.reset`, `Players.players`, or `Player.decide`.
- No errors are caught or transformed by `Game.play`.

!!! example "Example"
    ```python
    reward, field, gun, comm, shoot = game.play()
    ```

## Data & State

- Attributes (public):
  - `game_env`: `GameEnv`, scalar.
  - `players`: `Players`, scalar.

- Side effects:
  - Calls `reset()` on both environment and players.
  - Advances internal player state by one decision round.

- Thread-safety:
  - Not thread-safe. Uses mutable environment and player state.

## Planned (design-spec)

- None. The implemented orchestration sequence matches the design specification
  for a single QSeaBattle game round. fileciteturn3file0

## Deviations

- Return value ordering:
  - Design document refers to returning `(reward, field, gun, comm, shoot)`.
  - Implementation returns `(reward, field, gun, comm_noisy, int(shoot))`,
    i.e. the *noisy* communication vector and an explicit `int` cast for `shoot`. fileciteturn3file0

- Player reset responsibility:
  - Design implies player reset as part of game start.
  - Implementation explicitly calls `self.players.reset()` inside `play()`,
    rather than delegating this to a higher-level controller. fileciteturn3file0

## Notes for Contributors

- Keep `Game` free of learning or optimisation logic; it should only orchestrate calls.
- Do not add randomness here; randomness belongs in `GameEnv` or player implementations.
- If the game protocol changes (e.g. additional communication rounds),
  update this class first and treat it as the single source of truth for sequencing.

## Related

- See also: `GameEnv` (environment), `Players` (player factory), `Tournament` (multi-game orchestration).

## Changelog

- {date.today().isoformat()} — Author: Rob Hendriks — Initial specification page for `Game`.
