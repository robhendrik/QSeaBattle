# Game

> Role: Orchestrates a single sequential QSeaBattle episode by coordinating environment resets, observation sampling, Player A communication, channel noise, Player B action, and reward evaluation.
Location: `Q_Sea_Battle.game.Game`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_env | GameEnv, constraints: instance of `.game_env.GameEnv`, shape: N/A | Environment instance providing observations, channel noise, and reward evaluation. |
| players | Players, constraints: instance of `.players_base.Players`, shape: N/A | Factory providing the two player instances and supporting reset between games. |

Preconditions

- `game_env` must provide `reset()`, `provide()`, `apply_channel_noise(comm)`, and `evaluate(shoot)` methods.
- `players` must provide `reset()` and `players()` methods returning two player objects with `decide(...)`.
- Observation and communication objects are expected to be NumPy arrays, but dtype/shape constraints are not specified.

Postconditions

- `self.game_env` references the provided `game_env`.
- `self.players` references the provided `players`.

Errors

- Not specified; any exceptions raised by `game_env` or `players` methods may propagate.

!!! example "Example"
    ```python
    from Q_Sea_Battle.game import Game
    
    game = Game(game_env=game_env, players=players)
    ```

## Public Methods

### `play()`

Run a single game and return its outcome.

Control flow

- Resets the environment and players.
- Instantiates concrete Player A and Player B via `self.players.players()`.
- Obtains `(field, gun)` observations from `self.game_env.provide()`.
- Player A computes `comm = player_a.decide(field, supp=None)`.
- Environment applies channel noise: `comm_noisy = self.game_env.apply_channel_noise(comm)`.
- Player B computes `shoot = player_b.decide(gun, comm_noisy, supp=None)`.
- Environment evaluates reward: `reward = self.game_env.evaluate(shoot)`.
- Returns `(reward, field, gun, comm_noisy, int(shoot))`.

Returns

- `reward`: float, constraints: not specified, shape: scalar.
- `field`: np.ndarray, dtype: not specified, constraints: flattened field observation, shape: not specified.
- `gun`: np.ndarray, dtype: not specified, constraints: flattened gun observation, shape: not specified.
- `comm_noisy`: np.ndarray, dtype: not specified, constraints: communication after channel noise, shape: not specified.
- `shoot`: int, constraints: cast from Player B action, shape: scalar.

Preconditions

- `self.game_env.reset()` and `self.players.reset()` must be callable.
- `self.players.players()` must return `(player_a, player_b)`.
- `player_a.decide(field, supp=None)` must accept `supp=None` and return a value acceptable to `self.game_env.apply_channel_noise`.
- `player_b.decide(gun, comm_noisy, supp=None)` must accept `supp=None` and produce an action acceptable to `self.game_env.evaluate`.
- `self.game_env.provide()` must return exactly two values `(field, gun)`.

Postconditions

- The environment has been reset and evaluated once for the produced action.
- The players have been reset and each queried once for a decision.

Errors

- Not specified; any exceptions raised by `reset`, `players`, `provide`, `decide`, `apply_channel_noise`, or `evaluate` may propagate.

!!! example "Example"
    ```python
    reward, field, gun, comm_noisy, shoot = game.play()
    ```

## Data & State

- `game_env`: GameEnv, constraints: assigned from constructor argument, shape: N/A.
- `players`: Players, constraints: assigned from constructor argument, shape: N/A.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- The orchestrator assumes `provide()` returns `(field, gun)` in that order and that both are already flattened; if this changes in `GameEnv`, update `Game.play()` accordingly.
- Player A is called with `decide(field, supp=None)` while Player B is called with `decide(gun, comm_noisy, supp=None)`; keep this asymmetric interface consistent with the `Players`/player implementations.

## Related

- `Q_Sea_Battle.game_env.GameEnv`
- `Q_Sea_Battle.players_base.Players`

## Changelog

- Not specified.