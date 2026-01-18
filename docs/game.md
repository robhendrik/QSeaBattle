# Game

> Role: Orchestrates a single QSeaBattle game between two players by coordinating a `GameEnv` and a `Players` factory.

Location: `Q_Sea_Battle.game.Game`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| `game_env` | `GameEnv`, not specified, shape N/A | Game environment instance. |
| `players` | `Players`, not specified, shape N/A | Players factory providing Player A and B. |

Preconditions

- `game_env` is not `None`.
- `players` is not `None`.
- `game_env` provides methods: `reset()`, `provide()`, `apply_channel_noise(comm)`, `evaluate(shoot)`.
- `players` provides methods: `reset()`, `players()` returning two player objects with method `decide(...)`.

Postconditions

- `self.game_env` references the provided `game_env`.
- `self.players` references the provided `players`.

Errors

- Not specified (constructor contains no explicit validation or exception handling).

Example

```python
from Q_Sea_Battle.game import Game
from Q_Sea_Battle.game_env import GameEnv
from Q_Sea_Battle.players_base import Players

game_env = GameEnv(...)  # Not specified
players = Players(...)   # Not specified

game = Game(game_env=game_env, players=players)
```

## Public Methods

### `play() -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, int]`

Play a single game round by resetting environment and players, obtaining player instances, providing `(field, gun)` from the environment, having Player A produce a communication, applying channel noise, having Player B decide whether to shoot, and evaluating the reward.

Returns

- `reward`: `float`, not specified, shape N/A.
- `field`: `np.ndarray`, not specified, shape not specified (documented as flattened in docstring).
- `gun`: `np.ndarray`, not specified, shape not specified (documented as flattened in docstring).
- `comm`: `np.ndarray`, not specified, shape not specified (returned value is the noisy communication; documented as flattened in docstring).
- `shoot`: `int`, constraints not specified, shape N/A (constructed as `int(shoot)` from Player B decision).

Errors

- Not specified (method contains no explicit exception handling; may propagate exceptions raised by `GameEnv`, `Players`, or player instances).

Example

```python
reward, field, gun, comm, shoot = game.play()
```

## Data & State

- `game_env`: `GameEnv`, not specified, shape N/A; stored reference to the environment used for `reset()`, `provide()`, `apply_channel_noise(...)`, and `evaluate(...)`.
- `players`: `Players`, not specified, shape N/A; stored reference to the players factory used for `reset()` and `players()`.

## Planned (design-spec)

- Not specified.

## Deviations

- The `play()` docstring states it returns `(reward, field, gun, comm, shoot)`, where `comm` refers to the communication; the implementation returns `comm_noisy` (noisy communication) in the `comm` position.

## Notes for Contributors

- `play()` assumes `GameEnv.provide()` returns `(field, gun)` in a format compatible with `player_a.decide(field, supp=None)` and `player_b.decide(gun, comm_noisy, supp=None)`; the precise dtypes/shapes are not enforced in code.
- `shoot` is cast to `int` on return; if `player_b.decide(...)` returns a non-scalar or non-castable object, `int(shoot)` will raise.

## Related

- `Q_Sea_Battle.game_env.GameEnv`
- `Q_Sea_Battle.players_base.Players`

## Changelog

- 0.1: Initial implementation of single-game orchestration via `Game.play()`.