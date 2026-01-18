# GameEnv

> Role: Core environment for generating game state (field, gun), providing player inputs, evaluating rewards, and applying communication channel noise.
Location: `Q_Sea_Battle.game_env.GameEnv`

## Derived constraints

- Let `field_size = n` from `self.game_layout.field_size`; then `n2 = n * n` and the flattened `field` and `gun` are shape `(n2,)`.
- `field` values are integers in `{0,1}` and shape `(n, n)`.
- `gun` is a one-hot integer array in `{0,1}` with exactly one `1` and shape `(n, n)`.
- Let `comms_size = m`; then `comm` vectors passed to `apply_channel_noise` are intended to have shape `(m,)` with values in `{0,1}` (length is not validated by the implementation).

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | Optional[GameLayout], nullable, shape N/A | Optional game configuration; if `None`, a default `GameLayout()` is constructed. |

Preconditions

- None.

Postconditions

- `self.game_layout` is set to the provided `GameLayout` or a new default instance.
- `self.field` is `None`.
- `self.gun` is `None`.

Errors

- Not specified.

Example

```python
from Q_Sea_Battle.game_env import GameEnv

env = GameEnv()
env.reset()
field_flat, gun_flat = env.provide()
r = env.evaluate(1)
```

## Public Methods

### reset

Signature: `reset(self) -> None`

Reset the environment state for a new game by creating a new random field and a new random one-hot gun position.

Arguments

- None.

Returns

- `None`, shape N/A.

Preconditions

- `self.game_layout.field_size` is expected to be usable as an integer `n` such that arrays of shape `(n, n)` and length `n2 = n * n` can be created.
- `self.game_layout.enemy_probability` is expected to be usable as a float probability for Bernoulli sampling.

Postconditions

- `self.field`: `np.ndarray, dtype int, values {0,1}, shape (field_size, field_size)`.
- `self.gun`: `np.ndarray, dtype int, values {0,1}, shape (field_size, field_size)`, with exactly one element equal to `1`.

Errors

- Not specified.

Example

```python
env.reset()
```

### provide

Signature: `provide(self) -> Tuple[np.ndarray, np.ndarray]`

Provide inputs to the players by returning the current `field` and `gun` arrays in flattened form (copies).

Arguments

- None.

Returns

- `Tuple[np.ndarray, np.ndarray]`: `(field, gun)` where each is `np.ndarray, dtype int, values {0,1}, shape (n2,)` with `n2 = field_size * field_size`.

Preconditions

- `reset()` has been called successfully so that `self.field` and `self.gun` are not `None`.

Postconditions

- Returns copies of internal arrays; mutating the returned arrays does not mutate `self.field` / `self.gun`.

Errors

- `RuntimeError`: If `self.field is None` or `self.gun is None` (environment not reset).

Example

```python
env.reset()
field_flat, gun_flat = env.provide()
```

### evaluate

Signature: `evaluate(self, shoot: int) -> float`

Evaluate the reward for a shooting decision.

Arguments

- `shoot`: `int, values {0,1}, shape N/A`. (The implementation casts via `int(shoot)`; other values are not validated.)

Returns

- `float, values {0.0, 1.0}, shape N/A`: `1.0` if the decision matches the true cell value at the gun position, otherwise `0.0`.

Preconditions

- `reset()` has been called successfully so that `self.field` and `self.gun` are not `None`.
- `self.gun` contains exactly one `1` so that exactly one field cell is selected.

Postconditions

- No mutation of `self.field` or `self.gun` is performed.

Errors

- `RuntimeError`: If `self.field is None` or `self.gun is None` (environment not reset).
- `RuntimeError`: If `self.gun` does not contain exactly one `1` (i.e., selected cell count is not 1).

Example

```python
env.reset()
reward = env.evaluate(0)
```

### apply_channel_noise

Signature: `apply_channel_noise(self, comm: np.ndarray) -> np.ndarray`

Apply independent bit-flip noise to a communication vector with per-bit flip probability `channel_noise`.

Arguments

- `comm`: `np.ndarray, dtype int convertible, intended values {0,1}, intended shape (comms_size,)` where `comms_size = m`. The implementation converts using `np.asarray(comm, dtype=int)` and does not validate values or length.

Returns

- `np.ndarray, dtype int, shape comm.shape`: Noisy communication vector.
- If `channel_noise <= 0.0`, returns an unchanged copy of `comm` (after conversion to `dtype int`).
- If `channel_noise >= 1.0`, returns `1 - comm` (bitwise flip for 0/1 semantics).
- Otherwise flips each entry independently with probability `channel_noise`.

Preconditions

- `self.game_layout.channel_noise` is expected to be usable as a float `c`.

Postconditions

- Returns a new array (copy) for `c <= 0.0` and for `0.0 < c < 1.0`.
- For `c >= 1.0`, the expression `1 - comm` produces a new array.

Errors

- Not specified.

Example

```python
env.reset()
comm = np.array([0, 1, 1, 0], dtype=int)
noisy = env.apply_channel_noise(comm)
```

## Data & State

- `game_layout`: `GameLayout, non-null, shape N/A`. Configuration object used to read `field_size`, `enemy_probability`, and `channel_noise`.
- `field`: `Optional[np.ndarray], nullable`. When set: `np.ndarray, dtype int, values {0,1}, shape (field_size, field_size)`.
- `gun`: `Optional[np.ndarray], nullable`. When set: `np.ndarray, dtype int, values {0,1}, shape (field_size, field_size)`, intended to be one-hot with exactly one `1`.

## Planned (design-spec)

- Not specified.

## Deviations

- No design notes were provided; no deviations can be assessed.

## Notes for Contributors

- `provide()` returns flattened copies; if new methods return views instead, explicitly document mutation and aliasing behavior.
- `apply_channel_noise()` assumes 0/1 semantics but does not validate `comm` contents; if validation is added, document the resulting errors and constraints.

## Related

- `Q_Sea_Battle.game_layout.GameLayout` (configuration dependency; imported as `from .game_layout import GameLayout`).

## Changelog

- Version: 0.1 (module docstring).