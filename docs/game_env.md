# GameEnv

> Role: Lightweight, single-instance QSeaBattle environment that samples a binary enemy field and one-hot query location, provides flattened observations, evaluates Player B’s binary decision, and optionally applies bit-flip channel noise.

Location: `Q_Sea_Battle.game_env.GameEnv`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | Optional[GameLayout], constraints: None or GameLayout instance, shape: scalar | Optional game configuration; if `None`, constructs a default `GameLayout()` and stores it in `self.game_layout`. |

Preconditions

- `game_layout` is `None` or a `GameLayout` instance.

Postconditions

- `self.game_layout`: GameLayout, shape: scalar, is set to `game_layout` or a newly constructed `GameLayout()`.
- `self.field`: Optional[np.ndarray], dtype Unknown (uninitialized), shape: scalar optional, is set to `None`.
- `self.gun`: Optional[np.ndarray], dtype Unknown (uninitialized), shape: scalar optional, is set to `None`.

Errors

- Not specified.

Example

```python
from Q_Sea_Battle.game_env import GameEnv
from Q_Sea_Battle.game_layout import GameLayout

env = GameEnv(GameLayout())
env.reset()
field_flat, gun_flat = env.provide()
reward = env.evaluate(1)
```

## Public Methods

### reset

Reset the environment state for a new game by sampling a new random `field` and a new random one-hot `gun` position.

Signature

- `reset(self) -> None`

Arguments

- None.

Returns

- `None`, constraints: always `None`, shape: scalar.

Preconditions

- `self.game_layout.field_size` is used as `n` (constraint not specified in code).
- `self.game_layout.enemy_probability` is used as Bernoulli parameter `p` (constraint not specified in code).

Postconditions

- `self.field`: np.ndarray, dtype int, values in `{0,1}`, shape `(n, n)`.
- `self.gun`: np.ndarray, dtype int, values in `{0,1}` with exactly one `1`, shape `(n, n)`.

Errors

- Not specified.

### provide

Provide flattened copies of the internal `field` and `gun` arrays.

Signature

- `provide(self) -> Tuple[np.ndarray, np.ndarray]`

Arguments

- None.

Returns

- `Tuple[np.ndarray, np.ndarray]`: `(field, gun)` where `field` is `np.ndarray, dtype int, constraints: copy of internal field values in {0,1}, shape (n2,)` and `gun` is `np.ndarray, dtype int, constraints: copy of internal gun values in {0,1}, shape (n2,)`, with $n2 = n \cdot n$.

Errors

- `RuntimeError`: If `self.field is None` or `self.gun is None` (environment not reset).

### evaluate

Evaluate Player B’s shooting decision against the true `field` value at the one-hot `gun` location.

Signature

- `evaluate(self, shoot: int) -> float`

Arguments

- `shoot`: int, constraints: intended to be binary `{0,1}` but cast with `int(shoot)`, shape: scalar.

Returns

- `float`, constraints: returns `1.0` if `int(shoot)` equals the selected cell value, else `0.0`, shape: scalar.

Preconditions

- Environment has been reset (`self.field` and `self.gun` are not `None`).
- `self.gun` contains exactly one `1` (enforced).

Errors

- `RuntimeError`: If `self.field is None` or `self.gun is None` (environment not reset).
- `RuntimeError`: If `self.gun` does not contain exactly one `1`.

### apply_channel_noise

Apply independent bit-flip noise to a communication vector using `self.game_layout.channel_noise`.

Signature

- `apply_channel_noise(self, comm: np.ndarray) -> np.ndarray`

Arguments

- `comm`: np.ndarray, dtype Unknown (converted via `np.asarray(comm, dtype=int)`), constraints: convertible to integer array, shape: arbitrary `S` (same shape returned).

Returns

- `np.ndarray`, dtype int, constraints: same shape as input `comm` after conversion, each element possibly flipped $0 \leftrightarrow 1$ according to `channel_noise`, shape: `S`.

Behavior

- Converts `comm` to `np.ndarray, dtype int`.
- Let `c = float(self.game_layout.channel_noise)`.
- If `c <= 0.0`, returns an unchanged copy.
- If `c >= 1.0`, returns `1 - comm` (deterministic full flip).
- Otherwise, flips each element independently with probability `c`.

Errors

- Not specified.

## Data & State

- `game_layout`: GameLayout, constraints: instance of `GameLayout`, shape: scalar; used to obtain `field_size`, `enemy_probability`, and `channel_noise`.
- `field`: Optional[np.ndarray], dtype int when set, constraints: values in `{0,1}`, shape `(n, n)`; `None` before `reset()`.
- `gun`: Optional[np.ndarray], dtype int when set, constraints: values in `{0,1}` with exactly one `1`, shape `(n, n)`; `None` before `reset()`.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- `provide()` intentionally returns flattened copies to prevent external mutation of internal state; preserve this copy semantics if refactoring.
- `evaluate()` enforces the one-hot property of `gun` at runtime; if future changes alter `gun` representation, update both selection logic and validation accordingly.
- `apply_channel_noise()` treats `channel_noise <= 0.0` and `>= 1.0` as special cases; maintain these branches for clarity and determinism.

## Related

- `Q_Sea_Battle.game_layout.GameLayout`

## Changelog

- Not specified.