# GameEnv

> **Role**: Runtime environment for a single QSeaBattle game round (field generation, gun position, and reward evaluation).

**Location**: `Q_Sea_Battle.game_env.GameEnv`

!!! note "Derived constraints from GameLayout"
    Let `field_size = n` and `n2 = n**2`. Let `comms_size = m`.

    - `n2` is a power of two.
    - `m | n2` (i.e. `n2 % m == 0`).

    These constraints are validated by `GameLayout` at construction time.

## Constructor

| Parameter | Type | Description |
|---|---|---|
| game_layout | GameLayout or None, scalar | Game configuration. If None, constructs `GameLayout()` with defaults. |

**Preconditions**

- If `game_layout` is provided, it is a `GameLayout` instance (already validated).
- If `game_layout` is None, `GameLayout()` construction succeeds.

**Postconditions**

- `self.game_layout` is set.
- `self.field` is None.
- `self.gun` is None.

**Errors**

- Any exception raised by `GameLayout()` (when `game_layout` is None) is propagated.

!!! example "Example"
    ```python
    from Q_Sea_Battle.game_env import GameEnv
    from Q_Sea_Battle.game_layout import GameLayout

    env = GameEnv(GameLayout(field_size=8, comms_size=4))
    env.reset()
    field, gun = env.provide()
    ```

## Public Methods

### reset

**Signature**

- `reset() -> None`

**Purpose**

Reset internal state for a new game by generating a new random `field` and a new random one-hot `gun`.

**Preconditions**

- `self.game_layout.field_size` is an `int`, `field_size > 0`.
- `self.game_layout.enemy_probability` is a `float` in `[0.0, 1.0]`.

**Postconditions**

- `self.field` is set to an array sampled i.i.d. Bernoulli(`enemy_probability`):
  - `np.ndarray`, dtype `int` {0,1}, shape `(field_size, field_size)`.
- `self.gun` is set to a one-hot array over all `n2` positions:
  - `np.ndarray`, dtype `int` {0,1}, shape `(field_size, field_size)`.
  - Exactly one entry equals `1` and the rest are `0`.

**Errors**

- No explicit exceptions are raised by this method.
- Exceptions from NumPy random generation may propagate (rare; typically only in misconfigured environments).

!!! tip "Reproducibility"
    The implementation uses NumPy’s global RNG (`np.random`). Use `np.random.seed(...)` in tests to make
    `reset()` deterministic.

### provide

**Signature**

- `provide() -> tuple[np.ndarray, np.ndarray]`

**Purpose**

Provide inputs to the players as flattened copies of the current `field` and `gun`.

**Arguments**

- None.

**Returns**

- `(field, gun)` where:
  - `field`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)`.
  - `gun`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)`, one-hot.

**Preconditions**

- `reset()` has been called for the current game:
  - `self.field` is not None and `self.gun` is not None.

**Postconditions**

- The returned arrays are **copies**; mutating them does not mutate internal state.

**Errors**

- Raises `RuntimeError` if the environment has not been reset yet.

!!! example "Example"
    ```python
    env.reset()
    field, gun = env.provide()
    ```

### evaluate

**Signature**

- `evaluate(shoot: int) -> float`

**Purpose**

Evaluate Player B’s shooting decision against the true `field` value at the one-hot `gun` index.

**Arguments**

- `shoot`: `int` {0,1}, scalar. (The implementation casts with `int(shoot)`.)

**Returns**

- `reward`: `float` {0.0, 1.0}, scalar.
  - `1.0` if `shoot` equals the true cell value.
  - `0.0` otherwise.

**Preconditions**

- `reset()` has been called for the current game:
  - `self.field` is not None and `self.gun` is not None.
- `self.gun` is one-hot (exactly one `1`).

**Postconditions**

- No state changes (pure evaluation with respect to the current `field`/`gun`).

**Errors**

- Raises `RuntimeError` if the environment has not been reset yet.
- Raises `RuntimeError` if `self.gun` does not contain exactly one `1`.

!!! example "Example"
    ```python
    env.reset()
    field, gun = env.provide()
    reward = env.evaluate(shoot=0)
    ```

### apply_channel_noise

**Signature**

- `apply_channel_noise(comm: np.ndarray) -> np.ndarray`

**Purpose**

Apply a binary symmetric channel to the communication vector by flipping each bit independently with probability
`channel_noise`.

**Arguments**

- `comm`: `np.ndarray`, dtype `int` {0,1}, shape `(m,)`.

**Returns**

- `noisy_comm`: `np.ndarray`, dtype `int` {0,1}, shape `(m,)`.

**Preconditions**

- `self.game_layout.channel_noise` is a `float` in `[0.0, 1.0]`.
- Intended input: `comm` contains only `0/1` values and has length `m`.

**Postconditions**

- If `channel_noise <= 0.0`, returns `comm.copy()`.
- If `channel_noise >= 1.0`, returns `1 - comm`.
- Otherwise flips each bit with probability `channel_noise`.

**Errors**

- No explicit exceptions are raised by this method.
- NumPy broadcasting rules apply for non-1D inputs; callers should pass `(m,)` to avoid unintended behavior.

!!! example "Example"
    ```python
    comm_noisy = env.apply_channel_noise(comm)
    ```

## Data & State

- Attributes (public):
  - `game_layout`: `GameLayout`, scalar.
  - `field`: `np.ndarray` or None.
    - When set: dtype `int` {0,1}, shape `(field_size, field_size)`.
  - `gun`: `np.ndarray` or None.
    - When set: dtype `int` {0,1}, shape `(field_size, field_size)`, one-hot.

- Side effects:
  - `reset()` mutates `field` and `gun`.
  - Other methods do not mutate state.

- Thread-safety:
  - Not thread-safe for concurrent use (mutable state; uses NumPy global RNG).

## Planned (design-spec)

- None. The class methods listed in the design are implemented: `reset`, `provide`, `evaluate`, `apply_channel_noise`.

## Deviations

- `evaluate(shoot)` input type:
  - Design spec describes `shoot` as boolean-like (True/False).
  - Implementation type-hints `shoot: int` and casts via `int(shoot)` (accepting bool and numeric scalars).

- Input validation:
  - Design spec implicitly treats `comm` as length `m` and bits in `{0,1}`.eciteturn2file14
  - Implementation converts `comm` to `np.asarray(..., dtype=int)` and does not validate shape `(m,)` or value set `{0,1}`.

## Notes for Contributors

- Keep `provide()` returning **copies** to prevent external mutation of internal environment state.
- If you add stricter input validation (e.g., enforcing `comm.shape == (m,)` or `shoot in {0,1}`),
  document the new `Errors` precisely and update dependent tests.
- For deterministic tests:
  - Seed NumPy RNG (`np.random.seed(...)`) before calling `reset()` and `apply_channel_noise(...)`.
- Avoid adding tournament-level logic here; `GameEnv` should remain a single-round environment.

## Related

- See also: `GameLayout` (configuration), `Game` (single-round orchestration), `Tournament` (multi-round loop).

## Changelog

- 2026-01-07 — Author: Rob Hendriks — Initial specification page for `GameEnv`.
