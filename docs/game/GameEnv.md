# Game Env

> **Role**: Environment for a single QSeaBattle game round. Generates `field` and `gun`, provides
> flattened inputs to players, evaluates the reward, and applies channel noise to communication.

**Location**: `Q_Sea_Battle.game_env.GameEnv`

## Constructor

| Parameter     | Type                   | Description                                                           |
| ------------- | ---------------------- | --------------------------------------------------------------------- |
| `game_layout` | `GameLayout` \| `None` | Optional configuration. If `None`, a default `GameLayout` is created. |

**Preconditions**

*   `field_size > 0` and `n2 = field_size^2`.
*   `enemy_probability ∈ [0.0, 1.0]`.
*   `channel_noise ∈ [0.0, 1.0]`.
*   Shapes derived from `GameLayout`: flattened arrays have `shape (n2,)`, dtype `int {0,1}`.
*   For evaluation, `gun` must be one‑hot (exactly one `1`).

**Postconditions**

*   On construction, `game_layout` is set; `field` and `gun` are `None`.
*   After `reset()`, `field` and `gun` are created and ready for `provide()` and `evaluate()`.

## Public Methods

### `reset() -> None`

**Purpose**: Recreate the environment state for a new game.  
**Args**: *None*.  
**Returns**: `None`.  
**Raises**: *None*.

!!! example "Example"
`python
    env.reset()
    `

### `provide() -> tuple[np.ndarray, np.ndarray]`

**Purpose**: Provide flattened inputs to players.  
**Args**: *None*.  
**Returns**:

*   `field`: `np.ndarray`, dtype `int {0,1}`, `shape (n2,)`.
*   `gun`: `np.ndarray`, dtype `int {0,1}`, one‑hot, `shape (n2,)`.  
    Both arrays are **copies** to prevent external mutation.  
    **Raises**: `RuntimeError` if `reset()` has not been called.

!!! note "Returned arrays are copies"
`provide()` returns copies of internal arrays to preserve environment integrity.

!!! example "Example"
`python
    field, gun = env.provide()
    `

### `evaluate(shoot: int) -> float`

**Purpose**: Evaluate Player B’s decision against the true cell value at the gun position.  
**Args**:

*   `shoot` — `int {0,1}`, scalar; `1` means “shoot”, `0` means “do not shoot”.  
    **Returns**: `float {0.0, 1.0}` — reward `1.0` if the decision matches the true cell value, else `0.0`.  
    **Raises**:
*   `RuntimeError` if `reset()` has not been called.
*   `RuntimeError` if `gun` is not strictly one‑hot (internal invariant check).

!!! example "Example"
`python
    reward = env.evaluate(shoot=1)
    `

### `apply_channel_noise(comm: np.ndarray) -> np.ndarray`

**Purpose**: Flip bits in communication independently with probability `channel_noise`.  
**Args**:

*   `comm` — `np.ndarray`, dtype `int {0,1}`, `shape (m,)`.  
    **Returns**: `np.ndarray`, dtype `int {0,1}`, `shape (m,)`.
*   If `channel_noise <= 0.0`: returns an unchanged copy.
*   If `channel_noise >= 1.0`: returns the bitwise complement.  
    **Raises**: *None* (`GameLayout` is expected to validate the range of `channel_noise`).

!!! tip "Channel extremes"
At `channel_noise = 0.0` there is no noise; at `channel_noise = 1.0` all bits are flipped.

!!! example "Example"
`python
    noisy_comm = env.apply_channel_noise(comm)
    `

## Data & State

*   Attributes (public):
    *   `game_layout` — `GameLayout` — immutable configuration reference after construction.
    *   `field` — `np.ndarray | None`, dtype `int {0,1}`, `shape (n2,)` when set — regenerated on `reset()`.
    *   `gun` — `np.ndarray | None`, dtype `int {0,1}`, one‑hot, `shape (n2,)` when set — regenerated on `reset()`.
*   Side effects:
    *   `reset()` mutates internal state (`field`, `gun`).
    *   `evaluate()` and `apply_channel_noise()` are pure with respect to environment state.
*   Thread‑safety:
    *   Not specified; treat `GameEnv` as single‑threaded in typical usage.

## Notes for Contributors

*   **Shapes & symbols**: maintain consistent use of `field_size`, `comms_size`, `n2`, `m`; all external
    I/O is flattened (`shape (n2,)`).
*   **One‑hot gun invariant**: keep the invariant enforced *inside* `evaluate()`; violations must raise.
*   **Copies on provide**: preserve copy‑out semantics to avoid external mutation of internal arrays.
*   **Error discipline**: rely on `GameLayout` for parameter validation (probabilities, divisibility) rather
    than duplicating checks inside `GameEnv`.
*   **Reproducibility**: the design specifies deterministic behavior under a global seed; `GameEnv` uses
    NumPy’s RNG. Ensure the runner sets seeds consistently when constructing tournaments or tests.

## Deviations

*   **Seeding**: the design document states *“All pseudo‑random operations must be deterministic given a global
    seed.”* In this implementation, `GameEnv` does not accept a seed parameter; it uses the global NumPy RNG.
    To meet the spec, seed control should be provided by the calling framework (e.g., test harness, tournament).

## Related

*   See also: `GameLayout` (configuration), `Players` interfaces (`PlayerA`, `PlayerB`), and `Tournament`
    orchestration in the QSeaBattle specification.

## Testing Hooks (suggested invariants)

*   After `reset()`:
    *   `field.shape == (n2,)` and `gun.shape == (n2,)`.
    *   `field.dtype == gun.dtype == int` and values in `{0,1}`.
    *   `gun.sum() == 1` (strict one‑hot).
*   `provide()` returns arrays whose identities differ from internal storage (copies).
*   `apply_channel_noise()` preserves shape/dtype; extreme cases `c<=0` (identity) and `c>=1` (full flip) hold.
*   `evaluate(0)` equals `1.0 - evaluate(1)` at the same state.

!!! warning "Call order"
Always call `reset()` before `provide()` or `evaluate()`; otherwise a `RuntimeError` is raised.

## Changelog

*   2026‑01‑05 — Initial class page · Author: Rob Hendriks
