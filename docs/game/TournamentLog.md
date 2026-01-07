# TournamentLog

> **Role**: Structured accumulator for per-game tournament results and simple aggregate statistics.

**Location**: `Q_Sea_Battle.tournament_log.TournamentLog`

!!! note "Derived constraints from GameLayout"
    Let `field_size = n` and `n2 = n**2`. Let `comms_size = m`.

    - `n2` is a power of two.
    - `m | n2` (i.e. `n2 % m == 0`).

    These constraints are validated by `GameLayout` during construction and determine standard shapes logged per game.

## Constructor

| Parameter | Type | Description |
|---|---|---|
| game_layout | `GameLayout`, scalar | Layout providing the log column names.|

**Preconditions**

- `game_layout` is a valid `GameLayout`, scalar.
- `game_layout.log_columns` is a `list[str]` defining DataFrame column names.

**Postconditions**

- `self.game_layout` is set to `game_layout`. 
- `self.log` is an empty `pd.DataFrame` with columns `game_layout.log_columns`.

**Errors**

- Exceptions raised by `pandas.DataFrame(...)` may propagate (e.g., invalid column labels).

!!! example "Example"
    ```python
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.tournament_log import TournamentLog

    layout = GameLayout()
    log = TournamentLog(layout)
    ```

## Public Methods

### update

**Signature**

- `update(field, gun, comm, shoot, cell_value, reward) -> None`

**Purpose**

Append one game result row to the log, initialising optional fields to `None`.

**Arguments**

- `field`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)`.
- `gun`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)`, one-hot.
- `comm`: `np.ndarray`, dtype `int` {0,1}, shape `(m,)`.
- `shoot`: `int` {0,1}, scalar.
- `cell_value`: `int` {0,1}, scalar.
- `reward`: `float` {0.0, 1.0}, scalar.

**Returns**

- `None`.

**Preconditions**

- The provided arrays follow the standard shapes derived from `game_layout`.
- `gun` is one-hot (exactly one `1`) if `cell_value` is expected to be consistent with `field`.  
  (Consistency is not checked by `TournamentLog` itself.)

**Postconditions**

- Exactly one new row is appended to `self.log`.
- The appended row contains:
  - the provided inputs (`field`, `gun`, `comm`, `shoot`, `cell_value`, `reward`),
  - plus initial placeholders:
    - `logprob_comm = None`, `logprob_shoot = None`,
    - `game_id = None`, `tournament_id = None`, `meta_id = None`, `game_uid = None`,
    - `prev_measurements = None`, `prev_outcomes = None`.

**Errors**

- May raise exceptions from Pandas assignment if the DataFrame is misconfigured or the values are incompatible.
  (No explicit validation is performed.)

### update_log_probs

**Signature**

- `update_log_probs(logprob_comm: float, logprob_shoot: float) -> None`

**Purpose**

Update log-probability fields for the **last** logged game.

**Arguments**

- `logprob_comm`: `float`, scalar.
- `logprob_shoot`: `float`, scalar.

**Returns**

- `None`.

**Preconditions**

- The log contains at least one row.

**Postconditions**

- For the last row:
  - `logprob_comm` is set to `float(logprob_comm)`.
  - `logprob_shoot` is set to `float(logprob_shoot)`.

**Errors**

- Raises `RuntimeError` if the log is empty (via `_last_row_index`).

### update_log_prev

**Signature**

- `update_log_prev(prev_meas: Any, prev_out: Any) -> None` 

**Purpose**

Update previous-measurement and previous-outcome fields for the **last** logged game.

**Arguments**

- `prev_meas`: `Any`, scalar or array-like.
  - Typical: per-shared-layer measurements; shape depends on player architecture.
- `prev_out`: `Any`, scalar or array-like.
  - Typical: per-shared-layer outcomes; shape depends on player architecture.

**Returns**

- `None`.

**Preconditions**

- The log contains at least one row.

**Postconditions**

- For the last row:
  - `prev_measurements` is set to `prev_meas`.
  - `prev_outcomes` is set to `prev_out`.

**Errors**

- Raises `RuntimeError` if the log is empty (via `_last_row_index`).

### update_indicators

**Signature**

- `update_indicators(game_id: int, tournament_id: int, meta_id: int) -> None`

**Purpose**

Update identifier fields for the **last** logged game and generate a unique `game_uid`.

**Arguments**

- `game_id`: `int`, scalar.
- `tournament_id`: `int`, scalar.
- `meta_id`: `int`, scalar.

**Returns**

- `None`.

**Preconditions**

- The log contains at least one row.

**Postconditions**

- For the last row:
  - `game_id`, `tournament_id`, `meta_id` are set to their `int(...)` values.
  - `game_uid` is set to a new UUID4 hex string (`str`, scalar).

**Errors**

- Raises `RuntimeError` if the log is empty (via `_last_row_index`).

### outcome

**Signature**

- `outcome() -> tuple[float, float]`

**Purpose**

Compute aggregate tournament performance statistics from the `reward` column.

**Arguments**

- None.

**Returns**

- `(mean_reward, std_error)` where:
  - `mean_reward`: `float`, scalar.
  - `std_error`: `float`, scalar.
    - Standard error of the mean computed as sample std (`ddof=1`) divided by `sqrt(n)` for `n > 1`.

!!! note "Empty or singleton logs"
    - If the log is empty, returns `(0.0, 0.0)`.
    - If the log has exactly one row, returns `(mean_reward, 0.0)`.

**Preconditions**

- None (handles empty logs).

**Postconditions**

- No mutation of `self.log`.

**Errors**

- Pandas / NumPy conversion errors may propagate if the `reward` column contains non-numeric values.

## Internal Methods

### _last_row_index

**Signature**

- `_last_row_index() -> int`

**Purpose**

Return index of the last logged row.

**Arguments**

- None.

**Returns**

- `idx`: `int`, scalar.

**Preconditions**

- The log contains at least one row.

**Postconditions**

- No mutation.

**Errors**

- Raises `RuntimeError` if the log is empty.

!!! tip "Internal helper"
    This is an internal helper method and may change without notice; prefer the public update APIs.

## Data & State

- Attributes (public):
  - `game_layout`: `GameLayout`, scalar.
  - `log`: `pd.DataFrame`, rows `n_games`, columns `len(game_layout.log_columns)`.
    - Contains one row per game.

- Side effects:
  - `update(...)` appends a row to `log`.
  - `update_*` methods mutate the last row in `log`.

- Thread-safety:
  - Not thread-safe (mutates a shared DataFrame).

## Planned (design-spec)

- Log schema evolution:
  - The design document defines `log_columns` in `GameLayout`. Future extensions may add additional columns (e.g. extra
    metadata or diagnostic fields). This class should remain backward compatible by not assuming column presence beyond
    `game_layout.log_columns`.

## Deviations

- Column initialisation:
  - The DataFrame is created with columns `game_layout.log_columns`, but `update(...)` writes a fixed row dictionary that
    includes columns (e.g. `logprob_comm`, `game_id`, `prev_measurements`) that must also be present in `log_columns` to
    avoid implicit column creation or assignment errors.
  - The design intent is that `GameLayout.log_columns` includes these names; if it does not, this becomes a runtime
    mismatch.

- Stored shapes:
  - The class stores `field`, `gun`, and `comm` as entire arrays per row. This matches the current design, but implies
    memory growth proportional to `n_games * (n2 + n2 + m)` in array elements (plus DataFrame overhead).

## Notes for Contributors

- Keep updates safe and deterministic:
  - `update(...)` uses `self.log.loc[len(self.log)] = row` to avoid deprecated append patterns.
- If you change the log schema:
  - Update `GameLayout.log_columns` defaults and ensure `TournamentLog.update(...)` initialises all new columns.
- If performance becomes a concern:
  - Consider preallocating lists and converting to a DataFrame at the end, but document any API changes carefully.
- Do not add game execution logic here; this class must remain a passive accumulator.

## Related

- See also: `Tournament` (orchestration), `Game` (single-round), `GameLayout` (defines `log_columns`). 
## Changelog

- 2026-01-07 — Author: Rob Hendriks — Initial specification page for `TournamentLog`.
