# TournamentLog

> Role: Structured log for storing QSeaBattle tournament results.
Location: `Q_Sea_Battle.tournament_log.TournamentLog`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | GameLayout, not specified, shape N/A | Layout providing the log column names; used to set `self.game_layout` and to initialize an empty `pd.DataFrame` with `columns=game_layout.log_columns`. |

Preconditions

- `game_layout.log_columns` exists and is compatible with `pd.DataFrame(columns=...)`.

Postconditions

- `self.game_layout` is set to the provided `game_layout`.
- `self.log` is a `pd.DataFrame` with columns `game_layout.log_columns` and zero rows.

Errors

- Not specified.

Example

```python
from Q_Sea_Battle.tournament_log import TournamentLog
from Q_Sea_Battle.game_layout import GameLayout

game_layout = GameLayout(...)  # Not specified
tlog = TournamentLog(game_layout=game_layout)
```

## Public Methods

### update

Append a new game result to the log.

Signature: `update(field, gun, comm, shoot, cell_value, reward) -> None`

Parameters

- `field`: np.ndarray, dtype not specified, shape not specified; stored under column `"field"`.
- `gun`: np.ndarray, dtype not specified, shape not specified; stored under column `"gun"`.
- `comm`: np.ndarray, dtype not specified, shape not specified; stored under column `"comm"`.
- `shoot`: int, constraints not specified, shape N/A; converted via `int(shoot)` and stored under column `"shoot"`.
- `cell_value`: int, constraints not specified, shape N/A; converted via `int(cell_value)` and stored under column `"cell_value"`.
- `reward`: float, constraints not specified, shape N/A; converted via `float(reward)` and stored under column `"reward"`.

Returns

- `None`: NoneType, shape N/A.

Preconditions

- `self.log` is a `pd.DataFrame` that can accept assignment at index `len(self.log)` with the constructed `row` mapping.
- `self.log` has (or can accept) the columns: `"field"`, `"gun"`, `"comm"`, `"shoot"`, `"cell_value"`, `"reward"`, `"logprob_comm"`, `"logprob_shoot"`, `"game_id"`, `"tournament_id"`, `"meta_id"`, `"game_uid"`, `"prev_measurements"`, `"prev_outcomes"`.

Postconditions

- Appends exactly one new row to `self.log` with the provided values and with the following columns set to `None`: `"logprob_comm"`, `"logprob_shoot"`, `"game_id"`, `"tournament_id"`, `"meta_id"`, `"game_uid"`, `"prev_measurements"`, `"prev_outcomes"`.

Errors

- Not specified.

Example

```python
import numpy as np
from Q_Sea_Battle.tournament_log import TournamentLog

tlog = TournamentLog(game_layout=game_layout)  # game_layout not specified
field = np.zeros((5, 5), dtype=int)
gun = np.array([0, 1], dtype=int)
comm = np.array([1, 0], dtype=int)

tlog.update(field=field, gun=gun, comm=comm, shoot=3, cell_value=0, reward=1.0)
```

### update_log_probs

Update log-probabilities for the last logged game.

Signature: `update_log_probs(logprob_comm, logprob_shoot) -> None`

Parameters

- `logprob_comm`: float, constraints not specified, shape N/A; converted via `float(logprob_comm)` and stored in column `"logprob_comm"` for the last row.
- `logprob_shoot`: float, constraints not specified, shape N/A; converted via `float(logprob_shoot)` and stored in column `"logprob_shoot"` for the last row.

Returns

- `None`: NoneType, shape N/A.

Preconditions

- `self.log` is not empty.

Postconditions

- The last row in `self.log` has updated values in `"logprob_comm"` and `"logprob_shoot"`.

Errors

- `RuntimeError`: raised if no rows have been logged yet (propagated from `_last_row_index`).

Example

```python
tlog.update_log_probs(logprob_comm=-0.12, logprob_shoot=-1.34)
```

### update_log_prev

Update previous measurements/outcomes for the last game.

Signature: `update_log_prev(prev_meas, prev_out) -> None`

Parameters

- `prev_meas`: Any, constraints not specified, shape not specified; stored in column `"prev_measurements"` for the last row.
- `prev_out`: Any, constraints not specified, shape not specified; stored in column `"prev_outcomes"` for the last row.

Returns

- `None`: NoneType, shape N/A.

Preconditions

- `self.log` is not empty.

Postconditions

- The last row in `self.log` has updated values in `"prev_measurements"` and `"prev_outcomes"`.

Errors

- `RuntimeError`: raised if no rows have been logged yet (propagated from `_last_row_index`).

Example

```python
tlog.update_log_prev(prev_meas={"layer0": [1, 2]}, prev_out={"layer0": [0]})
```

### update_indicators

Update identifier fields for the last logged game and generate a unique `game_uid`.

Signature: `update_indicators(game_id, tournament_id, meta_id) -> None`

Parameters

- `game_id`: int, constraints not specified, shape N/A; converted via `int(game_id)` and stored in column `"game_id"` for the last row.
- `tournament_id`: int, constraints not specified, shape N/A; converted via `int(tournament_id)` and stored in column `"tournament_id"` for the last row.
- `meta_id`: int, constraints not specified, shape N/A; converted via `int(meta_id)` and stored in column `"meta_id"` for the last row.

Returns

- `None`: NoneType, shape N/A.

Preconditions

- `self.log` is not empty.

Postconditions

- The last row in `self.log` has updated values in `"game_id"`, `"tournament_id"`, and `"meta_id"`.
- The last row in `self.log` has `"game_uid"` set to `uuid.uuid4().hex` (str, hex characters, length not specified by the code).

Errors

- `RuntimeError`: raised if no rows have been logged yet (propagated from `_last_row_index`).

Example

```python
tlog.update_indicators(game_id=7, tournament_id=2, meta_id=42)
```

### outcome

Compute aggregate statistics over the tournament.

Signature: `outcome() -> Tuple[float, float]`

Parameters

- None.

Returns

- `Tuple[float, float]`: tuple, shape (2,); `(mean_reward, std_error)` where `mean_reward` is the mean of the `"reward"` column and `std_error` is $s / \sqrt{n}$ with sample standard deviation `s` computed using `ddof=1` when `n > 1`, otherwise `0.0`.

Preconditions

- If `self.log` is non-empty, column `"reward"` exists and values are convertible to `float` via `astype(float)`.

Postconditions

- Does not mutate `self.log`.

Errors

- Not specified.

Example

```python
mean_reward, std_error = tlog.outcome()
```

## Data & State

- `game_layout`: GameLayout, not specified, shape N/A; layout instance provided at construction time.
- `log`: pd.DataFrame, constraints not specified, shape (n_games, n_cols); DataFrame containing one row per game, initially empty with `columns=game_layout.log_columns`, and later appended/updated by methods in this class.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- `_last_row_index()` is a private helper that raises `RuntimeError` when `self.log` is empty; public mutators that update the "last row" rely on this behavior.
- `update()` sets several fields to `None` regardless of `game_layout.log_columns`; ensure `game_layout.log_columns` is compatible with the columns written by this class to avoid assignment/column-mismatch issues.

## Related

- `Q_Sea_Battle.game_layout.GameLayout` (provides `log_columns` used to initialize `self.log`).

## Changelog

- 0.1: Initial implementation with row append, last-row updates (log probs, previous values, identifiers), and aggregate outcome statistics.