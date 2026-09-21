# TournamentLog

> Role: Structured log for storing QSeaBattle tournament results as a Pandas `DataFrame` with one row per game, supporting late-bound per-row updates.

Location: `Q_Sea_Battle.tournament_log.TournamentLog`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | `GameLayout`, constraints: must provide attribute `log_columns`; shape: N/A | Layout providing the log column names used to initialize the underlying `pd.DataFrame` columns. |

Preconditions

- `game_layout.log_columns` exists and is accepted by `pd.DataFrame(columns=...)`.

Postconditions

- `self.game_layout` is set to the provided `game_layout`.
- `self.log` is an empty `pd.DataFrame` with columns equal to `game_layout.log_columns`.

Errors

- Not specified.

!!! example "Example"
    ```python
    from Q_Sea_Battle.tournament_log import TournamentLog
    from Q_Sea_Battle.game_layout import GameLayout

    layout = GameLayout(...)
    tlog = TournamentLog(game_layout=layout)
    ```

## Public Methods

### update

Append a new game result row to the log and initialize late-bound fields to `None`.

| Parameter | Type | Description |
| --- | --- | --- |
| field | `np.ndarray`, constraints: Not specified; shape: Not specified | Game field state for the game. |
| gun | `np.ndarray`, constraints: Not specified; shape: Not specified | Gun state/action representation for the game. |
| comm | `np.ndarray`, constraints: Not specified; shape: Not specified | Communication representation for the game. |
| shoot | `int`, constraints: convertible via `int(shoot)`; shape: scalar | Shot/cell index selected for the game. |
| cell_value | `int`, constraints: convertible via `int(cell_value)`; shape: scalar | Observed value at the shot cell. |
| reward | `float`, constraints: convertible via `float(reward)`; shape: scalar | Scalar reward for the game. |

Returns

- `None`, shape: N/A.

Preconditions

- `self.log` is a `pd.DataFrame`.
- Column names referenced by this method are present or are accepted by Pandas row assignment: `field`, `gun`, `comm`, `shoot`, `cell_value`, `reward`, `logprob_comm`, `logprob_shoot`, `game_id`, `tournament_id`, `meta_id`, `game_uid`, `prev_measurements`, `prev_outcomes`.

Postconditions

- A new row is added at index `len(self.log) - 1` containing provided values (with `shoot`, `cell_value`, `reward` coerced to `int`, `int`, `float` respectively).
- The following fields for the new row are set to `None`: `logprob_comm`, `logprob_shoot`, `game_id`, `tournament_id`, `meta_id`, `game_uid`, `prev_measurements`, `prev_outcomes`.

Errors

- Not specified.

!!! example "Example"
    ```python
    import numpy as np
    from Q_Sea_Battle.tournament_log import TournamentLog
    from Q_Sea_Battle.game_layout import GameLayout

    layout = GameLayout(...)
    tlog = TournamentLog(layout)

    field = np.zeros((4, 4), dtype=int)
    gun = np.array([1, 0, 0], dtype=int)
    comm = np.array([0.1, 0.9], dtype=float)

    tlog.update(field=field, gun=gun, comm=comm, shoot=3, cell_value=1, reward=0.5)
    ```

### update_log_probs

Update log-probabilities for the last logged game.

| Parameter | Type | Description |
| --- | --- | --- |
| logprob_comm | `float`, constraints: convertible via `float(logprob_comm)`; shape: scalar | Log-probability associated with the communication decision. |
| logprob_shoot | `float`, constraints: convertible via `float(logprob_shoot)`; shape: scalar | Log-probability associated with the shooting decision. |

Returns

- `None`, shape: N/A.

Preconditions

- The log is non-empty.

Postconditions

- For the last row, `logprob_comm` and `logprob_shoot` are set to the provided values coerced to `float`.

Errors

- `RuntimeError`: If no rows have been logged yet (raised by `_last_row_index`).

!!! example "Example"
    ```python
    tlog.update_log_probs(logprob_comm=-0.12, logprob_shoot=-1.83)
    ```

### update_log_prev

Update previous measurements/outcomes for the last logged game.

| Parameter | Type | Description |
| --- | --- | --- |
| prev_meas | `Any`, constraints: Not specified; shape: Not applicable | Previous measurements per shared layer; stored as an opaque object. |
| prev_out | `Any`, constraints: Not specified; shape: Not applicable | Previous outcomes per shared layer; stored as an opaque object. |

Returns

- `None`, shape: N/A.

Preconditions

- The log is non-empty.

Postconditions

- For the last row, `prev_measurements` is set to `prev_meas` and `prev_outcomes` is set to `prev_out`.

Errors

- `RuntimeError`: If no rows have been logged yet (raised by `_last_row_index`).

!!! example "Example"
    ```python
    tlog.update_log_prev(prev_meas={"layer0": [1, 2]}, prev_out={"layer0": [0, 1]})
    ```

### update_indicators

Update identifier fields for the last logged game and generate a unique `game_uid`.

| Parameter | Type | Description |
| --- | --- | --- |
| game_id | `int`, constraints: convertible via `int(game_id)`; shape: scalar | Identifier of the game within a tournament. |
| tournament_id | `int`, constraints: convertible via `int(tournament_id)`; shape: scalar | Identifier of the tournament. |
| meta_id | `int`, constraints: convertible via `int(meta_id)`; shape: scalar | Identifier for experimental metadata. |

Returns

- `None`, shape: N/A.

Preconditions

- The log is non-empty.

Postconditions

- For the last row: `game_id`, `tournament_id`, and `meta_id` are set to the provided values coerced to `int`.
- For the last row: `game_uid` is set to a UUID4 hex string (`uuid.uuid4().hex`).

Errors

- `RuntimeError`: If no rows have been logged yet (raised by `_last_row_index`).

!!! example "Example"
    ```python
    tlog.update_indicators(game_id=7, tournament_id=2, meta_id=42)
    ```

### outcome

Compute aggregate reward statistics over the logged games.

| Parameter | Type | Description |
| --- | --- | --- |
| (none) | (none) | (none) |

Returns

- `Tuple[float, float]`, constraints: `(0.0, 0.0)` if the log is empty; shape: `(2,)` as a 2-tuple: `(mean_reward, std_error)`.

Preconditions

- `self.log` has a `reward` column containing values convertible to `float`.

Postconditions

- No mutation of `self.log` is performed.

Errors

- Not specified.

!!! note "Computation details"
    If the log is non-empty, rewards are converted via `self.log["reward"].astype(float).to_numpy()`, `mean_reward` is the arithmetic mean, and `std_error` is $0.0$ for $n \le 1$ else $s / \sqrt{n}$ where $s$ is the sample standard deviation computed with `ddof=1`.

!!! example "Example"
    ```python
    mean_reward, std_error = tlog.outcome()
    ```

## Data & State

- `game_layout`: `GameLayout`, constraints: must provide `log_columns`; shape: N/A; set in `__init__`.
- `log`: `pd.DataFrame`, constraints: columns initialized from `game_layout.log_columns`; shape: `(n_rows, n_cols)` where `n_rows` is the number of logged games; rows contain at least the keys written by `update` and may include additional columns present in `game_layout.log_columns`.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- `_last_row_index` is a private helper that raises `RuntimeError` when the log is empty; public `update_*` methods rely on this behavior.
- `update` assigns the new row via `self.log.loc[len(self.log)] = row`, avoiding deprecated/inefficient `DataFrame.append`.

## Related

- `Q_Sea_Battle.game_layout.GameLayout` (provides `log_columns` used to define the log schema).
- Pandas `DataFrame` (storage backend).

## Changelog

- Not specified.