# GameLayout

> **Role**: Immutable configuration container for a QSeaBattle game and its surrounding tournament settings.

**Location**: `Q_Sea_Battle.game_layout.GameLayout`

## Constructor

| Parameter | Type | Description |
|---|---|---|
| field_size | int | Board dimension `field_size = n` for an $n \times n$ field. Must satisfy `field_size > 0` and `$n^2$` is a power of two. Default: `4`. |
| comms_size | int | Communication vector length `m`. Must satisfy `comms_size > 0` and `m | n2` where `n2 = field_size**2`. Default: `1`. |
| enemy_probability | float | Probability that a cell in the field equals `1`. Must be in `[0.0, 1.0]`. Default: `0.5`. |
| channel_noise | float | Per-bit flip probability applied to the communication vector by the channel. Must be in `[0.0, 1.0]`. Default: `0.0`. |
| number_of_games_in_tournament | int | Number of games played per tournament run. Must satisfy `number_of_games_in_tournament > 0`. Default: `100`. |
| log_columns | list[str] | Tournament-log schema: list of column names (strings). Default: see “Data & State”. |

**Preconditions**

- `field_size` is an `int` and `field_size > 0`.
- Let `n2 = field_size**2`. Then `n2` is a power of two.
- `comms_size` is an `int`, `comms_size > 0`, and `n2 % comms_size == 0` (i.e. `m | n2`).
- `enemy_probability` and `channel_noise` are `float` and lie in `[0.0, 1.0]`.
- `number_of_games_in_tournament` is an `int` and `number_of_games_in_tournament > 0`.
- `log_columns` is a `list[str]`.

**Postconditions**

- The instance is immutable (`@dataclass(frozen=True)`); attributes cannot be modified after construction.
- All constraints above are validated at construction time (via `__post_init__`).

**Errors**

- Raises `TypeError` if any of `field_size`, `comms_size`, or `number_of_games_in_tournament` is not an `int`, or if `log_columns` is not a `list[str]`.
- Raises `ValueError` if any value violates the constraints above.

!!! note "Derived constraints"
    - `n2 = field_size**2`.
    - `m = comms_size`.
    - The class enforces `m | n2` (segmentations and reshapes that rely on equal partitions may assume this).

## Public Methods

### `from_dict(parameters)`
**Purpose**: Construct a validated `GameLayout` from a parameter dictionary, ignoring unknown keys.  
**Args**:  
- `parameters` — `dict[str, Any]`, shape n/a, parameter overrides; unknown keys are ignored.  
**Returns**: `GameLayout`, shape n/a.  
**Raises**: `TypeError` or `ValueError` if the resulting instance violates constructor constraints.

!!! example "Example"
    ```python
    from Q_Sea_Battle.game_layout import GameLayout

    layout = GameLayout.from_dict({"field_size": 8, "comms_size": 4})
    ```

### `to_dict()`
**Purpose**: Return all layout fields as a dictionary.  
**Args**: None.  
**Returns**: `dict[str, Any]`, shape n/a, mapping field-name to current value.  
**Raises**: No exceptions are raised by this method (it is a pure projection).

!!! example "Example"
    ```python
    d = layout.to_dict()
    ```

## Data & State

- Attributes (public):
  - `field_size` — `int`, scalar, `field_size > 0`, `n2 = field_size**2` is a power of two.
  - `comms_size` — `int`, scalar, `comms_size > 0`, `comms_size | n2`.
  - `enemy_probability` — `float`, scalar, in `[0.0, 1.0]`.
  - `channel_noise` — `float`, scalar, in `[0.0, 1.0]`.
  - `number_of_games_in_tournament` — `int`, scalar, `> 0`.
  - `log_columns` — `list[str]`, schema for tournament logging.

- Default `log_columns` contents:
  - `field`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)`.
  - `gun`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)` (one-hot).
  - `comm`: `np.ndarray`, dtype `int` {0,1}, shape `(m,)`.
  - `shoot`: `int` {0,1}, scalar.
  - `cell_value`: `int` {0,1}, scalar.
  - `reward`: `float` {0.0, 1.0}, scalar.
  - `sample_weight`: `float`, scalar, `>= 0.0`.
  - `logprob_comm`: `float`, scalar.
  - `logprob_shoot`: `float`, scalar.
  - `game_id`: `int`, scalar.
  - `tournament_id`: `int`, scalar.
  - `meta_id`: `int`, scalar.
  - `game_uid`: `str`, scalar.
  - `prev_measurements`: `list` or array-like, shape depends on player architecture (typically per shared layer).
  - `prev_outcomes`: `list` or array-like, shape depends on player architecture (typically per shared layer).

- Side effects: None (pure configuration).
- Thread-safety: Safe to share between threads/processes (immutable).

!!! tip "Typical shape usage"
    Many APIs derive shapes from `n2` and `m`. Examples:
    - Field vectors: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)`.
    - Communication vectors: `np.ndarray`, dtype `int` {0,1}, shape `(m,)`.

## Planned (design-spec)

- None. The implemented class matches the design specification for `GameLayout`.

## Deviations

- None. The implementation matches the design document for:
  - the validated constraints (`n2` is power of two, `m | n2`, and probability ranges), and
  - `log_columns` defaults.

## Notes for Contributors

- Keep `GameLayout` immutable. If you need “derived” configuration (e.g., cached `n2`), compute it locally in callers instead of adding mutable fields.
- Maintain validation in `__post_init__` and keep error messages explicit (include offending values and derived `n2` where relevant).
- If future architectures introduce additional log columns, append new names to `log_columns` in a backward-compatible way (avoid renaming).
- Avoid adding non-deterministic behavior here; randomness belongs in environment/player layers, not in configuration.

## Related

- See also: `GameEnv` (environment), `TournamentLog` (logging schema consumer).

## Changelog

- 2026-01-07 — Author: Rob Hendriks — Initial specification page for `GameLayout`.
