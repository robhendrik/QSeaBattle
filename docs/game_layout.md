# GameLayout

> Role: Provide an immutable, validated configuration for a QSeaBattle game and its tournament logging schema.
Location: `Q_Sea_Battle.game_layout.GameLayout`

## Derived constraints

Definitions used throughout this specification: $n = \texttt{field_size}$, $n2 = n^2$, $m = \texttt{comms_size}$.  
Constraints enforced at initialization: $n > 0$, $n2$ is a power of two, $m > 0$, $m$ divides $n2$, $\texttt{enemy_probability} \in [0.0, 1.0]$, $\texttt{channel_noise} \in [0.0, 1.0]$, $\texttt{number_of_games_in_tournament} > 0$, and $\texttt{log_columns}$ is a list of strings.

## Constructor

Parameter | Type | Description
--- | --- | ---
field_size | int, constraints: $> 0$ and $n2=\texttt{field_size}^2$ is a power of two, shape: scalar | Size $n$ of one dimension of the square field; flattened field length is $n2$.
comms_size | int, constraints: $> 0$ and divides $n2$, shape: scalar | Communication vector length $m$; must satisfy $n2 \bmod m = 0$.
enemy_probability | float, constraints: $0.0 \le p \le 1.0$, shape: scalar | Probability that a cell in the field equals 1.
channel_noise | float, constraints: $0.0 \le p \le 1.0$, shape: scalar | Probability that a bit is flipped in the channel.
number_of_games_in_tournament | int, constraints: $> 0$, shape: scalar | Number of games per tournament.
log_columns | List[str], constraints: all elements are str, shape: (k,) | Column names for the tournament log.

!!! note "Preconditions"
    The instance is validated in `__post_init__`; callers must provide values meeting the constraints above or expect an exception.

!!! note "Postconditions"
    The created instance is immutable (`@dataclass(frozen=True)`) and has validated field values.

!!! warning "Errors"
    - Raises `TypeError` if `field_size`, `comms_size`, or `number_of_games_in_tournament` is not an `int`.
    - Raises `TypeError` if `log_columns` is not a `list` of `str`.
    - Raises `ValueError` if `field_size <= 0`.
    - Raises `ValueError` if $n2$ is not a power of two.
    - Raises `ValueError` if `comms_size <= 0` or if `comms_size` does not divide $n2$.
    - Raises `ValueError` if `enemy_probability` is not in $[0.0, 1.0]$.
    - Raises `ValueError` if `channel_noise` is not in $[0.0, 1.0]$.
    - Raises `ValueError` if `number_of_games_in_tournament <= 0`.

!!! example "Example"
    ```python
    from Q_Sea_Battle.game_layout import GameLayout

    layout = GameLayout(field_size=4, comms_size=2, enemy_probability=0.25, channel_noise=0.0, number_of_games_in_tournament=100)
    ```

## Public Methods

### from_dict

Create a `GameLayout` instance from a dictionary of parameters, ignoring unknown keys and using dataclass defaults for missing keys.

Parameters | Type | Description
--- | --- | ---
parameters | Dict, constraints: keys may include any dataclass field names; unknown keys are ignored, shape: mapping | Dictionary containing parameter overrides.

Returns | Type | Description
--- | --- | ---
layout | GameLayout, constraints: validated instance, shape: scalar | A new validated `GameLayout` instance.

!!! warning "Errors"
    May raise `TypeError` or `ValueError` as described for the constructor because the created instance is validated via `__post_init__`.

!!! example "Example"
    ```python
    from Q_Sea_Battle.game_layout import GameLayout

    layout = GameLayout.from_dict({"field_size": 4, "comms_size": 1, "unknown": 123})
    ```

### to_dict

Return a dictionary representation containing all dataclass fields and their current values.

Parameters | Type | Description
--- | --- | ---
(no parameters) | None, constraints: not applicable, shape: not applicable | Not specified.

Returns | Type | Description
--- | --- | ---
layout_dict | Dict, constraints: keys are the dataclass field names, shape: mapping | Dictionary with all layout parameters and their current values.

!!! example "Example"
    ```python
    from Q_Sea_Battle.game_layout import GameLayout

    d = GameLayout().to_dict()
    ```

## Data & State

- `field_size`: int, constraints: $> 0$ and $n2=\texttt{field_size}^2$ is a power of two, shape: scalar.
- `comms_size`: int, constraints: $> 0$ and divides $n2$, shape: scalar.
- `enemy_probability`: float, constraints: $0.0 \le p \le 1.0$, shape: scalar.
- `channel_noise`: float, constraints: $0.0 \le p \le 1.0$, shape: scalar.
- `number_of_games_in_tournament`: int, constraints: $> 0$, shape: scalar.
- `log_columns`: List[str], constraints: all elements are str, shape: (k,).
- Immutability: instances are frozen; direct mutation of fields is not allowed by the dataclass.

## Planned (design-spec)

Not specified.

## Deviations

Not specified.

## Notes for Contributors

- Validation is implemented in `__post_init__`; changes to constraints should be reflected there to preserve the immutability + validate-on-creation contract.
- `_is_power_of_two` uses a bitwise test and assumes an `int` input; keep it consistent with the existing validation flow.

## Related

- `dataclasses.dataclass` (frozen dataclass behavior)
- `GameLayout._is_power_of_two` (internal validation helper)

## Changelog

- 0.1: Initial version (module docstring version).