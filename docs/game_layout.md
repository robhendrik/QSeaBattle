# GameLayout

> Role: Immutable configuration container for a QSeaBattle game and its tournament/logging schema.
Location: `Q_Sea_Battle.game_layout.GameLayout`

## Derived constraints

- Define $n=\texttt{field_size}$ (int, $n>0$), $n2=n^2$ (int), and $m=\texttt{comms_size}$ (int).  
- Constraint: $n2$ must be a power of two.  
- Constraint: $m>0$ and $n2 \bmod m = 0$.  
- Constraint: $\texttt{enemy_probability} \in [0.0, 1.0]$ and $\texttt{channel_noise} \in [0.0, 1.0]$.  
- Constraint: $\texttt{number_of_games_in_tournament} > 0$.  
- Constraint: $\texttt{log_columns}$ is `list[str]` (minimal validation: list and all elements are `str`).  

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| field_size | int, constraints: $>0$ and $n2=\texttt{field_size}^2$ is a power of two, shape: scalar | Side length $n$ of the square field; flattened length is $n2$. Default: `4`. |
| comms_size | int, constraints: $>0$ and divides $n2$, shape: scalar | Communication vector length $m$. Default: `1`. |
| enemy_probability | float, constraints: $0.0 \le p \le 1.0$, shape: scalar | Probability that a generated field cell equals `1`. Default: `0.5`. |
| channel_noise | float, constraints: $0.0 \le p \le 1.0$, shape: scalar | Bit-flip probability for the channel. Default: `0.0`. |
| number_of_games_in_tournament | int, constraints: $>0$, shape: scalar | Number of games played per tournament. Default: `100`. |
| log_columns | list[str], constraints: list and all elements `str`, shape: (k,) | Column names used when logging tournament/game events. Default: a predefined list of strings (see source). |

### Preconditions

- Inputs must satisfy the constraints described in the table and in `__post_init__`.

### Postconditions

- The instance is created and validated; the dataclass is frozen (immutable) after creation.

### Errors

- `TypeError`: If `field_size`, `comms_size`, or `number_of_games_in_tournament` is not an `int`.
- `ValueError`: If `field_size <= 0`.
- `ValueError`: If $n2=\texttt{field_size}^2$ is not a power of two.
- `ValueError`: If `comms_size <= 0` or if `comms_size` does not divide $n2$.
- `ValueError`: If `enemy_probability` or `channel_noise` is outside `[0.0, 1.0]`.
- `ValueError`: If `number_of_games_in_tournament <= 0`.
- `TypeError`: If `log_columns` is not a `list` of `str`.

### Example

!!! example "Constructing a layout"
    ```python
    from Q_Sea_Battle.game_layout import GameLayout

    layout = GameLayout(field_size=4, comms_size=4, enemy_probability=0.3, channel_noise=0.1)
    ```

## Public Methods

### from_dict

- Signature: `@classmethod def from_dict(cls, parameters: Dict) -> "GameLayout"`

Creates a validated `GameLayout` from a mapping; unknown keys are ignored and missing keys use dataclass defaults.

**Parameters**

- `parameters`: Dict, constraints: mapping of field names to override values; unknown keys ignored, shape: not applicable.

**Returns**

- `GameLayout`, constraints: validated instance (validation performed by `__post_init__`), shape: scalar object.

**Errors**

- `TypeError` / `ValueError`: Propagated from `__post_init__` if provided or defaulted values violate constraints.

**Example**

!!! example "Creating from a dict"
    ```python
    from Q_Sea_Battle.game_layout import GameLayout

    layout = GameLayout.from_dict({"field_size": 4, "comms_size": 2, "unknown_key": 123})
    ```

### to_dict

- Signature: `def to_dict(self) -> Dict`

Converts this layout to a dictionary containing all dataclass fields.

**Parameters**

- None.

**Returns**

- `Dict`, constraints: keys are dataclass field names; values are the corresponding field values, shape: not applicable.

**Errors**

- Not specified.

**Example**

!!! example "Converting to dict"
    ```python
    from Q_Sea_Battle.game_layout import GameLayout

    layout = GameLayout()
    d = layout.to_dict()
    ```

## Data & State

- Mutability: Immutable (`@dataclass(frozen=True)`); state cannot be modified after construction.
- Fields:
  - `field_size`: int, constraints: $>0$ and $n2=\texttt{field_size}^2$ is a power of two, shape: scalar.
  - `comms_size`: int, constraints: $>0$ and divides $n2$, shape: scalar.
  - `enemy_probability`: float, constraints: $0.0 \le p \le 1.0$, shape: scalar.
  - `channel_noise`: float, constraints: $0.0 \le p \le 1.0$, shape: scalar.
  - `number_of_games_in_tournament`: int, constraints: $>0$, shape: scalar.
  - `log_columns`: list[str], constraints: list and all elements `str`, shape: (k,).

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- Validation is implemented in `__post_init__`; changes to constraints should be reflected there to preserve the guarantee that all created instances are valid.
- `from_dict` intentionally ignores unknown keys; if stricter behavior is desired, it must be implemented explicitly.

## Related

- `dataclasses.dataclass` (used with `frozen=True` to enforce immutability).
- Internal helper: `GameLayout._is_power_of_two(value: int) -> bool` (static method; not documented here as a public method).

## Changelog

- Not specified.