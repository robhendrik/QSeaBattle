# Class GameLayout

## Purpose
Define the immutable parameters that specify a QSeaBattle game configuration, ensuring consistency across environment, players, tournaments, and logs.

## Location
- **Module:** `src/Q_Sea_Battle/game_layout.py`
- **Class:** `GameLayout`

## Public attributes
| Name | Type | Constraints | Description |
|----|----|----|----|
| `field_size` | `int` | power of 2 | Board dimension n; total cells n^2 |
| `comms_size` | `int` | divides n^2 | Number of communication bits |
| `enemy_probability` | `float` | [0,1] | Probability a field cell equals 1 |
| `channel_noise` | `float` | [0,1] | Bit-flip probability on comm channel |
| `number_of_games_in_tournament` | `int` | >0 | Games per tournament |
| `log_columns` | `list[str]` | — | Columns stored in TournamentLog |

## Public methods
### `from_dict(parameters: dict) -> GameLayout`
Create a validated GameLayout from a dictionary.

### `to_dict() -> dict`
Return all layout parameters as a dictionary.

## Invariants
- `field_size` MUST be a power of two.
- `comms_size` MUST divide `field_size^2`.
- All components MUST share the same GameLayout instance.

## Failure modes
- `ValueError` if constraints are violated.