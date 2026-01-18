# TrainableAssistedPlayers

> Role: Players-style wrapper that wires a trainable sender (A) and receiver (B) assisted player pair and owns their underlying models and shared state.

Location: `Q_Sea_Battle.trainable_assisted_players.TrainableAssistedPlayers`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | Any, not specified, must provide attributes `field_size` and `comms_size` convertible to `int` | Game-layout-like object used to parameterize default models and passed into player wrappers. |
| p_high | float, not specified | Forward-compatibility parameter; currently unused by this class when building default linear models. |
| num_iterations | Optional[int], not specified | Forward-compatibility parameter; currently unused by this class when building default linear models. |
| hidden_dim | int, not specified | Forward-compatibility parameter; currently unused by this class when building default linear models. |
| L_meas | Optional[int], not specified | Forward-compatibility parameter; currently unused by this class when building default linear models. |
| model_a | Optional[LinTrainableAssistedModelA], default `None` | If provided, used as the A-side model; otherwise a default `LinTrainableAssistedModelA` is constructed from `game_layout.field_size` and `game_layout.comms_size`. |
| model_b | Optional[LinTrainableAssistedModelB], default `None` | If provided, used as the B-side model; otherwise a default `LinTrainableAssistedModelB` is constructed from `game_layout.field_size` and `game_layout.comms_size`. |

Preconditions

- `game_layout` has attributes `field_size` and `comms_size` such that `int(getattr(game_layout, "field_size"))` and `int(getattr(game_layout, "comms_size"))` succeed when default models are constructed.

Postconditions

- `self.game_layout` is set to `game_layout` (type: Any, constraints: not specified).
- `self.explore` is initialized to `False` (type: bool, constraints: {True, False}).
- `self.model_a` is set (type: LinTrainableAssistedModelA, constraints: not specified).
- `self.model_b` is set (type: LinTrainableAssistedModelB, constraints: not specified).
- `self.previous` is initialized to `None` (type: Any | None, constraints: typically either `None` or a tuple `(measurements_per_layer, outcomes_per_layer)` where both are Python lists of tensors each shaped `(B, n2)`).
- Lazy player wrappers `self._playerA` and `self._playerB` are initialized to `None` (type: Optional[TrainableAssistedPlayerA] and Optional[TrainableAssistedPlayerB], constraints: not specified).
- `self.has_prev` is initialized to `True` (type: bool, constraints: {True, False}).

Errors

- Any exception raised by `int(getattr(game_layout, "field_size"))` or `int(getattr(game_layout, "comms_size"))` when constructing default models.
- Any exception raised by `LinTrainableAssistedModelA(...)` or `LinTrainableAssistedModelB(...)` constructors.

Example

```python
from Q_Sea_Battle.trainable_assisted_players import TrainableAssistedPlayers

class Layout:
    field_size = 10
    comms_size = 2

tap = TrainableAssistedPlayers(Layout())
player_a, player_b = tap.players()
tap.set_explore(True)
tap.reset()
```

## Public Methods

### check_model_correspondence

Check that model A and B are compatible by comparing exposed `field_size` and `comms_size` attributes if present.

Arguments

- None

Returns

- bool, constraints: {True, False}, shape: scalar.

Errors

- Not specified; internal exceptions while accessing model attributes are caught and result in `True`.

### players

Return the (PlayerA, PlayerB) wrappers, constructing them lazily and persisting wrapper state until `reset()`.

Arguments

- None

Returns

- Tuple[TrainableAssistedPlayerA, TrainableAssistedPlayerB], constraints: 2-tuple `(player_a, player_b)`, shape: length 2.

Errors

- Any exception raised by `TrainableAssistedPlayerA(self.game_layout, model_a=self.model_a)` or `TrainableAssistedPlayerB(self.game_layout, model_b=self.model_b)` during lazy construction.

### reset

Reset internal state between games by resetting both wrappers (if they exist) and clearing `previous`.

Arguments

- None

Returns

- NoneType, constraints: always `None`, shape: scalar.

Errors

- Any exception raised by `self._playerA.reset()` or `self._playerB.reset()` if those wrappers exist.

### set_explore

Set the exploration flag for both players and the wrapper itself.

Arguments

- flag: bool, constraints: {True, False}, shape: scalar.

Returns

- NoneType, constraints: always `None`, shape: scalar.

Errors

- Not specified.

## Data & State

- has_log_probs: bool, constraints: {True, False}, shape: scalar; class attribute set to `True`.
- game_layout: Any, constraints: not specified; expected to provide `field_size` and `comms_size` attributes.
- model_a: LinTrainableAssistedModelA, constraints: not specified.
- model_b: LinTrainableAssistedModelB, constraints: not specified.
- explore: bool, constraints: {True, False}, shape: scalar; shared exploration flag propagated to wrappers when they exist.
- previous: Any | None, constraints: typically `(measurements_per_layer, outcomes_per_layer)` or `None`; when present, expected contract is `(meas_list, out_list)` where both are Python lists of tensors each shaped `(B, n2)`.
- has_prev: bool, constraints: {True, False}, shape: scalar; initialized to `True`.
- _playerA: Optional[TrainableAssistedPlayerA], constraints: either `None` or an instantiated wrapper; created lazily by `players()`.
- _playerB: Optional[TrainableAssistedPlayerB], constraints: either `None` or an instantiated wrapper; created lazily by `players()`.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- `p_high`, `num_iterations`, `hidden_dim`, and `L_meas` are accepted by the constructor for forward compatibility but are currently unused by this class when creating default linear models.
- The `previous` state is intended to be produced by Player A and consumed by Player B via the `parent` reference set in `players()`; this module does not enforce tensor types beyond the documented contract in the module docstring.

## Related

- `Q_Sea_Battle.trainable_assisted_player_a.TrainableAssistedPlayerA`
- `Q_Sea_Battle.trainable_assisted_player_b.TrainableAssistedPlayerB`
- `Q_Sea_Battle.lin_trainable_assisted_model_a.LinTrainableAssistedModelA`
- `Q_Sea_Battle.lin_trainable_assisted_model_b.LinTrainableAssistedModelB`

## Changelog

- 0.1: Initial version (as indicated by module docstring).