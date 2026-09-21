# TrainableAssistedPlayers

> Role: Factory/wrapper that provides a coordinated (A, B) PR-assisted player pair.

Location: `Q_Sea_Battle.trainable_assisted_players.TrainableAssistedPlayers`

## Derived constraints

The instance coordinates two trainable models and two player wrappers; the shared dimensions are derived from `game_layout.field_size` and `game_layout.comms_size`. Define $n2 = \text{field\_size}^2$, where `field_size` is the side length of the square field, and `comms_size` is the number of communication bits/logits.

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| `game_layout` | `Any`, must provide attributes `field_size: int` and `comms_size: int`, scalar | Game-layout-like object used to parameterize default models and to construct player wrappers. |
| `p_rule` | `float`, unconstrained, scalar | Unused by current linear models; retained for compatibility with other/older configurations. |
| `num_iterations` | `Optional[int]`, unconstrained, scalar | Unused by current linear models; retained for compatibility with other/older configurations. |
| `hidden_dim` | `int`, unconstrained, scalar | Unused by current linear models; retained for compatibility with other/older configurations. |
| `L_meas` | `Optional[int]`, unconstrained, scalar | Unused by current linear models; retained for compatibility with other/older configurations. |
| `model_a` | `Optional[LinTrainableAssistedModelA]`, scalar | Optional pre-constructed model for player A; if `None`, a default `LinTrainableAssistedModelA` is constructed from `game_layout`. |
| `model_b` | `Optional[LinTrainableAssistedModelB]`, scalar | Optional pre-constructed model for player B; if `None`, a default `LinTrainableAssistedModelB` is constructed from `game_layout`. |

Preconditions

- `game_layout` provides `field_size` and `comms_size` attributes readable via `getattr`.
- If `model_a is None` or `model_b is None`, `int(getattr(game_layout, "field_size"))` and `int(getattr(game_layout, "comms_size"))` must succeed.

Postconditions

- `self.game_layout` is set to `game_layout`.
- `self.explore` is initialized to `False`.
- `self._playerA` and `self._playerB` are initialized to `None` (lazy wrapper construction).
- `self.has_prev` is set to `True`.
- `self.previous` is initialized to `None`.
- `self.model_a` and `self.model_b` are set either from provided models or from newly constructed defaults using `sr_mode="sample"` and `seed=123`.

Errors

- Raises `AttributeError` or `TypeError` if `game_layout` lacks required attributes or they cannot be converted via `int(...)` when constructing default models.
- Any exception raised by `LinTrainableAssistedModelA(...)` or `LinTrainableAssistedModelB(...)` during default construction is propagated.

Example

!!! example "Construct and obtain wrapper players"
    ```python
    from Q_Sea_Battle.trainable_assisted_players import TrainableAssistedPlayers

    # game_layout must provide: field_size, comms_size
    players_factory = TrainableAssistedPlayers(game_layout)

    player_a, player_b = players_factory.players()

    players_factory.set_explore(True)
    players_factory.reset()
    ```

## Public Methods

### `check_model_correspondence()`

Check that model A and model B appear dimensionally compatible by comparing `field_size` and `comms_size` when available.

Returns

- `bool`, scalar: `True` if basic dimensions match (or cannot be checked due to missing attributes or other errors), otherwise `False`.

Errors

- No exceptions are intended to propagate; exceptions encountered while reading model attributes are caught and result in returning `True`.

### `players()`

Return the `(player A, player B)` wrappers, constructing them lazily and caching them so state persists across calls until `reset()`.

Returns

- `Tuple[TrainableAssistedPlayerA, TrainableAssistedPlayerB]`, shape `(2,)`: A tuple `(player_a, player_b)`.

Side effects

- On first call (or if a cached wrapper is missing), constructs `TrainableAssistedPlayerA(game_layout, model_a=self.model_a)` and/or `TrainableAssistedPlayerB(game_layout, model_b=self.model_b)`.
- Sets `player_a.explore` and `player_b.explore` to `self.explore`.
- Sets `player_a.parent` and `player_b.parent` to `self`.

Errors

- Propagates any exception thrown by the `TrainableAssistedPlayerA`/`TrainableAssistedPlayerB` constructors or attribute assignments.

### `reset()`

Reset per-game state by clearing cached `previous` tensors and forwarding reset to any already-instantiated player wrappers.

Returns

- `None`.

Side effects

- If instantiated, calls `self._playerA.reset()` and `self._playerB.reset()`.
- Sets `self.previous = None`.

Errors

- Propagates any exception thrown by the underlying wrapper `reset()` methods.

### `set_explore(flag)`

Enable or disable exploration for both players.

Parameters

- `flag`: `bool`, scalar: If `True`, players may sample actions and store log-probabilities (when supported); if `False`, they act greedily.

Returns

- `None`.

Side effects

- Sets `self.explore = bool(flag)`.
- If wrappers are instantiated, updates `self._playerA.explore` and `self._playerB.explore` to match.

Errors

- Propagates any exception thrown by setting `explore` on instantiated wrappers.

## Data & State

- `has_log_probs`: `bool`, scalar: Class attribute set to `True`.
- `game_layout`: `Any`, must provide `field_size: int` and `comms_size: int`, scalar: Stored reference to the game layout.
- `model_a`: `LinTrainableAssistedModelA`, scalar: Internal trainable model used by player A.
- `model_b`: `LinTrainableAssistedModelB`, scalar: Internal trainable model used by player B.
- `explore`: `bool`, scalar: Shared exploration flag propagated to wrapper players.
- `_playerA`: `Optional[TrainableAssistedPlayerA]`, scalar: Cached wrapper instance for player A, created lazily.
- `_playerB`: `Optional[TrainableAssistedPlayerB]`, scalar: Cached wrapper instance for player B, created lazily.
- `has_prev`: `bool`, scalar: Set to `True`; semantics are not specified in this module beyond indicating previous-state support.
- `previous`: `Any | None`, expected structure `(measurements_per_layer, outcomes_per_layer)` or `None`, scalar: Storage written by player A and consumed by player B; expected to be a 2-tuple of Python lists of tensors where each tensor is expected to have shape `(B, n2)`.

!!! note "Definition of n2 and tensor shape expectation"
    This module uses the convention $n2 = \text{field\_size}^2$. The module docstring states each tensor in `previous` is expected to have shape `(B, n2)` where `B` is the batch dimension; tensor dtype and framework are not specified here.

## Planned (design-spec)

Not specified.

## Deviations

Not specified.

## Notes for Contributors

- The constructor includes parameters (`p_rule`, `num_iterations`, `hidden_dim`, `L_meas`) that are currently unused by the default linear models; keep them in place if backward/forward compatibility with configuration code is required.
- `players()` sets `parent` on wrapper players; any wrapper implementation changes should preserve this linkage if other components depend on it.
- `check_model_correspondence()` is intentionally permissive: if model attributes are missing or inaccessible it returns `True`.

## Related

- `Q_Sea_Battle.trainable_assisted_player_a.TrainableAssistedPlayerA`
- `Q_Sea_Battle.trainable_assisted_player_b.TrainableAssistedPlayerB`
- `Q_Sea_Battle.lin_trainable_assisted_model_a.LinTrainableAssistedModelA`
- `Q_Sea_Battle.lin_trainable_assisted_model_b.LinTrainableAssistedModelB`

## Changelog

- Not specified.