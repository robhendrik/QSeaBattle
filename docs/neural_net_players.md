# NeuralNetPlayers

> Role: Factory that constructs, caches, persists, and trains a matched pair of neural-network players (Player A for communication logits; Player B for shoot logits) sharing underlying Keras models.
Location: `Q_Sea_Battle.neural_net_players.NeuralNetPlayers`

## Derived constraints

Symbols: `field_size` = `game_layout.field_size` (int, $>0$), `comms_size` = `game_layout.comms_size` (int, $>0$), `n2 = field_size^2` (int, $>0$), `m = comms_size` (int, $>0$).  
Default model interfaces: `model_a` input shape `(n2,)` output shape `(m,)`; `model_b` input shape `(1 + m,)` output shape `(1,)`.

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | Optional[`Q_Sea_Battle.game_layout.GameLayout`], nullable | Game layout used to derive `field_size`, `comms_size`; if `None`, a default `GameLayout()` is constructed. |
| model_a | Optional[`tf.keras.Model`], nullable | Optional pre-constructed model for Player A; if `None`, a default architecture is built on first use. |
| model_b | Optional[`tf.keras.Model`], nullable | Optional pre-constructed model for Player B; if `None`, a default architecture is built on first use. |
| explore | bool, constraint: `{True, False}` | Exploration flag propagated to created players. |

Preconditions: `game_layout is None` or `game_layout` is a `GameLayout` instance; if `model_a` is provided it must be compatible with input `tf.Tensor, dtype float32, shape (B, n2)` and output `tf.Tensor, dtype float32, shape (B, m)`; if `model_b` is provided it must be compatible with input `tf.Tensor, dtype float32, shape (B, 1+m)` and output `tf.Tensor, dtype float32, shape (B, 1)`.
Postconditions: `self.game_layout` is set (never `None`); `self.model_a` and `self.model_b` are stored as provided; players are not created (`self._playerA is None` and `self._playerB is None`); `self.explore` equals `explore`.
Errors: Not specified.
Example:
```python
from Q_Sea_Battle.neural_net_players import NeuralNetPlayers

factory = NeuralNetPlayers(explore=False)
player_a, player_b = factory.players()
```

## Public Methods

### players

Create or return a neural Player A/B pair; lazily builds default models if missing and caches created players.

Arguments: None.
Returns: Tuple[`Q_Sea_Battle.players_base.PlayerA`, `Q_Sea_Battle.players_base.PlayerB`], shape `(2,)` as `(player_a, player_b)`.
Errors: Not specified.

### reset

Reset per-game state of the created players without modifying underlying Keras model parameters.

Arguments: None.
Returns: `None`.
Errors: Not specified.

### set_explore

Set exploration behavior for both created players (and updates already-created players in-place).

Arguments:
- `flag`: bool, constraint: `{True, False}`; if `True`, players act stochastically; if `False`, players act deterministically.
Returns: `None`.
Errors: Not specified.

### store_models

Serialize the underlying Keras models to disk; builds default models if missing.

Arguments:
- `filenameA`: str, constraint: non-empty path-like string; output path for Player A model.
- `filenameB`: str, constraint: non-empty path-like string; output path for Player B model.
Returns: `None`.
Errors: Not specified.

### load_models

Load Keras models from disk and attach them to this factory; updates already-created players in-place to reference the newly loaded models.

Arguments:
- `filenameA`: str, constraint: non-empty path-like string; path to serialized Player A model.
- `filenameB`: str, constraint: non-empty path-like string; path to serialized Player B model.
Returns: `None`.
Errors: Not specified.

### train

Legacy training API (no-op); emits a deprecation warning and returns without performing training.

Arguments:
- `dataset`: Unknown; not specified (accepted for backward compatibility).
- `training_settings`: Unknown; not specified (accepted for backward compatibility).
Returns: `None`.
Errors: Emits `UserWarning` via `warnings.warn(...)`.

### train_model_a

Train the communication model (`model_a`) using binary cross-entropy from logits.

Arguments:
- `dataset`: pandas DataFrame-like, constraint: must support `dataset["field"]`, `dataset["comm"]`, `dataset.columns`, and `.to_numpy()`; expected to contain columns: `field`, `comm`, and optionally `sample_weight`.
- `training_settings`: Mapping-like (e.g., `dict`), constraint: must support `.get(key, default)`; supported keys include `epochs`, `batch_size`, `learning_rate`, `verbose`, `use_sample_weight`.
Returns: `None`.
Errors: Not specified.

### train_model_b

Train the shoot model (`model_b`) using binary cross-entropy from logits; converts gun one-hot to normalized index and concatenates with communication vector.

Arguments:
- `dataset`: pandas DataFrame-like, constraint: must support `dataset["gun"]`, `dataset["comm"]`, `dataset["shoot"]`, `dataset.columns`, and `.to_numpy()`; expected to contain columns: `gun`, `comm`, `shoot`, and optionally `sample_weight`.
- `training_settings`: Mapping-like (e.g., `dict`), constraint: must support `.get(key, default)`; supported keys include `epochs`, `batch_size`, `learning_rate`, `verbose`, `use_sample_weight`.
Returns: `None`.
Errors: Not specified.

## Data & State

- `has_log_probs`: bool, constant-like class attribute, constraint: `{True}`, meaning tournament logic may attempt to read log-probabilities via `get_log_prob` from underlying players.
- `game_layout`: `Q_Sea_Battle.game_layout.GameLayout`, non-null; used to derive `field_size`, `comms_size`, `n2`, `m`.
- `explore`: bool, constraint: `{True, False}`; propagated to created players; can be updated via `set_explore`.
- `model_a`: Optional[`tf.keras.Model`], nullable; when built by default, maps `tf.Tensor, dtype float32, shape (B, n2)` to `tf.Tensor, dtype float32, shape (B, m)`.
- `model_b`: Optional[`tf.keras.Model`], nullable; when built by default, maps `tf.Tensor, dtype float32, shape (B, 1+m)` to `tf.Tensor, dtype float32, shape (B, 1)`.
- `_playerA`: Optional[`Q_Sea_Battle.neural_net_player_a.NeuralNetPlayerA`], nullable; lazily created and cached.
- `_playerB`: Optional[`Q_Sea_Battle.neural_net_player_b.NeuralNetPlayerB`], nullable; lazily created and cached.

## Planned (design-spec)

Not specified.

## Deviations

Not specified.

## Notes for Contributors

- This module enables eager execution globally via `tf.config.run_functions_eagerly(True)` at import time; changing this may affect debugging and performance characteristics.
- Training methods assume dataset entries are stackable into dense NumPy arrays and reshapeable to `(-1, n2)`, `(-1, m)`, and `(-1, 1)` as applicable; ensure upstream data preparation matches these expectations.
- Default model architectures are built in `_build_model_a` and `_build_model_b`; changes must preserve the documented input/output shapes driven by `n2` and `m`.

## Related

- `Q_Sea_Battle.neural_net_player_a.NeuralNetPlayerA` (uses `model_a`)
- `Q_Sea_Battle.neural_net_player_b.NeuralNetPlayerB` (uses `model_b`)
- `Q_Sea_Battle.players_base.Players`, `Q_Sea_Battle.players_base.PlayerA`, `Q_Sea_Battle.players_base.PlayerB`
- `Q_Sea_Battle.game_layout.GameLayout`

## Changelog

Not specified.