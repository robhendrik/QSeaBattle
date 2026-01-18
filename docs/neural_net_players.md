# NeuralNetPlayers

> Role: Factory and manager for a shared pair of neural-network-based Sea Battle players (Player A for communication, Player B for shooting), including model persistence and separate training entry points.

Location: `Q_Sea_Battle.neural_net_players.NeuralNetPlayers`

## Constructor

Parameter | Type | Description
--- | --- | ---
game_layout | GameLayout or None, default None, constraint: if None then a default `GameLayout()` is constructed | Game configuration used to derive dimensions such as field_size and comms_size.
model_a | tf.keras.Model or None, default None | Optional pre-constructed communication model for Player A; if None, built lazily on first use.
model_b | tf.keras.Model or None, default None | Optional pre-constructed shoot model for Player B; if None, built lazily on first use.
explore | bool, default False | Exploration flag propagated to created child players; True enables stochastic behavior, False deterministic thresholding.

Preconditions

- `game_layout` is either None or a `GameLayout` instance.
- If provided, `model_a` and `model_b` are compatible with the input/output shapes implied by the current `game_layout` (compatibility is not validated here).

Postconditions

- `self.game_layout: GameLayout` is set (constructed if `game_layout is None`).
- `self.model_a: tf.keras.Model or None` and `self.model_b: tf.keras.Model or None` are stored without modification.
- Internal cached players `self._playerA` and `self._playerB` are initialized to None.
- `self.explore: bool` is set.

Errors

- Not specified; downstream errors may be raised by `GameLayout()` construction or by incompatible model usage later.

Example

```python
from Q_Sea_Battle.neural_net_players import NeuralNetPlayers

factory = NeuralNetPlayers(explore=True)
player_a, player_b = factory.players()
```

## Public Methods

### players()

Create or return a neural Player A/B pair, lazily constructing default models and caching created players.

Returns

- Tuple[PlayerA, PlayerB], shape (2,), constraints: `(player_a, player_b)` where instances are cached and reused across calls.

Side effects

- If `self.model_a is None`, assigns `self.model_a = self._build_model_a()`.
- If `self.model_b is None`, assigns `self.model_b = self._build_model_b()`.
- If cached players are missing, constructs `NeuralNetPlayerA` and/or `NeuralNetPlayerB` with `game_layout=self.game_layout`, the relevant model, and `explore=self.explore`.

Errors

- Not specified; may raise errors from TensorFlow/Keras model construction or player constructors.

Example

```python
player_a, player_b = factory.players()
```

### reset()

Reset internal state of the neural players (clears per-game log-probabilities) without modifying the underlying Keras models.

Returns

- None

Side effects

- Calls `self._playerA.reset()` if `self._playerA is not None`.
- Calls `self._playerB.reset()` if `self._playerB is not None`.

Errors

- Not specified.

Example

```python
factory.reset()
```

### set_explore(flag)

Set exploration behavior for both cached players (if they exist) and update the factory flag used for newly created players.

Parameters

- flag: bool, constraints: {True, False}

Returns

- None

Side effects

- Sets `self.explore = flag`.
- If cached players exist, sets `self._playerA.explore = flag` and `self._playerB.explore = flag`.

Errors

- Not specified.

Example

```python
factory.set_explore(False)
```

### store_models(filenameA, filenameB)

Store the underlying Keras models to disk, lazily building them first if needed.

Parameters

- filenameA: str, constraints: path-like string accepted by `tf.keras.Model.save`
- filenameB: str, constraints: path-like string accepted by `tf.keras.Model.save`

Returns

- None

Side effects

- Ensures `self.model_a` and `self.model_b` are non-None by building defaults if needed.
- Calls `self.model_a.save(filenameA)` and `self.model_b.save(filenameB)`.

Errors

- Not specified; may raise I/O errors or TensorFlow/Keras serialization errors.

Example

```python
factory.store_models("model_a.keras", "model_b.keras")
```

### load_models(filenameA, filenameB)

Load Keras models from disk, attach them to this factory, and update existing cached players to reference the loaded models.

Parameters

- filenameA: str, constraints: path-like string accepted by `tf.keras.models.load_model`
- filenameB: str, constraints: path-like string accepted by `tf.keras.models.load_model`

Returns

- None

Side effects

- Sets `self.model_a = tf.keras.models.load_model(filenameA)`.
- Sets `self.model_b = tf.keras.models.load_model(filenameB)`.
- If cached players exist, updates `self._playerA.model_a` and/or `self._playerB.model_b` to point to the newly loaded models.

Errors

- Not specified; may raise I/O errors or TensorFlow/Keras deserialization errors.

Example

```python
factory.load_models("model_a.keras", "model_b.keras")
```

### train(dataset, training_settings)

Legacy training API; currently a no-op that emits a deprecation warning.

Parameters

- dataset: Unknown, constraints: Not specified
- training_settings: Unknown, constraints: Not specified

Returns

- None, constraints: always returns without training

Side effects

- Emits `UserWarning` via `warnings.warn(...)`.

Errors

- Not specified.

Example

```python
factory.train(dataset, training_settings)
```

### train_model_a(dataset, training_settings)

Train the communication model (model_a) on a dataset.

Parameters

- dataset: pandas.DataFrame, constraints: must contain column `"field"` with array-like per-row entries and column `"comm"` with array-like per-row entries; optional column `"sample_weight"` if enabled; shapes: `"field"` entries reshape to (n2,) and `"comm"` entries reshape to (m,)
- training_settings: dict-like, constraints: supports keys `"use_sample_weight"` (bool-like), `"epochs"` (int-like), `"batch_size"` (int-like), `"learning_rate"` (float-like), `"verbose"` (int-like)

Returns

- None

Behavior

- Defines $n2 = \mathrm{field\_size}^2$ and $m = \mathrm{comms\_size}$ from `self.game_layout`.
- Builds `self.model_a` via `_build_model_a()` if it is None.
- Constructs training inputs: `fields_scaled` is produced by stacking `dataset["field"]`, reshaping to `(-1, n2)`, converting to float32, and applying `_scale_field(...)`.
- Constructs targets: `comms_teacher` by stacking `dataset["comm"]`, reshaping to `(-1, m)`, converting to float32.
- Optionally uses `sample_weight` if `training_settings["use_sample_weight"]` is truthy and `"sample_weight"` exists in `dataset.columns`.
- Compiles `self.model_a` with `Adam(learning_rate)`, `BinaryCrossentropy(from_logits=True)`, and metric `"accuracy"`.
- Calls `.fit(...)` with `epochs`, `batch_size`, and `verbose` from `training_settings` (defaults: 3, 32, 1e-3, 0).

Errors

- Not specified; may raise KeyError for missing required columns, NumPy reshape/stack errors for inconsistent shapes, and TensorFlow/Keras training errors.

Example

```python
training_settings = {"epochs": 5, "batch_size": 64, "learning_rate": 1e-3, "verbose": 1}
factory.train_model_a(dataset=df, training_settings=training_settings)
```

### train_model_b(dataset, training_settings)

Train the shoot model (model_b) on a dataset.

Parameters

- dataset: pandas.DataFrame, constraints: must contain column `"gun"` with array-like per-row entries, column `"comm"` with array-like per-row entries, and column `"shoot"` with scalar/array-like per-row entries; optional column `"sample_weight"` if enabled; shapes: `"gun"` entries reshape to (n2,), `"comm"` entries reshape to (m,), `"shoot"` reshapes to (1,)
- training_settings: dict-like, constraints: supports keys `"use_sample_weight"` (bool-like), `"epochs"` (int-like), `"batch_size"` (int-like), `"learning_rate"` (float-like), `"verbose"` (int-like)

Returns

- None

Behavior

- Defines $n2 = \mathrm{field\_size}^2$ and $m = \mathrm{comms\_size}$ from `self.game_layout`.
- Builds `self.model_b` via `_build_model_b()` if it is None.
- Constructs gun index feature: stacks `dataset["gun"]`, reshapes to `(-1, n2)`, converts to float32, then calls `_gun_one_hot_to_index(guns)` to obtain `gun_idx_norm`, documented in code as shape `(N, 1)`.
- Constructs communication feature: stacks `dataset["comm"]`, reshapes to `(-1, m)`, converts to float32.
- Concatenates features: `x = np.concatenate([gun_idx_norm, comms], axis=1)`, yielding shape `(N, 1 + m)`.
- Constructs targets: `shoots = dataset["shoot"].to_numpy().astype("float32").reshape((-1, 1))`.
- Optionally uses `sample_weight` if `training_settings["use_sample_weight"]` is truthy and `"sample_weight"` exists in `dataset.columns`.
- Compiles `self.model_b` with `Adam(learning_rate)`, `BinaryCrossentropy(from_logits=True)`, and metric `"accuracy"`.
- Calls `.fit(...)` with `epochs`, `batch_size`, and `verbose` from `training_settings` (defaults: 3, 32, 1e-3, 0).

Errors

- Not specified; may raise KeyError for missing required columns, NumPy shape/concat errors, and TensorFlow/Keras training errors.

Example

```python
training_settings = {"epochs": 3, "batch_size": 32, "learning_rate": 1e-3, "verbose": 0}
factory.train_model_b(dataset=df, training_settings=training_settings)
```

## Data & State

- has_log_probs: bool, constraints: always True in this implementation; indicates Tournament should attempt to read log-probabilities via `get_log_prob` from underlying players.
- game_layout: GameLayout, constraints: non-None after construction; used to derive `field_size` and `comms_size`.
- explore: bool, constraints: {True, False}; propagated to created/cached players and used when creating new ones.
- model_a: tf.keras.Model or None, constraints: if non-None should map input shape (n2,) to output shape (m,) with logits semantics (trained with `BinaryCrossentropy(from_logits=True)`).
- model_b: tf.keras.Model or None, constraints: if non-None should map input shape (1 + m,) to output shape (1,) with logits semantics (trained with `BinaryCrossentropy(from_logits=True)`).
- _playerA: NeuralNetPlayerA or None, constraints: cached instance; created lazily by `players()`.
- _playerB: NeuralNetPlayerB or None, constraints: cached instance; created lazily by `players()`.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- Dimension symbols used throughout training are derived from `self.game_layout`: $n2 = \mathrm{field\_size}^2$ and $m = \mathrm{comms\_size}$.
- Default model architectures are constructed lazily; changes to `game_layout` after instantiation may desynchronize expected shapes from previously built models (no validation is performed here).
- `train()` is retained only for backward compatibility and intentionally performs no training; callers should be migrated to `train_model_a()` and `train_model_b()`.

## Related

- `Q_Sea_Battle.players_base.Players` (base factory interface)
- `Q_Sea_Battle.neural_net_player_a.NeuralNetPlayerA` and `_scale_field`
- `Q_Sea_Battle.neural_net_player_b.NeuralNetPlayerB` and `_gun_one_hot_to_index`
- `Q_Sea_Battle.game_layout.GameLayout`

## Changelog

- 0.1: Initial version in module header; `train_model_a` and `train_model_b` present alongside deprecated no-op `train()`.