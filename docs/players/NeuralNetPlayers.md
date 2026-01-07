# Class NeuralNetPlayers

**Module import path**: `Q_Sea_Battle.neural_net_players.NeuralNetPlayers`

> Factory that owns two Keras models and serves a cached `(NeuralNetPlayerA, NeuralNetPlayerB)` pair.

!!! note "Derived constraints from GameLayout"
    Let `field_size = n` and `n2 = n**2`. Let `comms_size = m`.

    - `n2` is a power of two.
    - `m | n2` (i.e. `n2 % m == 0`).

## Overview

`NeuralNetPlayers` is a `Players`-style factory that:

- Holds (or builds) `model_a` and `model_b` (`tf.keras.Model`).
- Constructs and caches `NeuralNetPlayerA` and `NeuralNetPlayerB` wrappers for inference and logging.
- Provides convenience methods to train, save, and load the two models.

!!! warning "TensorFlow side effects"
    The module may enable eager execution for TensorFlow functions at import time in the current implementation.
    This can affect performance and tracing behaviour across your process.

## Constructor

### Signature

- `NeuralNetPlayers(game_layout=None, model_a=None, model_b=None, explore=True)`

### Arguments

- `game_layout`: `GameLayout` or `None`, scalar.
- `model_a`: `tf.keras.Model` or `None`, scalar.
- `model_b`: `tf.keras.Model` or `None`, scalar.
- `explore`: `bool`, scalar.

### Returns

- `NeuralNetPlayers`, scalar.

### Preconditions

- If provided, `game_layout` is a valid `GameLayout`, scalar.
- If `game_layout` is `None`, `GameLayout()` construction succeeds.
- If provided, `model_a` and `model_b` are compatible with the shapes below.

### Postconditions

- `self.game_layout` is a valid `GameLayout`, scalar.
- `self.model_a` and `self.model_b` reference the provided models (or remain `None` until built).
- `self.explore` is set to the provided value.
- Internal cached players (`self._playerA`, `self._playerB`) are initialised to `None`.

### Errors

- Propagates exceptions raised by `GameLayout()` when `game_layout` is `None`.
- Propagates exceptions raised by model building when models are auto-built.

## Public Methods

### players

#### Signature

- `players() -> tuple[NeuralNetPlayerA, NeuralNetPlayerB]`

#### Arguments

- None.

#### Returns

- `(player_a, player_b)` where:
  - `player_a`: `NeuralNetPlayerA`, scalar.
  - `player_b`: `NeuralNetPlayerB`, scalar.

#### Preconditions

- `self.game_layout` is valid.
- If `self.model_a` is `None`, `_build_model_a()` succeeds.
- If `self.model_b` is `None`, `_build_model_b()` succeeds.

#### Postconditions

- If missing, models are built and attached to the instance.
- Cached player wrappers are constructed once and reused on subsequent calls.

#### Errors

- Propagates exceptions from model creation or player construction.

!!! note "Model I/O shapes"
    - `model_a` input:
      - `field_scaled`: `tf.Tensor`, dtype `float32`, shape `(B, n2)`.
    - `model_a` output:
      - `comm_logits`: `tf.Tensor`, dtype `float32`, shape `(B, m)`.

    - `model_b` input:
      - `gunidx_comm`: `tf.Tensor`, dtype `float32`, shape `(B, 1 + m)`.
        - first feature: normalised gun index in `[0.0, 1.0]`
        - remaining features: communication bits
    - `model_b` output:
      - `shoot_logit`: `tf.Tensor`, dtype `float32`, shape `(B, 1)`.

### reset

#### Signature

- `reset() -> None`

#### Arguments

- None.

#### Returns

- `None`.

#### Preconditions

- None.

#### Postconditions

- If cached players exist, calls `reset()` on each cached player wrapper.

#### Errors

- No explicit errors are raised by the base implementation.

### set_explore

#### Signature

- `set_explore(flag: bool) -> None`

#### Arguments

- `flag`: `bool`, scalar.

#### Returns

- `None`.

#### Preconditions

- None.

#### Postconditions

- Sets `self.explore = flag`.
- If cached players exist, sets `player.explore = flag` on both.

#### Errors

- No explicit errors are raised by the base implementation.

### store_models

#### Signature

- `store_models(filenameA: str, filenameB: str) -> None`

#### Arguments

- `filenameA`: `str`, scalar.
- `filenameB`: `str`, scalar.

#### Returns

- `None`.

#### Preconditions

- If `model_a` or `model_b` is missing, the corresponding default builder succeeds.

#### Postconditions

- Both models are saved to the given paths using `tf.keras.Model.save`.

#### Errors

- Propagates exceptions from TensorFlow/Keras save operations.

### load_models

#### Signature

- `load_models(filenameA: str, filenameB: str) -> None`

#### Arguments

- `filenameA`: `str`, scalar.
- `filenameB`: `str`, scalar.

#### Returns

- `None`.

#### Preconditions

- Both files exist and are readable by `tf.keras.models.load_model`.

#### Postconditions

- `self.model_a` and `self.model_b` reference the loaded models.
- If cached players exist, their internal model references are updated.

#### Errors

- Propagates exceptions from TensorFlow/Keras load operations.

### train

#### Signature

- `train(dataset, training_settings) -> None`

#### Arguments

- `dataset`: `Any`, scalar.
- `training_settings`: `Any`, scalar.

#### Returns

- `None`.

#### Preconditions

- None.

#### Postconditions

- No training is performed.
- Emits a warning indicating deprecation.

#### Errors

- No explicit errors are raised (warning emission may be configurable in some environments).

### train_model_a

#### Signature

- `train_model_a(dataset: pd.DataFrame, training_settings: dict[str, Any]) -> None`

#### Arguments

- `dataset`: `pd.DataFrame`, rows `N`, columns include:
  - `field`: array-like per row, intended `np.ndarray`, dtype `int` (0, 1), shape `(n2,)`
  - `comm`: array-like per row, intended `np.ndarray`, dtype `int` (0, 1), shape `(m,)`
  - `sample_weight` (optional): float scalar per row
- `training_settings`: `dict[str, Any]`, scalar, supported keys:
  - `use_sample_weight`: `bool`, scalar (default `False`)
  - `epochs`: `int`, scalar (default `3`)
  - `batch_size`: `int`, scalar (default `32`)
  - `learning_rate`: `float`, scalar (default `1e-3`)
  - `verbose`: `int`, scalar (default `0`)

#### Returns

- `None`.

#### Preconditions

- Stackable training tensors can be formed:
  - `x_field`: `np.ndarray`, dtype `float32`, shape `(N, n2)`
  - `y_comm`: `np.ndarray`, dtype `float32`, shape `(N, m)`
- If `use_sample_weight` is true:
  - `w`: `np.ndarray`, dtype `float32`, shape `(N,)`.

#### Postconditions

- `model_a` is compiled for binary classification from logits.
- `model_a` weights are updated by `model.fit(...)`.

#### Errors

- Propagates exceptions from:
  - dataset stacking/reshaping,
  - Keras compile/fit,
  - invalid `training_settings` types or missing dataset columns.

### train_model_b

#### Signature

- `train_model_b(dataset: pd.DataFrame, training_settings: dict[str, Any]) -> None`

#### Arguments

- `dataset`: `pd.DataFrame`, rows `N`, columns include:
  - `gun`: array-like per row, intended `np.ndarray`, dtype `int` (0, 1), shape `(n2,)`, one-hot
  - `comm`: array-like per row, intended `np.ndarray`, dtype `int` (0, 1), shape `(m,)`
  - `shoot`: scalar per row, intended `int` (0, 1)
  - `sample_weight` (optional): float scalar per row
- `training_settings`: `dict[str, Any]`, scalar, supported keys as in `train_model_a`.

#### Returns

- `None`.

#### Preconditions

- Stackable training tensors can be formed:
  - `x_gun`: `np.ndarray`, dtype `float32`, shape `(N, n2)`
  - `x_comm`: `np.ndarray`, dtype `float32`, shape `(N, m)`
  - derived `x`: `np.ndarray`, dtype `float32`, shape `(N, 1 + m)`
  - `y`: `np.ndarray`, dtype `float32`, shape `(N, 1)`
- If `use_sample_weight` is true:
  - `w`: `np.ndarray`, dtype `float32`, shape `(N,)`.

#### Postconditions

- `model_b` is compiled for binary classification from logits.
- `model_b` weights are updated by `model.fit(...)`.

#### Errors

- Propagates exceptions from:
  - dataset stacking/reshaping,
  - Keras compile/fit,
  - invalid `training_settings` types or missing dataset columns.

## Data & State

- `game_layout`: `GameLayout`, scalar.
- `explore`: `bool`, scalar.
- `model_a`: `tf.keras.Model` or `None`, scalar.
- `model_b`: `tf.keras.Model` or `None`, scalar.
- `_playerA`: `NeuralNetPlayerA` or `None`, scalar.
- `_playerB`: `NeuralNetPlayerB` or `None`, scalar.

## Planned (design-spec)

- None identified.

## Deviations

- Naming:
  - Requested in task: `NeuraNetPlayers`, `NeuraNetPlayerA`, `NeuraNetPlayerB`.
  - Implemented in codebase: `NeuralNetPlayers`, `NeuralNetPlayerA`, `NeuralNetPlayerB`.

## Notes for Contributors

- Keep preprocessing stable across training and inference:
  - `model_a` uses a scaled field representation (`field - 0.5`).
  - `model_b` uses a scalar gun-index encoding plus `comm`.
- If you change representations, update both:
  - training (`train_model_a`, `train_model_b`), and
  - runtime wrappers (`NeuralNetPlayerA`, `NeuralNetPlayerB`).
- Consider removing or scoping eager-execution configuration if it is not strictly required.

## Examples

!!! example "Minimal usage"
    ```python
    from Q_Sea_Battle.neural_net_players import NeuralNetPlayers

    players = NeuralNetPlayers(explore=False)
    player_a, player_b = players.players()
    ```

## Changelog

- 2026-01-07 — Author: Rob Hendriks
