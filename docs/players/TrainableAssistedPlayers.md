# Class TrainableAssistedPlayers

**Module import path**: `Q_Sea_Battle.trainable_assisted_players.TrainableAssistedPlayers`

> Factory that wires `TrainableAssistedPlayerA` and `TrainableAssistedPlayerB` around two Keras models
> (`LinTrainableAssistedModelA/B`) and maintains the shared per-game state `previous`.

!!! note "Terminology"
    This module uses `sr_mode` where **SR = shared resource** (not "shared randomness").

!!! note "Derived symbols"
    Let `field_size = n`, `comms_size = m`, and `n2 = n**2`.

## Overview

`TrainableAssistedPlayers` owns:

- `model_a`: `LinTrainableAssistedModelA`, which produces communication logits and intermediate tensors.
- `model_b`: `LinTrainableAssistedModelB`, which consumes the intermediate tensors + gun + comms to produce shoot logits.
- `previous`: shared state set by Player A and consumed by Player B.

The factory exposes a `players()` method returning `(TrainableAssistedPlayerA, TrainableAssistedPlayerB)` wrappers
compatible with the `Tournament`/`Game` interfaces.

## Constructor

### Signature

- `TrainableAssistedPlayers(game_layout: Any, p_high: float = 0.9, num_iterations: int | None = None,`
  `hidden_dim: int = 32, L_meas: int | None = None, model_a: LinTrainableAssistedModelA | None = None,`
  `model_b: LinTrainableAssistedModelB | None = None) -> TrainableAssistedPlayers`

### Arguments

- `game_layout`: `Any`, scalar.
  - Must provide attributes:
    - `field_size`: `int`, scalar.
    - `comms_size`: `int`, scalar.
- `p_high`: `float`, scalar.
  - Present for forward compatibility (not necessarily used by the linear models).
- `num_iterations`: `int` or `None`, scalar.
  - Present for forward compatibility.
- `hidden_dim`: `int`, scalar.
  - Present for forward compatibility.
- `L_meas`: `int` or `None`, scalar.
  - Present for forward compatibility.
- `model_a`: `LinTrainableAssistedModelA` or `None`, scalar.
- `model_b`: `LinTrainableAssistedModelB` or `None`, scalar.

### Returns

- `TrainableAssistedPlayers`, scalar.

### Preconditions

- `game_layout.field_size` and `game_layout.comms_size` exist and are convertible to `int`.
- If `model_a` is `None`, a default model is constructed with `field_size=n`, `comms_size=m`.
- If `model_b` is `None`, a default model is constructed with `field_size=n`, `comms_size=m`.

### Postconditions

- `self.game_layout` is set.
- `self.model_a`, `self.model_b` are set.
- `self.previous` is initialised to `None`.
- Internal cached players are initialised to `None`.

### Errors

- Propagates exceptions raised by `LinTrainableAssistedModelA/B` construction.
- May raise `AttributeError` or `TypeError` if `game_layout` lacks required attributes.

## Public Methods

### players

#### Signature

- `players() -> tuple[TrainableAssistedPlayerA, TrainableAssistedPlayerB]`

#### Arguments

- None.

#### Returns

- `player_a`: `TrainableAssistedPlayerA`, scalar.
- `player_b`: `TrainableAssistedPlayerB`, scalar.

#### Preconditions

- Factory constructed successfully.

#### Postconditions

- Creates wrappers lazily on first call.
- Sets `player_a.parent = self` and `player_b.parent = self`.
- Sets `player_a.explore = self.explore` and `player_b.explore = self.explore`.

#### Errors

- Propagates exceptions from wrapper construction.

!!! example "Minimal usage"
    ```python
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.trainable_assisted_players import TrainableAssistedPlayers

    layout = GameLayout(field_size=4, comms_size=1)
    players = TrainableAssistedPlayers(layout)

    player_a, player_b = players.players()
    ```

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

- Calls `reset()` on cached player wrappers (if present).
- Sets `self.previous = None`.

#### Errors

- Propagates unexpected exceptions from player wrapper resets.

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

- Sets `self.explore` and propagates to existing player wrappers.

#### Errors

- None.

### check_model_correspondence

#### Signature

- `check_model_correspondence() -> bool`

#### Arguments

- None.

#### Returns

- `ok`: `bool`, scalar.

#### Preconditions

- None.

#### Postconditions

- Returns `True` if `model_a` and `model_b` expose compatible `field_size` and `comms_size` attributes.

#### Errors

- None (returns `True` on unexpected inspection failures).

## Shared State Contract

### previous

- `previous`: `tuple[list[tf.Tensor], list[tf.Tensor]]` or `None`, scalar.

Intended meaning:
- First element: list of measurement tensors per layer/step.
- Second element: list of outcome tensors per layer/step.

!!! warning "Shape depends on the model"
    The exact tensor shapes in `previous` are determined by `LinTrainableAssistedModelA.compute_with_internal(...)` and
    how `LinTrainableAssistedModelB` consumes them. Do not assume `shape == (B, n2)` unless validated for your model.

## Data & State

- `game_layout`: `Any`, scalar.
- `model_a`: `LinTrainableAssistedModelA`, scalar.
- `model_b`: `LinTrainableAssistedModelB`, scalar.
- `explore`: `bool`, scalar.
- `previous`: `Any | None`, scalar (typically `(meas_list, out_list)`).
- `_playerA`: `TrainableAssistedPlayerA | None`, scalar.
- `_playerB`: `TrainableAssistedPlayerB | None`, scalar.

## Planned (design-spec)

- None identified from the provided implementation.

## Deviations

- The module docstring describes each `previous` tensor as shaped `(B, n2)`, but `TrainableAssistedPlayerB` accepts both
  list and non-list forms and normalises them to lists at runtime. This suggests the exact `previous` tensor shapes may
  vary by architecture and should be treated as model-defined.

## Notes for Contributors

- Keep `sr_mode` naming consistent across trainable modules, but document it as **shared resource**.
- Any change to the `previous` contract must update both player wrappers and any tournament logging that inspects it.
- Prefer maintaining lazy construction and cached wrappers to preserve per-game state.

## Changelog

- 2026-01-11 — Author: Rob Hendriks
