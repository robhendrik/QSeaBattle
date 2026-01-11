# Class TrainableAssistedPlayerA

**Module import path**: `Q_Sea_Battle.trainable_assisted_player_a.TrainableAssistedPlayerA`

> Player A wrapper around `LinTrainableAssistedModelA`.
> Produces communication bits from a field and stores intermediate tensors on `parent.previous`.

!!! note "Derived symbols"
    Let `field_size = n`, `comms_size = m`, and `n2 = n**2`.

## Overview

`TrainableAssistedPlayerA` converts the field into a batched tensor, runs `model_a`, and produces a communication vector.

- Exploration mode:
  - `explore == False`: greedy threshold at 0.5.
  - `explore == True`: samples independent Bernoulli bits.

It stores model intermediates for Player B:

- `parent.previous = (meas_list, out_list)`

## Constructor

### Signature

- `TrainableAssistedPlayerA(game_layout: Any, model_a: LinTrainableAssistedModelA) -> TrainableAssistedPlayerA`

### Arguments

- `game_layout`: `Any`, scalar, provides `field_size: int` and `comms_size: int`.
- `model_a`: `LinTrainableAssistedModelA`, scalar.

### Returns

- `TrainableAssistedPlayerA`, scalar.

### Preconditions

- `field_size` and `comms_size` exist on `game_layout`.
- `model_a.compute_with_internal(field_batch)` is available.

### Postconditions

- `self.parent` is initialised to `None` (to be set by `TrainableAssistedPlayers.players()`).
- `self.last_logprob_comm` is initialised to `None`.
- `self.explore` is initialised to `False`.

### Errors

- Propagates exceptions from model calls or invalid `game_layout` attributes.

## Public Methods

### decide

#### Signature

- `decide(field: np.ndarray, supp: Any | None = None, explore: bool | None = None) -> np.ndarray`

#### Arguments

- `field`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)`.
- `supp`: `Any` or `None`, scalar. Ignored.
- `explore`: `bool` or `None`, scalar. Optional override of `self.explore`.

#### Returns

- `comm`: `np.ndarray`, dtype `int` {0,1}, shape `(m,)`.

#### Preconditions

- `field.shape == (n2,)`.
- `field` contains only values in `{0,1}`.
- `model_a.compute_with_internal(...)` returns logits compatible with `m`.

#### Postconditions

- If `self.parent` is set, updates:
  - `self.parent.previous = (meas_list, out_list)`.
- Sets `self.last_logprob_comm`: `float`, scalar.

#### Errors

- Raises `ValueError` if `field` shape is invalid.
- Raises `ValueError` if `field` contains values outside `{0,1}`.
- Propagates exceptions from TensorFlow/model execution.

!!! example "Minimal usage"
    ```python
    import numpy as np
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.trainable_assisted_players import TrainableAssistedPlayers

    layout = GameLayout(field_size=4, comms_size=1)
    factory = TrainableAssistedPlayers(layout)
    player_a, _ = factory.players()

    field = np.zeros(layout.field_size**2, dtype=int)
    comm = player_a.decide(field)
    ```

### get_log_prob

#### Signature

- `get_log_prob() -> float`

#### Arguments

- None.

#### Returns

- `logp`: `float`, scalar.

#### Preconditions

- `decide(...)` has been called since `reset()`.

#### Postconditions

- No mutation.

#### Errors

- Raises `RuntimeError` if no log-prob is available.

### get_prev

#### Signature

- `get_prev() -> Any | None`

#### Arguments

- None.

#### Returns

- `prev`: `Any | None`, scalar.
  - Typically `(meas_list, out_list)`.

#### Preconditions

- None.

#### Postconditions

- No mutation.

#### Errors

- None.

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

- Sets `self.last_logprob_comm = None`.

#### Errors

- None.

## Data & State

- `game_layout`: `Any`, scalar.
- `model_a`: `LinTrainableAssistedModelA`, scalar.
- `parent`: `TrainableAssistedPlayers | None`, scalar.
- `last_logprob_comm`: `float | None`, scalar.
- `explore`: `bool`, scalar.

## Planned (design-spec)

- None identified from the provided implementation.

## Deviations

- None identified.

## Notes for Contributors

- The `previous` contract is owned by the factory; ensure changes are coordinated with Player B.
- If you change sampling semantics, update `bernoulli_log_prob_from_logits` usage accordingly.

## Changelog

- 2026-01-11 — Author: Rob Hendriks
