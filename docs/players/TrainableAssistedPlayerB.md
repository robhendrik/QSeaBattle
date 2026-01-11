# Class TrainableAssistedPlayerB

**Module import path**: `Q_Sea_Battle.trainable_assisted_player_b.TrainableAssistedPlayerB`

> Player B wrapper around `LinTrainableAssistedModelB`.
> Consumes `parent.previous`, combines it with gun + comm, and decides a shoot action.

!!! note "Derived symbols"
    Let `field_size = n`, `comms_size = m`, and `n2 = n**2`.

## Overview

`TrainableAssistedPlayerB` expects Player A to have run first and stored intermediates on the shared parent:

- `parent.previous = (prev_meas_list, prev_out_list)`

It then runs `model_b` to obtain a shoot logit and decides 0/1 via greedy or sampling policy.

## Constructor

### Signature

- `TrainableAssistedPlayerB(game_layout: Any, model_b: LinTrainableAssistedModelB) -> TrainableAssistedPlayerB`

### Arguments

- `game_layout`: `Any`, scalar, provides `field_size: int` and `comms_size: int`.
- `model_b`: `LinTrainableAssistedModelB`, scalar.

### Returns

- `TrainableAssistedPlayerB`, scalar.

### Preconditions

- `field_size` and `comms_size` exist on `game_layout`.
- `model_b` is callable with the expected inputs.

### Postconditions

- `self.parent` is initialised to `None` (to be set by `TrainableAssistedPlayers.players()`).
- `self.last_logprob_shoot` is initialised to `None`.
- `self.explore` is initialised to `False`.

### Errors

- Propagates exceptions from model calls or invalid `game_layout` attributes.

## Public Methods

### decide

#### Signature

- `decide(gun: np.ndarray, comm: np.ndarray, supp: Any | None = None, explore: bool | None = None) -> int`

#### Arguments

- `gun`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)`.
- `comm`: `np.ndarray`, dtype `int` {0,1} or `float`, shape `(m,)`, values in `{0,1}` or `[0.0, 1.0]`.
- `supp`: `Any` or `None`, scalar. Ignored.
- `explore`: `bool` or `None`, scalar. Optional override of `self.explore`.

#### Returns

- `shoot`: `int` {0,1}, scalar.

#### Preconditions

- `gun.shape == (n2,)` and values in `{0,1}`.
- `comm.shape == (m,)`.
- `self.parent` is set and `self.parent.previous` is not `None`.
- `self.parent.previous` contains two list-like collections, each of length `>= 1`.

#### Postconditions

- Computes shoot decision from `model_b`.
- Sets `self.last_logprob_shoot`: `float`, scalar.

#### Errors

- Raises `ValueError` if `gun` shape is invalid.
- Raises `ValueError` if `comm` shape is invalid.
- Raises `RuntimeError` if `parent.previous` is missing (Player A must run first).
- Raises `TypeError`/`ValueError` if `parent.previous` is not in the expected structure.
- Propagates exceptions from TensorFlow/model execution.

!!! example "Minimal usage"
    ```python
    import numpy as np
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.trainable_assisted_players import TrainableAssistedPlayers

    layout = GameLayout(field_size=4, comms_size=1)
    factory = TrainableAssistedPlayers(layout)
    player_a, player_b = factory.players()

    field = np.zeros(layout.field_size**2, dtype=int)
    comm = player_a.decide(field)

    gun = np.zeros(layout.field_size**2, dtype=int)
    gun[0] = 1
    shoot = player_b.decide(gun, comm)
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

- Sets `self.last_logprob_shoot = None`.

#### Errors

- None.

## Data & State

- `game_layout`: `Any`, scalar.
- `model_b`: `LinTrainableAssistedModelB`, scalar.
- `parent`: `TrainableAssistedPlayers | None`, scalar.
- `last_logprob_shoot`: `float | None`, scalar.
- `explore`: `bool`, scalar.

## Planned (design-spec)

- None identified from the provided implementation.

## Deviations

- The factory docstring suggests a fixed tensor shape `(B, n2)` for `previous`, but this player accepts and normalises
  various list/non-list forms, implying model-defined shapes.

## Notes for Contributors

- Keep `previous` validation strict enough to catch programming errors, but avoid over-constraining shapes unless the
  architecture guarantees them.
- If `comm` is allowed to be relaxed (e.g. DRU), ensure log-prob semantics remain well-defined.

## Changelog

- 2026-01-11 — Author: Rob Hendriks
