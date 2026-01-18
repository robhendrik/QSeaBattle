# TrainableAssistedPlayerB

> Role: Trainable Player B wrapper that consumes Player A "previous" tensors plus local measurements to decide whether to shoot, while tracking the last action log-probability.

Location: `Q_Sea_Battle.trainable_assisted_player_b.TrainableAssistedPlayerB`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| `game_layout` | `Any`, not specified, shape N/A | Object providing `field_size` and `comms_size` attributes used to derive $n2 = field\_size^2$ and $m = comms\_size$. |
| `model_b` | `LinTrainableAssistedModelB`, not specified, shape N/A | Trainable model called as `model_b([gun_batch, comm_batch, prev_meas_batch, prev_out_batch])` to produce a shoot logit. |

Preconditions

- `game_layout` should expose `field_size` and `comms_size` attributes convertible to `int`; otherwise behavior is not specified (will likely raise at runtime).
- `model_b` must be callable and return a Tensor compatible with shape `(1, 1)` when invoked from `decide()`.

Postconditions

- `self.game_layout` is set to `game_layout`.
- `self.model_b` is set to `model_b`.
- `self.parent` is set to `None`.
- `self.last_logprob_shoot` is set to `None`.
- `self.explore` is set to `False`.

Errors

- Not specified by constructor code.

!!! example "Example"
    ```python
    from Q_Sea_Battle.trainable_assisted_player_b import TrainableAssistedPlayerB

    # game_layout must provide .field_size and .comms_size; model_b must be a LinTrainableAssistedModelB
    player_b = TrainableAssistedPlayerB(game_layout=layout, model_b=model_b)
    ```

## Public Methods

### decide(gun, comm, supp=None, explore=None)

Decide whether to shoot (`0` or `1`) based on `gun` + `comm` + `parent.previous` tensors.

Parameters

- `gun`: `np.ndarray`, dtype int, values in `{0,1}`, shape `(n2,)`, where $n2 = field\_size^2$.
- `comm`: `np.ndarray`, dtype not specified, shape `(m,)`, where $m = comms\_size$; values are validated for shape only (docstring notes ints in `{0,1}` or floats in `[0,1]` for DRU).
- `supp`: `Any | None`, ignored, shape N/A.
- `explore`: `bool | None`, if not `None` overrides `self.explore`, shape N/A.

Returns

- `int`, constraints `{0,1}`, shape `()`.

Preconditions

- `self.parent` is not `None` and `self.parent.previous` is not `None`.
- `getattr(self.game_layout, "field_size")` and `getattr(self.game_layout, "comms_size")` exist and are convertible to `int`.
- `gun.shape == (n2,)` and `gun` contains only `0/1`.
- `comm.shape == (m,)`.
- `self.parent.previous` is a tuple-like `(prev_meas_list, prev_out_list)` where each is a `list` (enforced before later normalization).
- `len(prev_meas_list) >= 1` and `len(prev_out_list) >= 1`.

Postconditions

- Computes `shoot_logit = self.model_b([gun_batch, comm_batch, prev_meas_batch, prev_out_batch])` where `gun_batch` has shape `(1, n2)` and `comm_batch` has shape `(1, m)`.
- Sets `self.last_logprob_shoot` to the log-probability (Python `float`) of the returned action under the computed logits.
- Returns `shoot` as greedy (`shoot_prob >= 0.5`) if not exploring, else samples via `Uniform(0,1) < shoot_prob`.

Errors

- `ValueError`: if `gun` shape mismatches `(n2,)`.
- `ValueError`: if `gun` contains values other than `0/1`.
- `ValueError`: if `comm` shape mismatches `(m,)`.
- `RuntimeError`: if `self.parent is None` or `self.parent.previous is None`.
- `TypeError`: if `self.parent.previous` is not `(list, list)` at the initial type check.
- `ValueError`: if either list in `self.parent.previous` has length `< 1`.

!!! example "Example"
    ```python
    import numpy as np

    # Assume player_b.parent has been set and player_a has already populated parent.previous
    n2 = int(player_b.game_layout.field_size) ** 2
    m = int(player_b.game_layout.comms_size)

    gun = np.zeros((n2,), dtype=int)
    comm = np.zeros((m,), dtype=int)

    shoot = player_b.decide(gun=gun, comm=comm, explore=True)
    logp = player_b.get_log_prob()
    ```

### get_log_prob()

Return log-probability of the last taken shoot action (as set by `decide()`).

Parameters

- None.

Returns

- `float`, constraints not specified (log-probability), shape `()`.

Preconditions

- `self.last_logprob_shoot` is not `None` (i.e., `decide()` has been called since the last `reset()`).

Errors

- `RuntimeError`: if `self.last_logprob_shoot is None`.

### reset()

Reset internal state.

Parameters

- None.

Returns

- `None`, shape N/A.

Postconditions

- `self.last_logprob_shoot` is set to `None`.

Errors

- Not specified.

## Data & State

- `game_layout`: `Any`, constraints not specified, shape N/A; must provide `field_size` and `comms_size` attributes used by `decide()`.
- `model_b`: `LinTrainableAssistedModelB`, constraints not specified, shape N/A; called by `decide()` to produce a shoot logit tensor.
- `parent`: `Any | None`, constraints not specified, shape N/A; expected (by `decide()`) to provide `.previous` containing prior tensors from Player A.
- `last_logprob_shoot`: `float | None`, constraints not specified, shape `()`; updated by `decide()`, cleared by `reset()`.
- `explore`: `bool`, constraints `{False, True}`, shape `()`; default `False`, optionally overridden per-call via `decide(..., explore=...)`.

## Planned (design-spec)

- Not specified (no design notes provided).

## Deviations

- The docstring claims `parent` is of type `TrainableAssistedPlayers` and is set by `TrainableAssistedPlayers.players()`, but the module only types it as `Any | None` and does not define or enforce this contract.
- `parent.previous` is initially required to be `(list, list)` via an explicit `isinstance(..., list)` check, yet later code contains normalization for non-list/tuple values ("linear case: single tensor → list of length 1") that is unreachable if the earlier check fails; these two behaviors conflict.

## Notes for Contributors

- Symbols used: $n2 = field\_size^2$ and $m = comms\_size$ are derived inside `decide()` from `self.game_layout`.
- `decide()` expects `self.parent.previous` to be populated before it is called; ensure Player A executes first in the calling sequence.
- `bernoulli_log_prob_from_logits` may be imported from `.logit_utils` or fall back to a local implementation; changing either affects `last_logprob_shoot` semantics.

## Related

- `Q_Sea_Battle.trainable_assisted_player_b.bernoulli_log_prob_from_logits` (imported if available; otherwise locally defined fallback)
- `Q_Sea_Battle.trainable_assisted_player_b.LinTrainableAssistedModelB` (dependency)
- `Q_Sea_Battle.trainable_assisted_player_b.PlayerB` (base class, imported if available; otherwise fallback)

## Changelog

- 0.1: Initial version (per module docstring).