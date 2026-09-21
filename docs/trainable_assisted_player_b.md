# TrainableAssistedPlayerB

> Role: Gameplay-facing Player B policy wrapper that turns a trainable TensorFlow model into a discrete shoot decision (0/1) and tracks the action log-probability for training.

Location: `Q_Sea_Battle.trainable_assisted_player_b.TrainableAssistedPlayerB`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | Any, constraints: must provide attributes `field_size` and `comms_size` convertible via `int(...)`, shape: N/A | Gameplay layout object used to derive $n2 = field\_size^2$ and $m = comms\_size$. |
| model_b | LinTrainableAssistedModelB, constraints: callable as `model_b([...])`; may also be an instance of `GameplayModelBAdapter`, shape: N/A | Underlying model for Player B (adapter path returns a bit; legacy path returns a logit). |

Preconditions: `game_layout.field_size` and `game_layout.comms_size` exist and are convertible to `int`.

Postconditions: `self.game_layout` and `self.model_b` are stored; `self.parent` is set to `None`; `self.last_logprob_shoot` is set to `None`; `self.explore` is set to `False`.

Errors: Not specified.

!!! example "Example"
    ```python
    from Q_Sea_Battle.trainable_assisted_player_b import TrainableAssistedPlayerB
    
    player_b = TrainableAssistedPlayerB(game_layout=layout, model_b=model_b)
    player_b.parent = parent  # parent.previous must be set by Player A before decide()
    ```

## Public Methods

### decide(gun, comm, supp=None, explore=None)

Decide whether to shoot (0/1) from gun bits, comm bits, and tensors stored on `parent.previous`.

| Parameter | Type | Description |
| --- | --- | --- |
| gun | np.ndarray, dtype int or float, constraints: values must be exactly in {0,1}, shape (n2,) | Local gun measurement bits, where $n2 = field\_size^2$. |
| comm | np.ndarray, dtype int or float, constraints: no value constraint enforced; shape (m,) | Received communication bits, where $m = comms\_size$. |
| supp | Any or None, constraints: unused, shape: N/A | Accepted for API compatibility; ignored. |
| explore | bool or None, constraints: if provided overrides `self.explore`, shape: N/A | If True, sample stochastically; if False, act greedily; if None, uses `self.explore`. |

Returns: int, constraints: in {0,1}, shape: scalar Python `int`; the shoot decision.

Preconditions: `self.parent` is not `None` and `self.parent.previous` is not `None`; `gun.shape == (n2,)`; `comm.shape == (m,)`; `gun` contains only 0/1 values; `self.parent.previous` is a 2-tuple `(prev_meas_list, prev_out_list)` where each element is a list/tuple or a single tensor/array convertible into a list of length >= 1.

Postconditions: Returns a discrete shoot bit in {0,1}; updates `self.last_logprob_shoot` to the log-probability (Python `float`) of the action actually taken under the Bernoulli distribution parameterized by the model logit(s); may print runtime warnings if `parent.previous` tensors/arrays appear non-binary within tolerance.

Errors:
- ValueError: if `gun` has wrong shape; if `comm` has wrong shape; if `gun` contains values outside {0,1}; if `parent.previous` lists have length < 1; if adapter path returns a `shoot_bit` not in {0,1}.
- RuntimeError: if `parent.previous` is missing (Player A must act first).
- TypeError: if elements of `prev_meas_list`/`prev_out_list` do not have a `.shape` (i.e., are not tensors/arrays).

!!! example "Example"
    ```python
    import numpy as np
    
    # Assume:
    # - layout.field_size and layout.comms_size are set
    # - player_a has already populated parent.previous
    gun = np.zeros((layout.field_size ** 2,), dtype=np.int32)
    comm = np.zeros((layout.comms_size,), dtype=np.int32)
    
    player_b.parent = parent
    shoot = player_b.decide(gun=gun, comm=comm, explore=True)
    ```

### get_log_prob()

Return the log-probability of the most recent shoot decision.

Returns: float, constraints: finite float not guaranteed/validated, shape: scalar Python `float`; the most recently stored log-probability.

Preconditions: `decide()` has been called since the last `reset()` such that `self.last_logprob_shoot` is not `None`.

Postconditions: Does not modify state.

Errors:
- RuntimeError: if no log-probability is available (i.e., `self.last_logprob_shoot is None`).

!!! example "Example"
    ```python
    shoot = player_b.decide(gun=gun, comm=comm)
    logp = player_b.get_log_prob()
    ```

### reset()

Reset per-episode/per-turn cached state.

Returns: None, constraints: N/A, shape: N/A.

Preconditions: None.

Postconditions: Sets `self.last_logprob_shoot` to `None`.

Errors: Not specified.

!!! example "Example"
    ```python
    player_b.reset()
    ```

## Data & State

- `game_layout`: Any, constraints: must expose `field_size` and `comms_size` used by `decide()`, shape: N/A.
- `model_b`: LinTrainableAssistedModelB, constraints: callable; may be a `GameplayModelBAdapter` instance for adapter path behavior, shape: N/A.
- `parent`: Any or None, constraints: when non-None must provide attribute `previous`; `previous` must be a 2-tuple `(prev_meas_list, prev_out_list)`, shape: N/A.
- `last_logprob_shoot`: float or None, constraints: set by `decide()` and cleared by `reset()`, shape: scalar Python `float`.
- `explore`: bool, constraints: default False; used as default exploration flag in `decide()` when `explore` argument is None, shape: scalar.

## Planned (design-spec)

Not specified.

## Deviations

Not specified.

## Notes for Contributors

- `decide()` supports two execution paths: adapter path when `model_b` is a `GameplayModelBAdapter`, and legacy path otherwise; keep both paths consistent in return types and the `(B, D)` rank-2 expectations enforced by `_ensure_rank2(...)`.
- `parent.previous` is treated as shared state populated by Player A; changing its structure requires coordinated changes across the gameplay pipeline.
- The module may emit print-based warnings for non-binary values in previous tensors; this is intended as a non-breaking gameplay safety check and must not mutate data.

## Related

- `Q_Sea_Battle.trainable_assisted_player_b.bernoulli_log_prob_from_logits` (imported if available; otherwise fallback defined in-module)
- `Q_Sea_Battle.trainable_assisted_player_b.GameplayModelBAdapter`
- `Q_Sea_Battle.trainable_assisted_player_b.LinTrainableAssistedModelB`

## Changelog

- Not specified.