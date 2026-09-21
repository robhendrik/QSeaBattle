# TrainableAssistedPlayerA

> Role: Wrap a trainable/adapter Model A to produce boundary communication bits for gameplay, compute their log-probability, and store intermediate tensors for Player B.

Location: `Q_Sea_Battle.trainable_assisted_player_a.TrainableAssistedPlayerA`

## Derived constraints

- Define symbols used below: field_size is `game_layout.field_size` (int, not validated), comms_size is `game_layout.comms_size` (int, not validated), $n2 = \text{field\_size}^2$, and $m = \text{comms\_size}$.
- `decide()` requires `field` to be a flat binary vector with shape $(n2,)$ and values in {0,1}.
- `decide()` returns a flat binary vector with shape $(m,)$ and dtype `int32`.
- `get_log_prob()` is only valid after at least one successful `decide()` since the last `reset()`.

## Constructor

Parameter | Type | Description
--- | --- | ---
game_layout | Any, constraints: must expose attributes `field_size` and `comms_size` convertible to `int`, shape: N/A | Game layout descriptor used to derive $n2$ and $m$ in `decide()`.
model_a | `LinTrainableAssistedModelA`, constraints: may also be an instance of `GameplayModelAAdapter` at runtime, shape: N/A | Underlying model/adapter used to compute communication outputs and intermediates.

Preconditions

- `game_layout.field_size` and `game_layout.comms_size` must exist and be convertible to `int` for `decide()` to function (not validated in `__init__`).
- `model_a` must support either the adapter call interface (if it is a `GameplayModelAAdapter`) or `compute_with_internal(field_batch)` (legacy path); this is not validated in `__init__`.

Postconditions

- `self.game_layout` is set to `game_layout`.
- `self.model_a` is set to `model_a`.
- `self.parent` is set to `None`.
- `self.last_logprob_comm` is set to `None`.
- `self.explore` is set to `False`.

Errors

- Not specified (constructor performs no explicit validation and raises no explicit exceptions).

Example

```python
from Q_Sea_Battle.trainable_assisted_player_a import TrainableAssistedPlayerA
from Q_Sea_Battle.lin_trainable_assisted_model_a import LinTrainableAssistedModelA

# game_layout must have .field_size and .comms_size attributes.
player_a = TrainableAssistedPlayerA(game_layout=game_layout, model_a=model_a)  # model_a: LinTrainableAssistedModelA
```

## Public Methods

### decide(field, supp=None, explore=None)

Decide boundary communication bits based on the current field, optionally storing intermediates on `parent.previous` and caching the action log-probability for later retrieval.

Parameter | Type | Description
--- | --- | ---
field | `np.ndarray`, dtype: any numeric (validated by value), constraints: values in {0,1}, shape $(n2,)$ | Flat binary field observation where $n2 = \text{field\_size}^2$.
supp | Any \| None, constraints: unused, shape: N/A | Present for interface compatibility; ignored.
explore | bool \| None, constraints: if not `None` overrides `self.explore`, shape: N/A | If `True`, use stochastic sampling; if `False`, use greedy thresholding; `None` uses `self.explore`.

Returns

- `np.ndarray`, dtype `int32`, constraints: values in {0,1}, shape $(m,)$: Boundary communication bits, where $m = \text{comms\_size}$.

Preconditions

- `field` must have exact shape $(n2,)$ where $n2 = \text{int}(game\_layout.field\_size)^2$.
- `field` must contain only 0/1 values.
- `game_layout` must expose `field_size` and `comms_size`.
- Adapter path: if `self.model_a` is a `GameplayModelAAdapter`, it must be callable as `self.model_a(field_batch, explore=..., return_comm_logits=True)` and return `(comm_bits_tf, meas_list, out_list, comm_logits)`.
- Legacy path: otherwise, `self.model_a` must implement `compute_with_internal(field_batch)` and return `(comm_logits, meas_list, out_list)` where `comm_logits` is compatible with shape $(1, m)$.

Postconditions

- Sets `self.last_logprob_comm` to `float` log-probability of the chosen boundary bits under independent Bernoulli bits parameterized by logits (sum across the last dimension), for the most recent call.
- If `self.parent is not None`, sets `self.parent.previous = (meas_list, out_list)` (the values returned by the underlying adapter/model).
- Returns boundary bits as a flat array of shape $(m,)$.

Errors

- `ValueError`: if `field.shape != (n2,)`.
- `ValueError`: if `field` contains values other than 0 or 1.
- `ValueError`: legacy path only, if `comm_logits` has statically known width not equal to $m$.
- Other exceptions may be raised by TensorFlow/NumPy operations or by the underlying `model_a` (not specified).

Example

```python
import numpy as np

# field must be flat binary shape (n2,)
field = np.zeros((game_layout.field_size * game_layout.field_size,), dtype=np.int32)

player_a.explore = False
comm_bits = player_a.decide(field)  # np.ndarray int32 shape (m,)

logp = player_a.get_log_prob()  # float
prev = player_a.get_prev()      # typically (meas_list, out_list) or None
```

### get_log_prob()

Return the cached log-probability of the last communication decision produced by `decide()`.

Returns

- `float`, constraints: finite real number (not validated), shape: scalar: Log-probability of the last boundary communication bits.

Preconditions

- `decide()` must have been called successfully since the last `reset()`.

Postconditions

- Does not modify object state.

Errors

- `RuntimeError`: if `self.last_logprob_comm is None` (i.e., `decide()` has not been called since `reset()`).

Example

```python
comm_bits = player_a.decide(field)
logp = player_a.get_log_prob()
```

### get_prev()

Return the stored intermediate tensors intended for Player B, if available.

Returns

- `Any | None`, constraints: if not `None`, equals `parent.previous`, shape: Not specified: Typically a tuple `(meas_list, out_list)` as set by `decide()`, or `None` if unavailable.

Preconditions

- None.

Postconditions

- Does not modify object state.

Errors

- Not specified.

Example

```python
prev = player_a.get_prev()
if prev is not None:
    meas_list, out_list = prev
```

### reset()

Reset per-episode/per-rollout state.

Returns

- `None`, shape: N/A.

Preconditions

- None.

Postconditions

- Sets `self.last_logprob_comm = None`.

Errors

- Not specified.

Example

```python
player_a.reset()
```

## Data & State

- `game_layout`: Any, constraints: must expose `field_size` and `comms_size` for `decide()`, shape: N/A.
- `model_a`: `LinTrainableAssistedModelA`, constraints: may be a `GameplayModelAAdapter` instance at runtime, shape: N/A.
- `parent`: Any \| None, constraints: if not `None` should expose writable attribute `previous`, shape: N/A.
- `last_logprob_comm`: float \| None, constraints: `None` until first successful `decide()` after `reset()`, shape: scalar.
- `explore`: bool, constraints: default `False`; may be overridden per-call via `decide(..., explore=...)`, shape: scalar.

## Planned (design-spec)

- Not specified (no design notes provided).

## Deviations

- Module docstring states that the exact contract for `decide()`, `get_log_prob()`, and stored `previous` payload is defined in project design documentation; this class enforces only shape/value checks on `field` and stores `(meas_list, out_list)` as returned by the underlying model/adapter without further schema validation.

## Notes for Contributors

- Two code paths exist: adapter path (`GameplayModelAAdapter`) where boundary bits are produced by the adapter, and legacy path where this class converts logits to boundary bits via sampling (`explore=True`) or thresholding (`explore=False`).
- `_warn_if_not_binary_list()` prints gameplay safety diagnostics if intermediate lists contain non-binary values; it does not raise and does not modify inputs.
- The log-probability is computed under independent Bernoulli bits parameterized by logits and summed over the last dimension; ensure any adapter returning `comm_logits` aligns with that assumption.

## Related

- `Q_Sea_Battle.trainable_assisted_player_a.bernoulli_log_prob_from_logits`
- `Q_Sea_Battle.lin_trainable_assisted_model_a.LinTrainableAssistedModelA`
- `Q_Sea_Battle.gameplay_adapters.GameplayModelAAdapter`

## Changelog

- Not specified.