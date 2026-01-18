# TrainableAssistedPlayerA

> Role: Player A wrapper that computes communication bits from a binary field using a trainable model, optionally sampling for exploration, and stores intermediate tensors on its parent for Player B.

Location: `Q_Sea_Battle.trainable_assisted_player_a.TrainableAssistedPlayerA`

## Derived constraints

- Let field_size be `int(game_layout.field_size)`, comms_size be `int(game_layout.comms_size)`, $n2 = field\_size^2$, and $m = comms\_size$.
- decide() requires field to be `np.ndarray, dtype int {0,1}, shape (n2,)` and returns `np.ndarray, dtype int {0,1}, shape (m,)`.

## Constructor

Parameter | Type | Description
--- | --- | ---
game_layout | Any, constraints: must provide attributes `field_size` and `comms_size` readable via `getattr`, shape: N/A | Game layout object used to derive $n2$ and $m$.
model_a | LinTrainableAssistedModelA, constraints: must provide `compute_with_internal(field_batch)` returning `(comm_logits, meas_list, out_list)`, shape: N/A | Trainable model used to compute communication logits and internal tensors.

Preconditions

- game_layout must have attributes `field_size` and `comms_size` convertible to int.

Postconditions

- self.game_layout is set to game_layout.
- self.model_a is set to model_a.
- self.parent is set to None.
- self.last_logprob_comm is set to None.
- self.explore is set to False.

Errors

- Not specified (constructor does not explicitly raise; attribute access failures may surface later in decide()).

Example

!!! example "Constructing a TrainableAssistedPlayerA"
    ```python
    from Q_Sea_Battle.trainable_assisted_player_a import TrainableAssistedPlayerA
    from Q_Sea_Battle.lin_trainable_assisted_model_a import LinTrainableAssistedModelA

    game_layout = ...  # must define field_size and comms_size
    model_a = LinTrainableAssistedModelA(...)
    player_a = TrainableAssistedPlayerA(game_layout=game_layout, model_a=model_a)
    ```

## Public Methods

### decide(field, supp=None, explore=None)

Decide communication bits based on the field, either greedily (threshold at 0.5) or by sampling independent Bernoulli bits, and record the log-probability of the chosen bits under the model logits.

Parameter | Type | Description
--- | --- | ---
field | np.ndarray, dtype int {0,1}, shape (n2,) | Flattened binary field input; must have shape $(n2,)$ and contain only 0/1.
supp | Any \| None, constraints: ignored, shape: N/A | Ignored support argument.
explore | bool \| None, constraints: if not None overrides `self.explore`, shape: N/A | Optional override controlling exploration (sampling) vs greedy selection.

Returns

- np.ndarray, dtype int {0,1}, shape (m,) communication bits.

Preconditions

- field must satisfy `field.shape == (n2,)` where $n2 = field\_size^2$.
- field must contain only values in {0,1}.
- model_a.compute_with_internal must accept `tf.Tensor, dtype float32, shape (1, n2)` and return `comm_logits` compatible with shape `(1, m)`.

Postconditions

- self.last_logprob_comm is set to the scalar log-probability (Python float) of the returned bits under independent Bernoulli with the computed logits.
- If self.parent is not None, then `self.parent.previous` is set to `(meas_list, out_list)` as returned by the model.

Errors

- ValueError if field shape is not `(n2,)`.
- ValueError if field contains values other than 0/1.

Example

!!! example "Deciding communication bits"
    ```python
    import numpy as np

    field_size = int(getattr(player_a.game_layout, "field_size"))
    n2 = field_size ** 2
    field = np.zeros((n2,), dtype=np.int32)

    comm_bits = player_a.decide(field, explore=True)
    ```

### get_log_prob()

Return the log-probability of the last taken communication action.

Parameters

- None.

Returns

- float, constraints: finite scalar expected, shape: scalar.

Preconditions

- decide() must have been called since the last reset() such that self.last_logprob_comm is not None.

Postconditions

- No state change.

Errors

- RuntimeError if self.last_logprob_comm is None (e.g., decide() has not been called since reset()).

Example

!!! example "Reading last log-probability"
    ```python
    lp = player_a.get_log_prob()
    ```

### get_prev()

Return the parent previous tensors if available.

Parameters

- None.

Returns

- Any | None, constraints: if not None then a 2-tuple `(meas_list, out_list)` as stored on `self.parent.previous`, shape: N/A.

Preconditions

- None.

Postconditions

- No state change.

Errors

- Not specified (method is non-blocking and returns None when unavailable).

Example

!!! example "Accessing previous tensors"
    ```python
    prev = player_a.get_prev()
    if prev is not None:
        meas_list, out_list = prev
    ```

### reset()

Reset internal state.

Parameters

- None.

Returns

- None.

Preconditions

- None.

Postconditions

- self.last_logprob_comm is set to None.

Errors

- Not specified.

Example

!!! example "Resetting"
    ```python
    player_a.reset()
    ```

## Data & State

- game_layout: Any, constraints: should provide `field_size` and `comms_size`, shape: N/A; set at construction.
- model_a: LinTrainableAssistedModelA, constraints: must implement `compute_with_internal`, shape: N/A; set at construction.
- parent: Any | None, constraints: if not None may be expected to have attribute `previous`, shape: N/A; default None and intended to be set externally.
- last_logprob_comm: float | None, constraints: None before decide() or after reset(), otherwise scalar float, shape: scalar.
- explore: bool, constraints: when True decide() samples, when False decide() is greedy, shape: scalar.

## Planned (design-spec)

- Not specified (no design notes provided).

## Deviations

- Not specified (no design notes provided to compare against).

## Notes for Contributors

- decide() computes $n2$ and $m$ dynamically from game_layout on each call; changes to game_layout at runtime will affect validation and output dimensionality.
- get_prev() reads `self.parent.previous` via getattr and returns None if unavailable; callers should handle None.

## Related

- LinTrainableAssistedModelA
- PlayerA (base class, imported from `.players` if available; otherwise a fallback stub exists in-module)
- bernoulli_log_prob_from_logits (imported from `.logit_utils` if available; otherwise a fallback implementation exists in-module)

## Changelog

- 0.1: Initial implementation per module docstring.