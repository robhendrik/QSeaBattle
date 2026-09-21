# NeuralNetPlayerA

> Role: Player A implementation that uses a Keras model to map a binary game field to a communication bit-vector, optionally sampling actions and tracking the last action log-probability.

Location: `Q_Sea_Battle.neural_net_player_a.NeuralNetPlayerA`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | `GameLayout`, constraints: not specified, shape: N/A | Game layout describing field size and communication dimensions. |
| model_a | `tf.keras.Model`, constraints: callable like `model_a(x, training=False)` and returns logits compatible with output shape `(batch, m)`, shape: N/A | Keras model mapping a batch of scaled field vectors (shape `(batch, n2)`) to per-bit logits (shape `(batch, m)`). |
| explore | `bool`, constraints: `True` enables Bernoulli sampling and `False` enables greedy thresholding, shape: scalar | If `True`, sample communication bits; if `False`, act greedily by thresholding probabilities. |

Preconditions

- `field_size`, `comms_size`, `n2`, and `m` are not defined in this module; the constructor assumes `game_layout` provides whatever `PlayerA` requires to make decisions of length `m` over fields of length `n2` (constraints not specified here).
- `model_a` must accept a `np.ndarray, dtype float32, shape (1, n2)` (after scaling) and return per-bit logits such that `.numpy()[0]` is array-like of shape `(m,)`.

Postconditions

- `self.model_a` is set to `model_a`.
- `self.explore` is set to `explore`.
- `self.last_logprob` is set to `None`.

Errors

- Not specified.

!!! example "Example"
    ```python
    import numpy as np
    import tensorflow as tf
    
    # game_layout must be provided by the package; details not shown here.
    player = NeuralNetPlayerA(game_layout=game_layout, model_a=tf.keras.Model(), explore=False)
    field = np.zeros((n2,), dtype=np.int32)
    msg = player.decide(field)
    lp = player.get_log_prob()
    player.reset()
    ```

## Public Methods

### decide

`decide(field: np.ndarray, supp: Any | None = None) -> np.ndarray`

Compute a communication vector for the given field.

Parameters

- `field`: `np.ndarray, dtype float32 (after internal conversion), constraints: values intended in {0,1}, shape (n2,)`.
- `supp`: `Any | None`, constraints: currently unused, shape: N/A.

Returns

- `actions`: `np.ndarray, dtype int {0,1}, shape (m,)`.

Errors

- Not specified.

Side effects

- Sets `self.last_logprob` to `float`, equal to the sum of per-bit log-probabilities for the returned action under the model logits.

Notes

- The input field is internally reshaped to `np.ndarray, dtype float32, shape (1, n2)` and scaled by subtracting `0.5`, mapping `0 -> -0.5` and `1 -> +0.5`.
- If `self.explore` is `True`, each bit is sampled independently via `rnd < probs`; otherwise actions are computed via `probs >= 0.5`.

### logit_to_probs

`logit_to_probs(logits: np.ndarray | float) -> np.ndarray | float`

Convert logits to probabilities (wrapper around `logit_to_prob`).

Parameters

- `logits`: `np.ndarray | float`, constraints: not specified, shape: scalar or any shape.

Returns

- `probs`: `np.ndarray | float`, constraints: not specified, shape: same as `logits`.

### logit_to_log_probs

`logit_to_log_probs(logits: np.ndarray | float, actions: np.ndarray | float) -> np.ndarray | float`

Compute per-bit log-probabilities for given actions under logits (wrapper around `logit_to_logprob`).

Parameters

- `logits`: `np.ndarray | float`, constraints: not specified, shape: scalar or any shape.
- `actions`: `np.ndarray | float`, constraints: intended values in `{0,1}`, shape: scalar or any shape broadcast-compatible with `logits`.

Returns

- `log_probs`: `np.ndarray | float`, constraints: not specified, shape: broadcast of `logits` and `actions`.

### get_log_prob

`get_log_prob() -> float`

Return the log-probability of the most recent decided action.

Returns

- `log_prob`: `float`, constraints: finite-ness not specified, shape: scalar.

Errors

- Raises `RuntimeError` if `self.last_logprob` is `None` (i.e., `decide()` has not been called since the last `reset()`).

### reset

`reset() -> None`

Reset internal state by clearing any stored log-probability.

Returns

- `None`: `NoneType`, shape: N/A.

## Data & State

- `model_a`: `tf.keras.Model`, constraints: callable and returns logits compatible with message length `m`, shape: N/A.
- `explore`: `bool`, constraints: if `True` sample actions; if `False` greedily threshold at `0.5`, shape: scalar.
- `last_logprob`: `Optional[float]`, constraints: `None` before any decision or after `reset()`, otherwise equals the summed per-bit log-probability of the most recent action, shape: scalar.

## Planned (design-spec)

- Not specified.

## Deviations

- No design notes were provided; deviations not specified.

## Notes for Contributors

- `n2` (flattened field size) and `m` (number of communication bits) are implied by the docstrings and model I/O shapes but are not defined in this module; keep documentation and tests aligned with the definitions in `GameLayout` and `PlayerA`.
- The helper function `_scale_field` is module-private and is used to shift binary inputs from `{0,1}` to `{-0.5,+0.5}` before model inference.

## Related

- `Q_Sea_Battle.neural_net_player_a._scale_field` (module-private helper used by `decide`).
- `Q_Sea_Battle.logit_utilities.logit_to_prob` (used by `logit_to_probs`).
- `Q_Sea_Battle.logit_utilities.logit_to_logprob` (used by `logit_to_log_probs`).
- `Q_Sea_Battle.players_base.PlayerA` (base class).

## Changelog

- Not specified.