# NeuralNetPlayerB

> Role: Player B policy backed by a Keras model that outputs a shoot logit given a compressed gun position and communication bits.

Location: `Q_Sea_Battle.neural_net_player_b.NeuralNetPlayerB`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | `GameLayout`, constraints: instance of `Q_Sea_Battle.game_layout.GameLayout`, shape: N/A | Shared environment layout passed to the base `PlayerB` constructor. |
| model_b | `tf.keras.Model`, constraints: callable like `model_b(x, training=False)` and returns a single logit per row, shape: input `np.ndarray, dtype float32, shape (B, 1 + m)` and output compatible with `np.ndarray, dtype float32/float64, shape (B, 1)` or `(B,)` | Keras model mapping features (normalized gun index + communication bits) to a shoot logit. |
| explore | `bool`, constraints: {True, False}, shape: scalar | If `True`, sample actions from Bernoulli($p$); if `False`, act greedily with threshold $p \ge 0.5$. |

Preconditions

- `game_layout` is a valid `GameLayout` instance accepted by `PlayerB.__init__(game_layout=...)`.
- `model_b` accepts NumPy input `x` formed by concatenating the normalized gun index and `comm` along axis 1.

Postconditions

- `self.model_b` is set to `model_b`.
- `self.explore` is set to `explore`.
- `self.last_logprob` is set to `None`.

Errors

- Not specified.

Example

```python
import numpy as np
import tensorflow as tf

from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.neural_net_player_b import NeuralNetPlayerB

game_layout = GameLayout(...)  # as defined by your project
model_b = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1 + 3,)),  # m=3 example
    tf.keras.layers.Dense(1),
])

player_b = NeuralNetPlayerB(game_layout=game_layout, model_b=model_b, explore=False)

n2 = 9
m = 3
gun = np.eye(n2, dtype=np.float32)[4]        # one-hot position
comm = np.array([1, 0, 1], dtype=np.float32) # length m
action = player_b.decide(gun=gun, comm=comm)
logp = player_b.get_log_prob()
```

## Public Methods

### decide(gun, comm, supp)

Decide whether Player B shoots by compressing the gun one-hot vector to a normalized scalar index, concatenating it with the communication vector, and forwarding through `model_b` to obtain a shoot logit.

Parameters

- `gun`: `np.ndarray`, constraints: convertible to `float32` and reshaped to `(1, n2)`, shape: `(n2,)` or `(1, n2)` or any array with first dimension batch-like such that `reshape(1, -1)` is valid for single decision use.
- `comm`: `np.ndarray`, constraints: convertible to `float32` and reshaped to `(1, m)`, shape: `(m,)` or `(1, m)` or any array with first dimension batch-like such that `reshape(1, -1)` is valid for single decision use.
- `supp`: `Any | None`, constraints: unused, shape: N/A.

Returns

- `int`, constraints: {0, 1}, shape: scalar; returns `1` if shooting is selected, otherwise `0`.

Errors

- Any exception raised by `np.asarray`, reshaping, `np.concatenate`, `self.model_b(...).numpy()`, or downstream conversion may propagate; additional error behavior is not specified.

### logit_to_probs(logits)

Backward-compatible wrapper around `Q_Sea_Battle.logit_utilities.logit_to_prob` converting Bernoulli logit(s) to probability/probabilities.

Parameters

- `logits`: `np.ndarray | float`, constraints: numeric logit(s), shape: scalar or arbitrary NumPy array shape.

Returns

- `np.ndarray | float`, constraints: probability value(s) corresponding to `logits`, shape: same structure as input.

### logit_to_log_probs(logits, actions)

Backward-compatible wrapper around `Q_Sea_Battle.logit_utilities.logit_to_logprob` computing the log-probability of Bernoulli action(s) under given logit(s).

Parameters

- `logits`: `np.ndarray | float`, constraints: numeric logit(s), shape: scalar or arbitrary NumPy array shape.
- `actions`: `np.ndarray | float`, constraints: action(s) encoded as 0/1 (or float equivalents), shape: compatible with `logits` for the wrapped utility function.

Returns

- `np.ndarray | float`, constraints: log-probability value(s), shape: structure compatible with inputs.

### get_log_prob()

Return the stored log-probability of the most recent action selected by `decide`.

Parameters

- None.

Returns

- `float`, constraints: finite scalar float if available, shape: scalar.

Errors

- Raises `RuntimeError` if `self.last_logprob` is `None` (no decision taken since last reset).

### reset()

Reset internal episode state by clearing any stored log-probability.

Parameters

- None.

Returns

- `None`, constraints: N/A, shape: N/A.

Errors

- Not specified.

## Data & State

- `model_b`: `tf.keras.Model`, constraints: callable as used by `decide`, shape: N/A.
- `explore`: `bool`, constraints: {True, False}, shape: scalar.
- `last_logprob`: `Optional[float]`, constraints: `None` or scalar float log-probability, shape: scalar.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- `decide` expects the public gun representation to be a flattened one-hot vector of length `n2`, but it uses `argmax` as a stable fallback even when the vector is not strictly one-hot (e.g., all zeros), which can hide upstream encoding errors.
- The feature vector passed to `model_b` is `x = concat([gun_idx_norm, comm], axis=1)` with `gun_idx_norm` in $[0, 1]$ and shape `(1, 1)`; `comm` must therefore be compatible with shape `(1, m)`.

## Related

- `Q_Sea_Battle.players_base.PlayerB`
- `Q_Sea_Battle.game_layout.GameLayout`
- `Q_Sea_Battle.logit_utilities.logit_to_prob`
- `Q_Sea_Battle.logit_utilities.logit_to_logprob`

## Changelog

- Not specified.