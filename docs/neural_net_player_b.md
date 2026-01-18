# NeuralNetPlayerB

> Role: Player B implementation driven by a Keras model that maps a compact state representation (normalised gun index + comm bits) to a shoot decision and stores the last action log-probability.
Location: `Q_Sea_Battle.neural_net_player_b.NeuralNetPlayerB`

## Constructor

| Parameter | Type | Description |
|---|---|---|
| game_layout | GameLayout, constraints Not specified, shape Not applicable | Shared `GameLayout` describing the environment; passed to `PlayerB` base class. |
| model_b | tf.keras.Model, constraints callable as `model_b(x, training=False)` and returns a single logit per sample, shape Not specified | Keras model mapping input vectors of shape $(1 + m,)$ (normalised gun index + comm bits) to a single shoot logit. |
| explore | bool, constraints {True, False}, shape scalar | If `True`, sample shoot actions from the Bernoulli distribution defined by the predicted probability; if `False`, act greedily using a `0.5` probability threshold. |

Preconditions: `game_layout` is a valid `GameLayout` instance; `model_b` accepts NumPy inputs compatible with the concatenated feature vector `x` and returns a scalar logit for the single sample.  
Postconditions: `self.model_b` is set; `self.explore` is set; `self.last_logprob` is initialised to `None`.  
Errors: Not specified.  
Example:

```python
import numpy as np
import tensorflow as tf

player_b = NeuralNetPlayerB(game_layout=game_layout, model_b=model_b, explore=True)
gun = np.zeros((n2,), dtype=np.float32)
gun[0] = 1.0
comm = np.zeros((m,), dtype=np.float32)
action = player_b.decide(gun=gun, comm=comm)
logp = player_b.get_log_prob()
player_b.reset()
```

## Public Methods

### decide(gun, comm, supp=None)

Decide whether to shoot based on the gun vector and communication bits.

Parameter | Type | Description
---|---|---
gun | np.ndarray, dtype float32 (after conversion), constraints flattened one-hot intended but argmax fallback if not strictly one-hot, shape (n2,) | Flattened gun vector of length `n2`; internally converted to a normalised scalar index in $[0, 1]$.
comm | np.ndarray, dtype float32 (after conversion), constraints Not specified, shape (m,) | Communication vector from Player A of length `m`.
supp | Any \| None, constraints unused, shape Not applicable | Optional supporting information; not used by this implementation.

Returns: int, constraints {0,1}, shape scalar; `1` to shoot, `0` to not shoot.  

Preconditions: `gun` and `comm` are array-like and can be reshaped to `(1, -1)`; `model_b(x, training=False)` is valid for `x` of shape `(1, 1 + m)` and yields a value convertible to a scalar logit.  
Postconditions: `self.last_logprob` is updated to the log-probability of the chosen action under the model’s Bernoulli distribution; returns the chosen action as `int`.  
Errors: Not specified.

### logit_to_probs(logits)

Backward-compatible wrapper around `logit_to_prob`.

Parameter | Type | Description
---|---|---
logits | np.ndarray \| float, constraints Not specified, shape scalar or broadcastable array | Logit(s) to convert to probability/probabilities.

Returns: np.ndarray \| float, constraints range $[0, 1]$, shape matches input; the probability/probabilities derived from `logits`.  

Preconditions: Not specified.  
Postconditions: Not specified.  
Errors: Not specified.

### logit_to_log_probs(logits, actions)

Backward-compatible wrapper around `logit_to_logprob`.

Parameter | Type | Description
---|---|---
logits | np.ndarray \| float, constraints Not specified, shape scalar or broadcastable array | Logit(s) used to compute log-probabilities.
actions | np.ndarray \| float, constraints Not specified, shape scalar or broadcastable array | Action(s) associated with the log-probability computation.

Returns: np.ndarray \| float, constraints Not specified, shape broadcasted from inputs; log-probability/log-probabilities for `actions` under the Bernoulli distribution parameterised by `logits`.  

Preconditions: Not specified.  
Postconditions: Not specified.  
Errors: Not specified.

### get_log_prob()

Return the log-probability of the last chosen action.

Returns: float, constraints finite scalar expected, shape scalar; log-probability of the last action.  

Preconditions: `decide()` has been called since the last `reset()` such that `self.last_logprob` is not `None`.  
Postconditions: Does not modify state.  
Errors: Raises `RuntimeError` if `self.last_logprob` is `None`.

### reset()

Reset internal state (stored log-probability).

Returns: None, constraints Not applicable, shape Not applicable.  

Preconditions: Not specified.  
Postconditions: `self.last_logprob` is set to `None`.  
Errors: Not specified.

## Data & State

- `model_b`: tf.keras.Model, constraints callable as `model_b(x, training=False)`, shape Not specified; the Keras model used to produce a shoot logit from the concatenated feature vector.
- `explore`: bool, constraints {True, False}, shape scalar; controls stochastic sampling (`True`) vs greedy thresholding (`False`).
- `last_logprob`: float \| None, constraints `None` before first decision or after reset, shape scalar; log-probability of the most recent action chosen by `decide()`.

## Planned (design-spec)

Not specified.

## Deviations

Not specified.

## Notes for Contributors

- The public `gun` interface is a flattened vector of length `n2`, but internally it is compressed via argmax into a single normalised index in $[0, 1]$ using $idx / \max(1, n2 - 1)$; non-strict one-hot inputs are handled by argmax fallback.
- `decide()` stores the log-probability of the selected action in `last_logprob`; callers relying on `get_log_prob()` must call `decide()` first and must handle the `RuntimeError` case after `reset()`.

## Related

- `Q_Sea_Battle.players_base.PlayerB`
- `Q_Sea_Battle.game_layout.GameLayout`
- `Q_Sea_Battle.logit_utilities.logit_to_prob`
- `Q_Sea_Battle.logit_utilities.logit_to_logprob`

## Changelog

- 0.1: Initial implementation of `NeuralNetPlayerB` with Keras-based decision-making and stored last-action log-probability.