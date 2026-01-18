# NeuralNetPlayerA

> Role: Player A implementation that uses a Keras model to produce a communication bit-vector from a scaled binary field, optionally sampling actions for exploration and tracking the last action log-probability.

Location: `Q_Sea_Battle.neural_net_player_a.NeuralNetPlayerA`

## Derived constraints

- Define $n2$ as the flattened field size and $m$ as the communication vector size.
- `decide()` reshapes `field` to `(1, n2)` and expects `model_a` to return logits broadcastable to shape `(m,)` after indexing `[0]`.
- `decide()` returns integer bits in `{0, 1}` with shape `(m,)` and stores `last_logprob` as a Python `float` equal to the sum of per-bit log-probabilities.

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| game_layout | GameLayout, constraints: not specified, shape: N/A | Shared `GameLayout` describing the environment; passed to `PlayerA` base constructor. |
| model_a | tf.keras.Model, constraints: callable on `tf.Tensor` or array-like input, shape: input `(1, n2)` and output convertible to `np.ndarray` with first dimension size `1` and remaining shape `(m,)` | Keras model mapping the scaled flattened field to communication logits. |
| explore | bool, constraints: {True, False}, shape: scalar | If `True`, sample communication bits from Bernoulli probabilities; if `False`, act greedily by thresholding at `0.5`. |

Preconditions

- `model_a(field_scaled, training=False)` must return a tensor/array that supports `.numpy()` and yields an array with shape `(1, m)` such that indexing `[0]` produces shape `(m,)`.

Postconditions

- `self.model_a` is set to `model_a`.
- `self.explore` is set to `explore`.
- `self.last_logprob` is set to `None`.

Errors

- Not specified.

Example

```python
import numpy as np
import tensorflow as tf

from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.neural_net_player_a import NeuralNetPlayerA

game_layout = GameLayout(...)  # Not specified
model_a = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(game_layout.field_size * game_layout.field_size,)),
    tf.keras.layers.Dense(game_layout.comms_size),
])

player = NeuralNetPlayerA(game_layout=game_layout, model_a=model_a, explore=False)
field = np.zeros((game_layout.field_size * game_layout.field_size,), dtype=int)
action = player.decide(field)
logp = player.get_log_prob()
```

## Public Methods

### decide

- Signature: `decide(self, field: np.ndarray, supp: Any | None = None) -> np.ndarray`

Parameter | Type | Description
- `field` | np.ndarray, dtype float32 (after conversion), constraints: values intended in {0,1}, shape (n2,) | Flattened field array; converted via `np.asarray(..., dtype=np.float32).reshape(1, -1)` and then scaled by subtracting `0.5`.
- `supp` | Any \| None, constraints: unused, shape: N/A | Optional supporting information; not used.

Returns

- np.ndarray, dtype int, constraints: elements in {0,1}, shape (m,) | Communication action bits computed either greedily (`probs >= 0.5`) or by sampling (`rnd < probs`).

Preconditions

- `field` must be reshapeable to `(1, n2)` via `.reshape(1, -1)`.
- `self.model_a` must accept `field_scaled` and produce logits compatible with `logit_to_probs`.
- `logit_to_prob()` and `logit_to_logprob()` must accept the produced logits and actions and return broadcast-compatible outputs.

Postconditions

- `self.last_logprob` is set to `float(np.sum(log_probs_bits))`, where `log_probs_bits = self.logit_to_log_probs(logits, actions)`.

Errors

- Any exceptions raised by NumPy reshaping/conversion, the Keras model call, `.numpy()`, or the utility functions are not caught.

Example

```python
field = np.random.randint(0, 2, size=(n2,), dtype=int)
action_bits = player.decide(field)
```

### logit_to_probs

- Signature: `logit_to_probs(logits: np.ndarray | float) -> np.ndarray | float`

Parameter | Type | Description
- `logits` | np.ndarray \| float, constraints: not specified, shape: any | Logits to convert to probabilities; passed through to `Q_Sea_Battle.logit_utilities.logit_to_prob`.

Returns

- np.ndarray \| float, constraints: not specified, shape: same as input/broadcast behavior of `logit_to_prob` | Probabilities corresponding to `logits`.

Errors

- Not specified (delegated to `logit_to_prob`).

Example

```python
probs = NeuralNetPlayerA.logit_to_probs(np.array([0.0, 1.0], dtype=np.float32))
```

### logit_to_log_probs

- Signature: `logit_to_log_probs(logits: np.ndarray | float, actions: np.ndarray | float) -> np.ndarray | float`

Parameter | Type | Description
- `logits` | np.ndarray \| float, constraints: not specified, shape: any | Scalar or array of logits.
- `actions` | np.ndarray \| float, constraints: intended in {0,1}, shape: any | Scalar or array of actions; broadcast with `logits`.

Returns

- np.ndarray \| float, constraints: not specified, shape: broadcast(logits, actions) | Log-probabilities corresponding to selecting `actions` under Bernoulli probabilities induced by `logits`.

Errors

- Not specified (delegated to `logit_to_logprob`).

Example

```python
logp_bits = NeuralNetPlayerA.logit_to_log_probs(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
```

### get_log_prob

- Signature: `get_log_prob(self) -> float`

Parameters

- None.

Returns

- float, constraints: finite not specified, shape: scalar | The stored log-probability of the last action returned by `decide()`.

Errors

- `RuntimeError`: Raised if `self.last_logprob is None` (i.e., no decision since construction or last `reset()`).

Example

```python
player.decide(field)
logp = player.get_log_prob()
```

### reset

- Signature: `reset(self) -> None`

Parameters

- None.

Returns

- None.

Postconditions

- `self.last_logprob` is set to `None`.

Errors

- Not specified.

Example

```python
player.reset()
```

## Data & State

- `model_a`: tf.keras.Model, constraints: callable as used by `decide()`, shape: N/A | The Keras model used to produce logits from scaled fields.
- `explore`: bool, constraints: {True, False}, shape: scalar | Controls whether actions are sampled or thresholded.
- `last_logprob`: float \| None, constraints: None or scalar float, shape: scalar | Sum of per-bit log-probabilities of the last action chosen by `decide()`; cleared by `reset()`.

## Planned (design-spec)

- Not specified.

## Deviations

- The class docstring states delegation is to `Q_Sea_Battle.logit_utils`, but the module imports from `.logit_utilities` and delegates to `Q_Sea_Battle.logit_utilities.logit_to_prob` and `Q_Sea_Battle.logit_utilities.logit_to_logprob`.

## Notes for Contributors

- `decide()` converts actions to `np.float32` during sampling/thresholding and returns `actions.astype(int)`; keep this in mind if downstream code expects specific dtypes.
- `_scale_field()` is a module-level helper (non-public) that applies the affine transform `x_scaled = x - 0.5` after casting to `np.float32`; changes to scaling will affect model input distribution.

## Related

- `Q_Sea_Battle.players_base.PlayerA`
- `Q_Sea_Battle.game_layout.GameLayout`
- `Q_Sea_Battle.logit_utilities.logit_to_prob`
- `Q_Sea_Battle.logit_utilities.logit_to_logprob`

## Changelog

- 0.1: Initial version in module docstring; provides `NeuralNetPlayerA` with exploration option and log-prob tracking.