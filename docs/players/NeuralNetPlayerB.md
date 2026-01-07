# Class NeuralNetPlayerB

**Module import path**: `Q_Sea_Battle.neural_net_player_b.NeuralNetPlayerB`

> Player B wrapper around `model_b` that produces a binary `shoot` decision and stores a log-probability.

!!! note "Derived constraints from GameLayout"
    Let `field_size = n` and `n2 = n**2`. Let `comms_size = m`.

    - `n2` is a power of two.
    - `m | n2` (i.e. `n2 % m == 0`).

## Constructor

### Signature

- `NeuralNetPlayerB(game_layout, model_b, explore=True)`

### Arguments

- `game_layout`: `GameLayout`, scalar.
- `model_b`: `tf.keras.Model`, scalar.
- `explore`: `bool`, scalar.

### Returns

- `NeuralNetPlayerB`, scalar.

### Preconditions

- `game_layout` is a valid `GameLayout`, scalar.
- `model_b` satisfies:
  - input: `tf.Tensor`, dtype `float32`, shape `(B, 1 + m)`
  - output: `tf.Tensor`, dtype `float32`, shape `(B, 1)`

### Postconditions

- `self.game_layout` references `game_layout`.
- `self.model_b` references `model_b`.
- `self.explore` is set.
- `self.last_logprob` is `None`.

### Errors

- No explicit errors are raised by the constructor.

## Public Methods

### decide

#### Signature

- `decide(gun: np.ndarray, comm: np.ndarray, supp=None) -> int`

#### Arguments

- `gun`: `np.ndarray`, dtype `int` (0, 1), shape `(n2,)`, one-hot (intended).
- `comm`: `np.ndarray`, dtype `int` (0, 1), shape `(m,)`.
- `supp`: `Any` or `None`, scalar.

#### Returns

- `shoot`: `int` (0, 1), scalar.

#### Preconditions

- `gun` can be converted and reshaped to:
  - `x_gun`: `np.ndarray`, dtype `float32`, shape `(1, n2)`.
- `comm` can be converted and reshaped to:
  - `x_comm`: `np.ndarray`, dtype `float32`, shape `(1, m)`.
- A model input can be formed:
  - `x`: `np.ndarray`, dtype `float32`, shape `(1, 1 + m)`,
    where the first feature is a normalised gun index derived from `argmax(gun)`.

#### Postconditions

- Stores `self.last_logprob`:
  - `float`, scalar, log-probability of the chosen `shoot` under the model logit.
- Returns `shoot` as a Python `int` in `(0, 1)`.

#### Errors

- Propagates exceptions from:
  - shape conversion,
  - model forward pass,
  - numerical/log-prob utility calls.

!!! note "Deterministic vs stochastic"
    - If `explore` is `False`, action is `1` if `prob >= 0.5` else `0`.
    - If `explore` is `True`, action is sampled from `Bernoulli(prob)`.

### get_log_prob

#### Signature

- `get_log_prob() -> float`

#### Arguments

- None.

#### Returns

- `log_prob`: `float`, scalar.

#### Preconditions

- `decide(...)` has been called since the last `reset()`.

#### Postconditions

- No mutation.

#### Errors

- Raises `RuntimeError` if `self.last_logprob` is `None`.

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

- `self.last_logprob` is set to `None`.

#### Errors

- No explicit errors are raised.

## Public Static Methods

### logit_to_probs

#### Signature

- `logit_to_probs(logits) -> probs`

#### Arguments

- `logits`: `np.ndarray`, dtype `float32`, shape `(k,)` or `float`, scalar.

#### Returns

- `probs`: `np.ndarray`, dtype `float32`, shape `(k,)` or `float`, scalar.

#### Preconditions

- Inputs are numeric.

#### Postconditions

- Pure conversion; no mutation.

#### Errors

- Propagates exceptions from numerical operations in the shared utility.

### logit_to_log_probs

#### Signature

- `logit_to_log_probs(logits, actions) -> log_probs`

#### Arguments

- `logits`: `np.ndarray`, dtype `float32`, shape `(k,)` or `float`, scalar.
- `actions`: `np.ndarray`, dtype `float32`, shape `(k,)` or `float`, scalar.

#### Returns

- `log_probs`: `np.ndarray`, dtype `float32`, shape `(k,)` or `float`, scalar.

#### Preconditions

- Inputs are numeric and broadcast-compatible.

#### Postconditions

- Pure conversion; no mutation.

#### Errors

- Propagates exceptions from numerical operations in the shared utility.

## Data & State

- `game_layout`: `GameLayout`, scalar.
- `model_b`: `tf.keras.Model`, scalar.
- `explore`: `bool`, scalar.
- `last_logprob`: `float` or `None`, scalar.

## Planned (design-spec)

- None identified.

## Deviations

- Naming:
  - Requested in task: `NeuraNetPlayerB`.
  - Implemented in codebase: `NeuralNetPlayerB`.
- Gun encoding:
  - Runtime wrapper compresses the one-hot gun to a normalised scalar index internally.

## Notes for Contributors

- Keep gun preprocessing stable:
  - Normalised index is derived from `argmax(gun)` and mapped to `[0.0, 1.0]`.
- Ensure `decide(...)` returns a Python `int` in `(0, 1)`.
- Keep `get_log_prob()` semantics aligned with tournament logging expectations.

## Examples

!!! example "Minimal usage"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.neural_net_player_b import NeuralNetPlayerB

    layout = GameLayout(field_size=4, comms_size=2)
    model_b = tf.keras.Sequential([
        tf.keras.Input(shape=(1 + layout.comms_size,)),
        tf.keras.layers.Dense(1),
    ])

    player_b = NeuralNetPlayerB(layout, model_b=model_b, explore=False)
    ```

## Changelog

- 2026-01-07 — Author: Rob Hendriks
