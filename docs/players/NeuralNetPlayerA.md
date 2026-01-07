# Class NeuralNetPlayerA

**Module import path**: `Q_Sea_Battle.neural_net_player_a.NeuralNetPlayerA`

> Player A wrapper around `model_a` that produces an `m`-bit communication vector and stores a log-probability.

!!! note "Derived constraints from GameLayout"
    Let `field_size = n` and `n2 = n**2`. Let `comms_size = m`.

    - `n2` is a power of two.
    - `m | n2` (i.e. `n2 % m == 0`).

## Constructor

### Signature

- `NeuralNetPlayerA(game_layout, model_a, explore=True)`

### Arguments

- `game_layout`: `GameLayout`, scalar.
- `model_a`: `tf.keras.Model`, scalar.
- `explore`: `bool`, scalar.

### Returns

- `NeuralNetPlayerA`, scalar.

### Preconditions

- `game_layout` is a valid `GameLayout`, scalar.
- `model_a` satisfies:
  - input: `tf.Tensor`, dtype `float32`, shape `(B, n2)`
  - output: `tf.Tensor`, dtype `float32`, shape `(B, m)`

### Postconditions

- `self.game_layout` references `game_layout`.
- `self.model_a` references `model_a`.
- `self.explore` is set.
- `self.last_logprob` is `None`.

### Errors

- No explicit errors are raised by the constructor.

## Public Methods

### decide

#### Signature

- `decide(field: np.ndarray, supp=None) -> np.ndarray`

#### Arguments

- `field`: `np.ndarray`, dtype `int` (0, 1), shape `(n2,)`.
- `supp`: `Any` or `None`, scalar.

#### Returns

- `comm`: `np.ndarray`, dtype `int` (0, 1), shape `(m,)`.

#### Preconditions

- `field` can be converted and reshaped to:
  - `x`: `np.ndarray`, dtype `float32`, shape `(1, n2)`.
- `model_a(x_scaled)` returns:
  - `logits`: `tf.Tensor`, dtype `float32`, shape `(1, m)`.

#### Postconditions

- Stores `self.last_logprob`:
  - `float`, scalar, log-probability of the chosen `comm` under the logits.
- Returns `comm` as integer bits in `(0, 1)`.

#### Errors

- Propagates exceptions from:
  - shape conversion,
  - model forward pass,
  - numerical/log-prob utility calls.

!!! note "Deterministic vs stochastic"
    - If `explore` is `False`, bits are chosen as `prob >= 0.5`.
    - If `explore` is `True`, each bit is sampled independently from `Bernoulli(prob)`.

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
- `model_a`: `tf.keras.Model`, scalar.
- `explore`: `bool`, scalar.
- `last_logprob`: `float` or `None`, scalar.

## Planned (design-spec)

- None identified.

## Deviations

- Naming:
  - Requested in task: `NeuraNetPlayerA`.
  - Implemented in codebase: `NeuralNetPlayerA`.

## Notes for Contributors

- Keep preprocessing stable:
  - Input field is scaled as `field - 0.5` before passing to `model_a`.
- Ensure output `comm` is `np.ndarray`, dtype `int` `(0, 1)`, shape `(m,)`.
- Do not change the `get_log_prob()` contract without updating tournament logging expectations.

## Examples

!!! example "Minimal usage"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.neural_net_player_a import NeuralNetPlayerA

    layout = GameLayout(field_size=4, comms_size=2)
    model_a = tf.keras.Sequential([
        tf.keras.Input(shape=(layout.field_size * layout.field_size,)),
        tf.keras.layers.Dense(layout.comms_size),
    ])

    player_a = NeuralNetPlayerA(layout, model_a=model_a, explore=False)
    ```

## Changelog

- 2026-01-07 — Author: Rob Hendriks
