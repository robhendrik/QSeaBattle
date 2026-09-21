# Q_Sea_Battle.pr_assisted_replay

> Role: Implements a PR-assisted replay shared-resource (SR) TensorFlow/Keras layer that consumes logits and produces logits, supporting both trace-replay and stochastic gameplay modes with differentiable PR gating.

Location: `Q_Sea_Battle.pr_assisted_replay`

## Overview

This module defines a logits-in/logits-out Keras layer implementing the project’s PR-assisted *replay* shared-resource (SR) behavior. Each call produces exactly one outcome tensor (logits) and is stateless with respect to call ordering; the caller must explicitly indicate whether the call corresponds to the first or second measurement via the `first_measurement` input tensor.

Two SR modes are supported: replay (`sr_mode="replay"`) and stochastic (`sr_mode="stochastic"`). In replay mode, first-measurement outcomes are prescribed by an optional `replay_outcome_logits` input (required when any element is first measurement), while second-measurement outcomes follow a differentiable PR-rule computation and may be noised via follow/violate sampling or replaced by its expectation. In stochastic mode, first measurements are sampled uniformly at random (50/50) and mapped to fixed-magnitude logits; second measurements follow or violate the PR rule with probability `p_rule`.

The PR rule is computed without hard thresholding: a smooth “soft-high” gate uses `sigmoid(alpha * logit)` and a soft AND to compute a flip probability, implementing “opposite” as logit negation.

## Public API

### Functions

#### `PRAssistedReplay.__init__(self, sr_mode: str = "replay", *, p_rule: float = 1.0, beta: float = 10.0, alpha: float = 5.0, seed: int | None = None, **kwargs: Any) -> None`

**Purpose:** Construct the PR-assisted SR layer, validating configuration and creating runtime `tf.Variable` knobs and an internal RNG.

**Arguments:**
- `sr_mode`: SR mode; one of `"replay"` or `"stochastic"`.
- `p_rule`: Probability of following the PR rule for second measurements.
- `beta`: Magnitude for mapping bits to logits in stochastic first measurements.
- `alpha`: Sharpness for the differentiable PR gate.
- `seed`: Optional RNG seed for reproducible stochastic sampling.
- `**kwargs`: Forwarded to `tf.keras.layers.Layer`.

**Returns:** `None`.

**Errors:**
- `ValueError`: If `sr_mode` is not in `{"replay","stochastic"}`.
- `ValueError`: If `p_rule` is not in `[0, 1]`.
- `ValueError`: If `beta <= 0`.
- `ValueError`: If `alpha <= 0`.

**Example:**
```python
import tensorflow as tf
from Q_Sea_Battle.pr_assisted_replay import PRAssistedReplay

layer = PRAssistedReplay(sr_mode="replay", p_rule=0.9, alpha=5.0, beta=10.0, seed=123)

k = 4
inputs = {
    "current_measurement": tf.zeros([2, k], tf.float32),
    "previous_measurement": tf.zeros([2, k], tf.float32),
    "previous_outcome": tf.ones([2, k], tf.float32),
    "first_measurement": tf.ones([2, 1], tf.float32),
    "replay_outcome_logits": tf.fill([2, k], 3.0),
}
y = layer(inputs)
```

#### `PRAssistedReplay.config(self) -> PRAssistedReplayConfig` (property)

**Purpose:** Return the immutable construction-time configuration dataclass captured at initialization.

**Arguments:** None.

**Returns:** `PRAssistedReplayConfig`.

**Errors:** Not specified.

**Example:**
```python
cfg = layer.config
print(cfg.sr_mode, cfg.p_rule, cfg.alpha, cfg.beta, cfg.seed)
```

#### `PRAssistedReplay.get_config(self) -> dict[str, Any]`

**Purpose:** Provide a Keras-serializable configuration dictionary (includes `sr_mode`, `p_rule`, `beta`, `alpha`, `seed`).

**Arguments:** None.

**Returns:** `dict[str, Any]`.

**Errors:** Not specified.

**Example:**
```python
keras_cfg = layer.get_config()
```

#### `PRAssistedReplay.set_alpha(self, alpha: float) -> None`

**Purpose:** Update the runtime `alpha` (soft gate sharpness) used by the differentiable PR gate.

**Arguments:**
- `alpha`: New positive value.

**Returns:** `None`.

**Errors:**
- `ValueError`: If `alpha <= 0`.

**Example:**
```python
layer.set_alpha(7.5)
```

#### `PRAssistedReplay.set_p_rule(self, p_rule: float) -> None`

**Purpose:** Update the runtime `p_rule` (follow probability) used for second-measurement follow/violate noise in replay and stochastic modes.

**Arguments:**
- `p_rule`: New probability in `[0, 1]`.

**Returns:** `None`.

**Errors:**
- `ValueError`: If `p_rule` is not in `[0, 1]`.

**Example:**
```python
layer.set_p_rule(0.25)
```

#### `PRAssistedReplay.set_beta(self, beta: float) -> None`

**Purpose:** Update the runtime `beta` used for mapping sampled bits `{0,1}` to logits `{-beta,+beta}` in stochastic first measurements.

**Arguments:**
- `beta`: New positive value.

**Returns:** `None`.

**Errors:**
- `ValueError`: If `beta <= 0`.

**Example:**
```python
layer.set_beta(12.0)
```

#### `PRAssistedReplay.set_sr_mode(self, sr_mode: str) -> None`

**Purpose:** Update the runtime SR mode.

**Arguments:**
- `sr_mode`: One of `"replay"` or `"stochastic"`.

**Returns:** `None`.

**Errors:**
- `ValueError`: If `sr_mode` is not in `{"replay","stochastic"}`.

**Example:**
```python
layer.set_sr_mode("stochastic")
```

#### `PRAssistedReplay.set_pr_noise_mode(self, mode: str) -> None`

**Purpose:** Control replay-mode second-measurement PR noise handling: sample follow/violate decisions or use their expectation.

**Arguments:**
- `mode`: Either `"sampled"` (sample follow/violate using `p_rule`) or `"expected"` (use expectation).

**Returns:** `None`.

**Errors:**
- `ValueError`: If `mode` is not one of `{"expected","sampled"}`.

**Example:**
```python
layer.set_sr_mode("replay")
layer.set_pr_noise_mode("expected")
```

#### `PRAssistedReplay.get_pr_noise_mode(self) -> str`

**Purpose:** Return the current PR noise mode as a Python string.

**Arguments:** None.

**Returns:** `"expected"` if the internal mode code equals expected, otherwise `"sampled"`.

**Errors:** Not specified.

**Example:**
```python
mode = layer.get_pr_noise_mode()
```

#### `PRAssistedReplay.call(self, inputs: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor`

**Purpose:** Compute outcome logits for either the first or second measurement, as indicated per element by `first_measurement`.

**Arguments:**
- `inputs`: Dictionary of tensors. Required keys:
  - `current_measurement`: `tf.Tensor` float32, shape `(..., k)`, logits.
  - `previous_measurement`: `tf.Tensor` float32, shape `(..., k)`, logits.
  - `previous_outcome`: `tf.Tensor` float32, shape `(..., k)`, logits.
  - `first_measurement`: `tf.Tensor` float32, shape `(..., 1)`, broadcastable to `(..., k)`. Values should be in `{0,1}`; `>= 0.5` is treated as “first measurement”.
  - `replay_outcome_logits` (optional): `tf.Tensor` float32, shape `(..., k)`, logits. Required in replay mode when any element indicates a first measurement.
- `training`: Keras training flag; accepted for compatibility but not used (behavior is controlled by SR mode and inputs).

**Returns:** `tf.Tensor` float32 of shape `(..., k)` containing outcome logits.

**Errors:**
- `TypeError`: If `inputs` is not a `dict`.
- `ValueError`: If required keys are missing.
- `ValueError`: If `current_measurement` and `previous_measurement` have different static ranks (when known).
- `ValueError`: If `current_measurement` and `previous_outcome` have different static ranks (when known).
- `ValueError`: If `first_measurement` last dimension is statically known and not `1`.
- `ValueError`: In eager mode, if `sr_mode` is replay, `replay_outcome_logits` is missing, and any element indicates first measurement.
- `tf.errors.InvalidArgumentError`: In graph mode (`tf.function`), under the same missing-`replay_outcome_logits` condition (via `tf.debugging.assert_equal`).

**Example:**
```python
import tensorflow as tf
from Q_Sea_Battle.pr_assisted_replay import PRAssistedReplay

layer = PRAssistedReplay(sr_mode="stochastic", p_rule=0.8, seed=1)

k = 3
inputs = {
    "current_measurement": tf.random.normal([5, k]),
    "previous_measurement": tf.random.normal([5, k]),
    "previous_outcome": tf.random.normal([5, k]),
    "first_measurement": tf.concat([tf.ones([2, 1]), tf.zeros([3, 1])], axis=0),
}
y = layer(inputs)  # shape [5, k], float32 logits
```

### Constants

- `PRAssistedReplay._SR_MODE_REPLAY`: Internal integer code for replay mode (`0`).
- `PRAssistedReplay._SR_MODE_STOCHASTIC`: Internal integer code for stochastic mode (`1`).
- `PRAssistedReplay._PR_NOISE_SAMPLED`: Internal integer code for sampled PR noise (`0`).
- `PRAssistedReplay._PR_NOISE_EXPECTED`: Internal integer code for expected PR noise (`1`).

### Types

#### `PRAssistedReplayConfig`

**Kind:** `@dataclass(frozen=True)`

**Purpose:** Immutable construction-time configuration for `PRAssistedReplay`, captured for serialization and inspection. Runtime execution uses internal `tf.Variable` instances for selected parameters to support updates via `set_*` methods.

**Fields:**
- `sr_mode: str = "replay"`: Shared-resource mode; one of `"replay"` or `"stochastic"`.
- `p_rule: float = 1.0`: Probability of following the PR rule on the second measurement.
- `beta: float = 10.0`: Magnitude for mapping bits `{0,1}` to logits `{-beta,+beta}`.
- `alpha: float = 5.0`: Sharpness (inverse temperature) for the differentiable soft-high gate.
- `seed: Optional[int] = None`: RNG seed for stochastic sampling (when provided).

## Dependencies

- Python stdlib: `dataclasses.dataclass`, `typing.Any`, `typing.Optional`
- Third-party: `tensorflow` (uses `tf.keras.layers.Layer`, `tf.Variable`, `tf.random.Generator`, and core tensor ops)

## Planned (design-spec)

Not specified (no design notes provided beyond the module docstring).

## Deviations

Not specified.

## Notes for Contributors

- The layer is logits-only: do not add sigmoid/thresholding to convert logits to bits inside this module; sign interpretation is intended to occur externally.
- First/second measurement selection is per-element using `first_measurement >= 0.5` and supports broadcasting from `(..., 1)` to `(..., k)`.
- Replay mode requires `replay_outcome_logits` when any element is a first measurement; eager mode raises `ValueError`, while graph mode uses a TensorFlow assertion to raise `InvalidArgumentError`.
- Stochastic behavior is driven by an internal `tf.random.Generator`; seed handling is implemented in `__init__` and should remain stable for backward compatibility.

## Related

- TensorFlow Keras: `tf.keras.layers.Layer`
- Module concept: PR-assisted shared-resource (SR) replay layer (project-specific; further references not specified)

## Changelog

Not specified.