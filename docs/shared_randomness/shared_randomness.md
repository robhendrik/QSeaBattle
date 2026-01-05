# Shared randomness resource (non-classical)

## Purpose
Define a two-party shared-randomness resource that produces **non-classical correlations** between
Player A and Player B outcomes, without increasing communication bandwidth.

This specification defines the **behavioral contract** of shared randomness. Multiple implementations
(e.g., a pure Python class and a Keras layer) MUST satisfy the same contract.


## Location
- **Python implementation (example):** `src/Q_Sea_Battle/shared_randomness.py` -- `class SharedRandomness`
- **Differentiable implementation (example):** `src/Q_Sea_Battle/shared_randomness_layer.py` -- `class SharedRandomnessLayer`


## Parameters
| Name | Type | Constraints | Description |
|------|------|-------------|-------------|
| `length` | `int` | `> 0` | Number of correlated outcome bits produced per measurement |
| `p_high` | `float` | `[0, 1]` | Correlation strength parameter |


## Public interface (conceptual)

### `reset() -> None`
Reset per-game state so the resource can be used again.

### `measurement_a(measurement: array_like) -> np.ndarray`
Player A performs a measurement and receives a binary outcome vector of length `length`.

### `measurement_b(measurement: array_like) -> np.ndarray`
Player B performs a measurement and receives a binary outcome vector of length `length`.

> The `measurement` input is a *measurement choice* (a â€œsettingâ€) and MAY be encoded as a binary vector,
> integer index, or tensor -- but it MUST be documented by the implementation and used consistently by both parties.


## Behavioral contract (normative)

### Single-use per party
- Each party MUST call its measurement method at most once per resource instance per game.
- Calling `measurement_a` twice, or `measurement_b` twice, MUST raise an error.

### Two-measurement lifecycle
- The resource supports up to two measurements total (one per party) between resets.

### Output format
- Outcomes MUST be binary arrays in `{0,1}` of shape `(length,)` (or `(B, length)` for batched/tensor implementations).

### Non-classical correlation requirement
Let `x` be Player Aâ€™s outcome, `y` Player Bâ€™s outcome (bitwise).  
The joint distribution MUST satisfy:

- `P(x_i = y_i) = p_high`
- `P(x_i â‰  y_i) = 1 - p_high`

for each bit index `i`, **conditional on both parties using the same measurement choice**.

If measurement choices differ, the implementation MUST still:
- return valid binary outcomes, and
- produce correlations that are a documented function of both measurement choices.

> This clause is what makes the resource â€œnon-classicalâ€ in this project: the correlation structure is an explicit,
> tunable contract, and not restricted to classical shared coins.

### No information leakage
- The resource MUST NOT depend on `field`, `gun`, or any game secrets.
- It MUST NOT encode additional side channels beyond its specified outputs.


## Failure modes
- `ValueError` (recommended) if a party measures twice.
- `RuntimeError` if used without reset across games (implementation-dependent).


## Notes (informative)
Two compliant implementations are common:
1. **Pure Python**: explicit sampling of correlated bits with the required joint distribution.
2. **Keras layer**: a differentiable proxy used during training that matches the same marginal/correlation behavior.
