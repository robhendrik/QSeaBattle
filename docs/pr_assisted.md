# PRAssisted

> Role: Stateful two-party PR-assisted shared resource that returns correlated 0/1 outcome strings per round, with optional deterministic replay.

Location: `Q_Sea_Battle.pr_assisted.PRAssisted`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| length | int, constraint: $>= 1$ | Number of bits per measurement/outcome string. |
| p_rule | float, constraint: $0.0 \le p\_rule \le 1.0$ | Correlation parameter controlling how likely the second outcome matches (or flips) the first as a function of both parties' measurement settings. |

Preconditions

- `length` is `int` and `length >= 1`.
- `p_rule` is `int | float` and `0.0 <= float(p_rule) <= 1.0`.

Postconditions

- `self.length: int` is set to `length`.
- `self.p_rule: float` is set to `float(p_rule)`.
- Per-round measurement state is cleared: `a_measured == False`, `b_measured == False`, `prev_party is None`, `prev_measurement is None`, `prev_outcome is None`.
- Replay is disabled and cleared: `_replay_enabled == False`, `_replay_a_outcome is None`, `_replay_b_outcome is None`, `_replay_first_party is None`, `_replay_consumed_a == False`, `_replay_consumed_b == False`.
- A local RNG is created: `_rng = np.random.default_rng()`.

Errors

- Raises `TypeError` if `length` is not an `int`.
- Raises `ValueError` if `length < 1`.
- Raises `TypeError` if `p_rule` is not an `int | float`.
- Raises `ValueError` if `p_rule` is not in `[0.0, 1.0]` (after conversion to `float`).

Example

```python
import numpy as np
from Q_Sea_Battle.pr_assisted import PRAssisted

sr = PRAssisted(length=8, p_rule=0.9)
sr.reset()

a_meas = np.zeros(8, dtype=int)
b_meas = np.ones(8, dtype=int)

a_out = sr.measurement_a(a_meas)
b_out = sr.measurement_b(b_meas)
```

## Public Methods

### measurement_a(measurement)

Query the resource for party A; each party may query at most once per round.

- First query of a round (by either party) returns a uniformly random 0/1 vector of length `self.length`.
- Second query returns a 0/1 vector correlated with the first according to `_second_measurement(...)` (stochastic mode only).
- In replay mode, returns the prescribed outcome for A (if provided) and does not enforce correlation.

Arguments

- measurement: np.ndarray, dtype int (after conversion), values in `{0,1}`, shape `(length,)`; Party A measurement setting.

Returns

- np.ndarray, dtype int, values in `{0,1}`, shape `(length,)`; Party A outcome vector.

Raises

- `ValueError` if party A already queried in this round (`self.a_measured` is already `True`).
- `ValueError` if `measurement` is not 1D, has shape not equal to `(length,)`, or contains values other than `0/1`.
- `RuntimeError` if replay mode is enabled and the prescribed outcome for A is missing (`_replay_a_outcome is None`).
- `RuntimeError` if replay mode has `first_party` set and the first query this round violates it.

Side effects

- Sets `self.a_measured = True` on entry (after validation).
- In replay mode, caches the query as the first query: sets `prev_party = "a"`, `prev_measurement` to a copy of the validated measurement, and `prev_outcome` to a copy of the returned outcome.
- In stochastic mode, may update cached first-query state via `_first_measurement(...)` if A is first in the round.

### measurement_b(measurement)

Query the resource for party B; each party may query at most once per round.

Behavior is symmetric to `measurement_a(...)`, with party label `"b"`.

Arguments

- measurement: np.ndarray, dtype int (after conversion), values in `{0,1}`, shape `(length,)`; Party B measurement setting.

Returns

- np.ndarray, dtype int, values in `{0,1}`, shape `(length,)`; Party B outcome vector.

Raises

- `ValueError` if party B already queried in this round (`self.b_measured` is already `True`).
- `ValueError` if `measurement` is not 1D, has shape not equal to `(length,)`, or contains values other than `0/1`.
- `RuntimeError` if replay mode is enabled and the prescribed outcome for B is missing (`_replay_b_outcome is None`).
- `RuntimeError` if replay mode has `first_party` set and the first query this round violates it.

Side effects

- Sets `self.b_measured = True` on entry (after validation).
- In replay mode, caches the query as the first query: sets `prev_party = "b"`, `prev_measurement` to a copy of the validated measurement, and `prev_outcome` to a copy of the returned outcome.
- In stochastic mode, may update cached first-query state via `_first_measurement(...)` if B is first in the round.

### reset()

Reset the resource for the next round; also disables and clears replay configuration.

Arguments

- None.

Returns

- None.

Side effects

- Clears per-round measurement state: `a_measured = False`, `b_measured = False`, `prev_party = None`, `prev_measurement = None`, `prev_outcome = None`.
- Calls `clear_replay_round()`, which disables and clears replay state for the current round.

### set_replay_round(*, a_outcome=None, b_outcome=None, first_party=None)

Enable replay mode for the current round; when enabled, measurements return prescribed outcomes instead of sampling.

Arguments

- a_outcome: np.ndarray | None, dtype int (after conversion), values in `{0,1}`, shape `(length,)`; Optional prescribed outcome for party A.
- b_outcome: np.ndarray | None, dtype int (after conversion), values in `{0,1}`, shape `(length,)`; Optional prescribed outcome for party B.
- first_party: str | None, constraint: in `{"a","b",None}`; If provided, enforces which party must make the first query in this round.

Returns

- None.

Raises

- `ValueError` if `first_party` is not in `{"a", "b", None}`.
- `ValueError` if `a_outcome` or `b_outcome` is not 1D, has shape not equal to `(length,)`, or contains values other than `0/1`.

Side effects

- Sets `_replay_enabled = True`.
- Stores validated outcomes (or `None`): `_replay_a_outcome`, `_replay_b_outcome`.
- Sets `_replay_first_party = first_party`.
- Resets consumption flags: `_replay_consumed_a = False`, `_replay_consumed_b = False`.

### clear_replay_round()

Disable replay mode and clear replay configuration for this round.

Arguments

- None.

Returns

- None.

Side effects

- Sets `_replay_enabled = False`.
- Clears replay configuration and consumption flags: `_replay_a_outcome = None`, `_replay_b_outcome = None`, `_replay_first_party = None`, `_replay_consumed_a = False`, `_replay_consumed_b = False`.

### replay_enabled()

Whether replay mode is enabled for the current round.

Arguments

- None.

Returns

- bool, constraint: in `{True, False}`; `True` iff replay mode is enabled for the current round.

## Data & State

Public attributes

- length: int, constraint: $>= 1$; Number of bits per measurement/outcome string.
- p_rule: float, constraint: $0.0 \le p\_rule \le 1.0$; Correlation parameter used by `_second_measurement(...)` in stochastic mode.
- a_measured: bool, constraint: in `{True, False}`; Whether party A has queried in the current round.
- b_measured: bool, constraint: in `{True, False}`; Whether party B has queried in the current round.
- prev_party: str | None, constraint: in `{"a","b",None}`; Party label for the first query in the current round.
- prev_measurement: np.ndarray | None, dtype int (after conversion), values in `{0,1}`, shape `(length,)`; Measurement vector cached from the first query in the current round.
- prev_outcome: np.ndarray | None, dtype int, values in `{0,1}`, shape `(length,)`; Outcome vector cached from the first query in the current round.

Private/internal state (implementation details)

- _replay_enabled: bool, constraint: in `{True, False}`; Replay mode toggle for the current round.
- _replay_a_outcome: np.ndarray | None, dtype int, values in `{0,1}`, shape `(length,)`; Prescribed replay outcome for party A.
- _replay_b_outcome: np.ndarray | None, dtype int, values in `{0,1}`, shape `(length,)`; Prescribed replay outcome for party B.
- _replay_first_party: str | None, constraint: in `{"a","b",None}`; Enforced first party in replay mode, if provided.
- _replay_consumed_a: bool, constraint: in `{True, False}`; Marks whether A's replay outcome has been returned in this round.
- _replay_consumed_b: bool, constraint: in `{True, False}`; Marks whether B's replay outcome has been returned in this round.
- _rng: np.random.Generator; Local RNG used for stochastic sampling.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- The class is stateful per round; use `reset()` between rounds to clear `a_measured`, `b_measured`, and cached first-query data.
- Replay mode returns prescribed outcomes and does not enforce the correlation rule described in `_second_measurement(...)`; this is intentional per docstrings.

## Related

- NumPy random Generator API: `np.random.default_rng()` (used internally for sampling).

## Changelog

- Not specified.