# Class <ClassName>

> Copy this file to the appropriate section (e.g. `docs/game/GameLayout.md`) and replace placeholders.

## Purpose
One sentence: what this class/module exists to do.

## Location
- **Module:** `src/Q_Sea_Battle/<module_name>.py`
- **Class:** `<ClassName>`

## Public interface

### Constructor
```python
<ClassName>(...)
```

### Methods
- `<method_name>(...) -> ...`
- `<method_name>(...) -> ...`

## Inputs
| Name | Shape | Type | Constraints | Description |
|------|-------|------|-------------|-------------|
| `<x>` | `(B, ...)` | `np.ndarray` / `tf.Tensor` | e.g. `{0,1}` | ... |

## Outputs
| Name | Shape | Type | Constraints | Description |
|------|-------|------|-------------|-------------|
| `<y>` | `(B, ...)` | `np.ndarray` / `tf.Tensor` | range | ... |

## Internal state
List the state that persists between calls (if any). If stateless, say so.

- `<state_name>`: type/shape — description

## Behavioral contract
Normative statements (use **MUST**, **MUST NOT**, **SHOULD**):

- The class **MUST** ...
- The method **MUST NOT** ...
- The implementation **SHOULD** ...

## Invariants
One-sentence truths that must always hold (also consider copying key ones into `docs/invariants.md`):

- `<invariant 1>`
- `<invariant 2>`

## Failure modes
Specify errors and when they occur:

- `ValueError` if ...
- `TypeError` if ...
- `RuntimeError` if ...

## Notes / rationale
Short explanation of design choices, tradeoffs, and links to theory.

## Related
- Used by: `<other classes>`
- Depends on: `<other modules>`
- Tests: `tests/<test_file>.py`

## Math (optional)
Inline math: $p_{high} \in [0,1]$

Display math:

$$
s_\text{ideal} = \frac{1}{2}\left(1 + \left(\frac{K}{4}\right)^{\log_2(n^2)}\right)
$$