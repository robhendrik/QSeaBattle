
# {Module Title}

> **One‑liner**: {what the module is responsible for in one sentence}.

**Module path**: `{python.import.path}`  
**Status**: {draft|stable} · **Version**: {x.y.z} · **Owner**: {team/person}

## Purpose & Scope
- **Goal**: {what problems the module solves; why it exists}
- **Non-goals**: {explicitly out-of-scope items}

## Public Interface (Summary)
- **Classes**: `{ClassA}`, `{ClassB}`, …
- **Key functions**: `{fn_a()}`, `{fn_b()}`, …
- **External contracts**: depends on `{OtherModule}`, emits `{TournamentLog}` rows, etc.

## Types & Shapes
- `n2 = field_size^2` (must be power of 2 where specified)
- `m = comms_size` (must divide `n2` where specified)
- Arrays: NumPy `np.ndarray` with dtype `int` in `{0,1}` unless noted
- Tensors: TensorFlow `tf.Tensor` with `float32` unless noted

## Behavior
1. {Step}  
2. {Step}  
3. {Step}

!!! example "Minimal usage"
    ```python
    from Q_Sea_Battle.{module} import {ClassA}

    layout = GameLayout(field_size=4, comms_size=1)
    comp = {ClassA}(layout)
    result = comp.do_something(...)
    ```

## Validation & Errors
- Raises `ValueError` when …  
- Raises `TypeError` when …  
- Runtime invariants: {invariants that must hold}

## Performance
- Complexity: {big‑O if relevant}  
- Notes on vectorisation / batching.

## Testing Hooks
- Suggested unit tests / invariants.

## Changelog
- {date} — {change}
