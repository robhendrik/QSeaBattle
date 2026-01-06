
# {ClassName}

> **Role**: {what this class does in one sentence}.

**Location**: `{python.import.path.ClassName}`

## Constructor
| Parameter | Type | Description |
|---|---|---|
| `{param}` | `{type}` | {explanation, constraints} |

**Preconditions**
- {shape/type constraints; example: `field_size > 0`, `m | n2`}

**Postconditions**
- {state after construction; immutable attributes, etc.}

## Public Methods

### `{method_name}({sig})`
**Purpose**: {one-sentence}.  
**Args**:  
- `{name}` — `{type}`, `shape (…)`, {meaning/constraints}.  
**Returns**: `{type}`, `shape (…)`.  
**Raises**: `{ErrorType}` when …

!!! example "Example"
    ```python
    out = obj.{method_name}(...)
    ```

## Data & State
- Attributes (public): `{attr}` — `{type}` — {constraints}
- Side effects: {if any}
- Thread-safety: {n/a unless specified}

## Notes for Contributors
- Implementation hints, pitfalls, and alignment with spec.

## Related
- See also: [{OtherClass}](../spec/other_class.md)
