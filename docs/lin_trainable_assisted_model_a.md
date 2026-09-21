# LinTrainableAssistedModelA

> Role: Backward-compatible wrapper that preserves a legacy constructor interface while delegating behavior to `LinInternalModelA`.

Location: `Q_Sea_Battle.lin_trainable_assisted_model_a.LinTrainableAssistedModelA`

## Constructor

Parameter | Type | Description
---|---|---
field_size | int, constraint: convertible via `int(field_size)`; shape: scalar | Size of the game field; stored on `self.field_size` and forwarded via a minimal layout object to the base class.
comms_size | int, constraint: convertible via `int(comms_size)`; shape: scalar | Size of the communication channel; stored on `self.comms_size` and forwarded via a minimal layout object to the base class.
sr_mode | str, constraint: any string accepted but must map to supported aliases; shape: scalar | SR mode; legacy aliases are normalized via `_map_sr_mode` before being passed to the base class; default is `"sample"`.
p_rule | float, constraints: not specified; shape: scalar | Probability of applying the rule-based/shared-resource path; forwarded to the base class.
beta | float, constraints: not specified; shape: scalar | Hyperparameter forwarded to the base class.
alpha | float, constraints: not specified; shape: scalar | Hyperparameter forwarded to the base class.
seed | Optional[int], constraints: `None` or `int`; shape: scalar | Optional random seed forwarded to the base class.
hidden_units_measure | int, constraints: not specified; shape: scalar | Hidden units for the internal "measure" MLP; forwarded to the base class.
hidden_units_combine | int, constraints: not specified; shape: scalar | Hidden units for the internal "combine" MLP; forwarded to the base class.
name | Optional[str], constraints: `None` or `str`; shape: scalar | Optional model name forwarded to the base class.
**kwargs | Any, constraints: accepted but ignored; shape: not applicable | Accepted for backward compatibility; currently unused and intentionally ignored.

Preconditions

- `sr_mode` must be one of the supported aliases accepted by `_map_sr_mode`: `"sample"`, `"stochastic"`, `"expected"`, or `"replay"` (case-insensitive after `str(...).lower()` conversion).
- `field_size` and `comms_size` must be convertible to `int` via `int(...)`.

Postconditions

- `self.field_size` is set to `int(field_size)`.
- `self.comms_size` is set to `int(comms_size)`.
- A minimal layout-like object with attributes `field_size` and `comms_size` is constructed and passed as the first positional argument to `LinInternalModelA.__init__`.
- `LinInternalModelA.__init__` is invoked with `sr_mode=_map_sr_mode(sr_mode)` and the forwarded hyperparameters.

Errors

- ValueError: Raised if `_map_sr_mode(sr_mode)` does not recognize the provided mode alias.
- Any exception raised by `int(field_size)`, `int(comms_size)`, or `LinInternalModelA.__init__` may propagate (not specified further).

Example

!!! example "Construct with legacy SR mode alias"
    ```python
    from Q_Sea_Battle.lin_trainable_assisted_model_a import LinTrainableAssistedModelA

    model = LinTrainableAssistedModelA(
        field_size=10,
        comms_size=4,
        sr_mode="sample",  # normalized to "stochastic"
        p_rule=1.0,
        beta=10.0,
        alpha=5.0,
        seed=123,
        hidden_units_measure=64,
        hidden_units_combine=64,
        name="legacy_wrapper",
        unused_legacy_arg="ignored",
    )
    ```

## Public Methods

Not specified in this module beyond `__init__`; all other public behavior is inherited from `LinInternalModelA` (not documented here).

## Data & State

- field_size: int, constraint: set as `int(field_size)`; shape: scalar; stored instance attribute.
- comms_size: int, constraint: set as `int(comms_size)`; shape: scalar; stored instance attribute.
- Inherited state: Not specified here; defined by `LinInternalModelA`.

## Planned (design-spec)

Not specified.

## Deviations

Not specified.

## Notes for Contributors

- This wrapper intentionally ignores `**kwargs` to preserve older call sites; avoid removing this behavior without a migration plan.
- SR mode normalization is implemented via the module-private `_map_sr_mode` helper; update that function if new legacy aliases need to be supported.

## Related

- `Q_Sea_Battle.lin_internal_model_a.LinInternalModelA` (base class; delegated behavior).
- `_map_sr_mode(sr_mode: str) -> str` (module-private helper used by the constructor; not part of the public API).

## Changelog

Not specified.