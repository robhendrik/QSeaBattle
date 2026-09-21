# LinTrainableAssistedModelB

> Role: Backward-compatible wrapper that preserves a legacy constructor interface while delegating learning behavior to `LinInternalModelB`.

Location: `Q_Sea_Battle.lin_trainable_assisted_model_b.LinTrainableAssistedModelB`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| field_size | int, constraint: convertible via `int(field_size)`; shape: scalar | Game field size used to construct the internal layout; stored as `self.field_size`. |
| comms_size | int, constraint: convertible via `int(comms_size)`; shape: scalar | Communication channel size used to construct the internal layout; stored as `self.comms_size`. |
| sr_mode | str, constraint: must map via `_map_sr_mode` to one of `{"stochastic","replay"}`; shape: scalar | Shared resource mode; legacy aliases accepted: `"sample"` maps to `"stochastic"`, `"expected"` maps to `"replay"`; default `"sample"`. |
| p_rule | float, constraint: not specified; shape: scalar | Probability of applying the rule-based component; forwarded to the internal model; default `1.0`. |
| beta | float, constraint: not specified; shape: scalar | Hyperparameter forwarded to the internal model; default `10.0`. |
| alpha | float, constraint: not specified; shape: scalar | Hyperparameter forwarded to the internal model; default `5.0`. |
| seed | Optional[int], constraint: `None` or int; shape: scalar | Optional RNG seed forwarded to the internal model; default `None`. |
| hidden_units_measure | int, constraint: not specified; shape: scalar | Width of the measurement subnetwork forwarded to the internal model; default `64`. |
| hidden_units_combine | int, constraint: not specified; shape: scalar | Width of the combine subnetwork forwarded to the internal model; default `64`. |
| name | Optional[str], constraint: `None` or str; shape: scalar | Optional model name forwarded to the internal model; default `None`. |
| **kwargs | Any, constraint: accepted but unused by this wrapper; shape: N/A | Extra keyword arguments accepted for backward compatibility; currently unused. |

Preconditions

- `field_size` and `comms_size` are convertible to `int` via `int(...)`.
- `sr_mode` is convertible to `str` and lowercased value is one of: `"sample"`, `"stochastic"`, `"expected"`, `"replay"`.

Postconditions

- `self.field_size` is set to `int(field_size)` and `self.comms_size` is set to `int(comms_size)`.
- The base class initializer `LinInternalModelB.__init__` has been invoked with an ad-hoc `layout` object providing `layout.field_size` and `layout.comms_size`, and with `sr_mode` normalized to `"stochastic"` or `"replay"`.

Errors

- Raises `ValueError` if `sr_mode` is not a supported alias/value (as determined by `_map_sr_mode`).

Example

!!! example "Construct using legacy SR mode alias"
    ```python
    from Q_Sea_Battle.lin_trainable_assisted_model_b import LinTrainableAssistedModelB

    model = LinTrainableAssistedModelB(
        field_size=10,
        comms_size=4,
        sr_mode="sample",  # legacy alias; normalized to "stochastic"
        seed=123,
        name="player_model",
    )
    ```

## Public Methods

Not specified in this module. This class inherits all public methods from `LinInternalModelB`.

## Data & State

- `field_size`: int, constraint: set by `int(field_size)`; shape: scalar; legacy field size retained on the wrapper instance.
- `comms_size`: int, constraint: set by `int(comms_size)`; shape: scalar; legacy comms size retained on the wrapper instance.

Other data/state are inherited from `LinInternalModelB` (not specified in this module).

## Planned (design-spec)

Not specified.

## Deviations

Not specified.

## Notes for Contributors

- This wrapper exists to preserve a historical constructor signature while providing a minimal `layout` object containing only `field_size` and `comms_size`.
- The internal model interface and learning logic are intentionally not duplicated here; changes to learning behavior should be made in `LinInternalModelB`.
- Legacy SR mode normalization is performed by the module-private `_map_sr_mode` helper; if adding new SR modes, ensure aliases remain backward compatible.

## Related

- `Q_Sea_Battle.lin_internal_model_b.LinInternalModelB` (base class; actual implementation target).

## Changelog

- Not specified.