# LinTrainableAssistedModelB

> Role: Player B linear trainable-assisted baseline model that consumes Player A’s previous measurement and PR-assisted outcome tensors to produce a shoot logit.

Location: `Q_Sea_Battle.lin_trainable_assisted_model_b.LinTrainableAssistedModelB`

## Derived constraints

Define `field_size` as the side length of the square field, `comms_size` as the size of the communication vector, and `n2 = field_size * field_size` as the flattened field length.

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| field_size | int, constraint: $field\_size \ge 1$ | Side length of the square field; used to compute `n2 = field_size * field_size`. |
| comms_size | int, constraint: $comms\_size \ge 1$ | Length of the communication vector consumed by the combine layer. |
| sr_mode | str, constraint: unspecified allowed values, default `"expected"` | Mode passed to `PRAssistedLayer` as `mode`. If the resulting `self.pr_assisted.mode == "sample"`, measurements are thresholded to binary via `meas_probs_b >= 0.5`. |
| seed | int \| None, constraint: unspecified, default `0` | Seed passed to `PRAssistedLayer`. |
| p_high | float, constraint: unspecified, default `0.9` | `p_high` passed to `PRAssistedLayer`. |
| resource_index | int, constraint: unspecified, default `0` | `resource_index` passed to `PRAssistedLayer`. |
| hidden_units_meas | Sequence[int], constraint: unspecified, default `(64,)` | Hidden layer configuration passed to `LinMeasurementLayerB(hidden_units=...)`. |
| hidden_units_combine | int \| Sequence[int], constraint: unspecified, default `(64, 64)` | Hidden layer configuration passed to `LinCombineLayerB(hidden_units=...)`. |
| name | str \| None, constraint: unspecified, default `None` | Keras model name; if `None`, uses `"LinTrainableAssistedModelB"`. |
| **kwargs | Any, constraint: unspecified | Forwarded to `tf.keras.Model.__init__`. |

Preconditions: `field_size` and `comms_size` must be convertible to `int`.

Postconditions: Sets `self.field_size: int`, `self.comms_size: int`, `self.n2: int`; constructs `self.measurement: LinMeasurementLayerB`, `self.pr_assisted: PRAssistedLayer`, `self.combine: LinCombineLayerB`; also defines backward-compatible aliases `measure_layer`, `pr_layer`, `resource_layer`, `sr_layer`, `combine_layer` referencing those layers.

Errors: Not specified (aside from potential errors raised by sublayer constructors or invalid type conversions).

!!! example "Example"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.lin_trainable_assisted_model_b import LinTrainableAssistedModelB

    model = LinTrainableAssistedModelB(field_size=5, comms_size=3, sr_mode="expected")

    B = 2
    n2 = 25
    gun = tf.zeros((B, n2), dtype=tf.float32)
    comm = tf.zeros((B, 3), dtype=tf.float32)
    prev_meas_list = [tf.zeros((B, n2), dtype=tf.float32)]
    prev_out_list = [tf.zeros((B, n2), dtype=tf.float32)]

    shoot_logit = model([gun, comm, prev_meas_list, prev_out_list], training=False)
    ```

## Public Methods

### _ensure_batched

Static method: `LinTrainableAssistedModelB._ensure_batched(x) -> (x_batched, already_batched)`

Parameters:

- `x`: tf.Tensor, dtype convertible to float32, shape (n2,) or (B, n2) or any tensor where rank is used to decide batching; rank-1 inputs are expanded on axis 0.

Returns:

- `x_batched`: tf.Tensor, dtype float32, shape (1, N) if input rank is 1, else same shape as input (rank not modified); `N` is the input’s last-dimension size when rank-1.
- `already_batched`: bool, constraint: `False` if input rank is 1 else `True`.

Errors: Not specified.

### compute_with_internal

Method: `compute_with_internal(gun_batch, comm_batch, prev_meas_list, prev_out_list, training=False) -> (shoot_logit, meas_list, out_list)`

Parameters:

- `gun_batch`: tf.Tensor, dtype float32, shape (B, n2) (rank-1 shape (n2,) is accepted and converted to (1, n2)); last dimension must equal `n2`.
- `comm_batch`: tf.Tensor, dtype float32, shape (B, comms_size) (rank-1 shape (comms_size,) is accepted and converted to (1, comms_size)); last dimension must equal `comms_size`.
- `prev_meas_list`: List[tf.Tensor] (or tf.Tensor accepted and normalized to a list), dtype float32, shape list length $\ge 1$ where element 0 has shape (B, n2).
- `prev_out_list`: List[tf.Tensor] (or tf.Tensor accepted and normalized to a list), dtype float32, shape list length $\ge 1$ where element 0 has shape (B, n2).
- `training`: bool, constraint: unspecified, default `False`; forwarded to `self.measurement(...)` and `self.combine(...)`.

Returns:

- `shoot_logit`: tf.Tensor, dtype float32, shape (B, 1) (if combine returns rank-1 (B,), it is expanded to (B, 1)).
- `meas_list`: List[tf.Tensor], constraint: list length 1, containing `meas_for_resource_b`: tf.Tensor, dtype float32, shape (B, n2); if `self.pr_assisted.mode == "sample"`, this tensor is binary-valued in {0.0, 1.0} via thresholding at 0.5, otherwise it equals `meas_probs_b` cast to float32.
- `out_list`: List[tf.Tensor], constraint: list length 1, containing `outcomes_b`: tf.Tensor, dtype not specified by this module, shape (B, n2) as produced by `self.pr_assisted(...)`.

Errors:

- Raises `ValueError` if `comm_batch` is not rank-2 after batching normalization, or if its last dimension is not `comms_size`.
- Raises `ValueError` if `gun_batch` is not rank-2 after batching normalization, or if its last dimension is not `n2`.
- Raises `ValueError` if `prev_meas_list` or `prev_out_list` is empty after normalization to list/tuple.
- Raises `ValueError` if `prev_meas` or `prev_out` is not rank-2, or if their last dimensions are not `n2`.
- Raises `ValueError` if `meas_for_resource_b` is not rank-2 or its last dimension is not `n2`.

!!! note "PR-assisted invocation contract"
    This method calls `self.pr_assisted` with a dict containing keys `"current_measurement"`, `"previous_measurement"`, `"previous_outcome"`, and `"first_measurement"` where `"first_measurement"` is a zeros tensor of shape (B, 1) and dtype float32.

### call

Method: `call(inputs, training=False, **kwargs) -> shoot_logit`

Parameters:

- `inputs`: list, constraint: must be a list/tuple of length 4 in the order `[gun_batch, comm_batch, prev_meas_list, prev_out_list]`; element constraints are as in `compute_with_internal`.
- `training`: bool, constraint: unspecified, default `False`; forwarded to `compute_with_internal`.
- `**kwargs`: Any, constraint: accepted but not used by this implementation.

Returns:

- `shoot_logit`: tf.Tensor, dtype float32, shape (B, 1).

Errors:

- Raises `ValueError` if `inputs` is not a list/tuple of length 4.
- Propagates `ValueError` from `compute_with_internal` for shape/rank violations.

## Data & State

Instance attributes created by the constructor:

- `field_size`: int, constraint: $field\_size \ge 1$, scalar.
- `comms_size`: int, constraint: $comms\_size \ge 1$, scalar.
- `n2`: int, constraint: $n2 = field\_size * field\_size$, scalar.
- `measurement`: LinMeasurementLayerB, constraint: constructed with `n2=self.n2` and `hidden_units=hidden_units_meas`.
- `pr_assisted`: PRAssistedLayer, constraint: constructed with `length=self.n2`, `p_high`, `mode=sr_mode`, `resource_index`, `seed`, and `name="PRAssistedLayerB"`.
- `combine`: LinCombineLayerB, constraint: constructed with `comms_size=self.comms_size` and `hidden_units=hidden_units_combine`.
- Backward-compatible aliases: `measure_layer`, `pr_layer`, `resource_layer`, `sr_layer`, `combine_layer` referencing the corresponding primary layer attributes.

## Planned (design-spec)

Not specified.

## Deviations

Not specified.

## Notes for Contributors

- `compute_with_internal` attempts two calling conventions for `LinCombineLayerB`: first positional `(outcomes_b, comm_batch, training=...)`, then a dict `{"outcomes": outcomes_b, "comm": comm_batch}` if a `TypeError` occurs; maintain compatibility if evolving `LinCombineLayerB`.
- The `"first_measurement"` flag provided to `PRAssistedLayer` is always zeros with shape (B, 1); any changes to the PR-assisted contract should account for this fixed behavior.
- The `call` signature accepts `**kwargs` but does not use it; adding usage should be done carefully to preserve Keras compatibility.

## Related

- `Q_Sea_Battle.lin_measurement_layer_b.LinMeasurementLayerB`
- `Q_Sea_Battle.lin_combine_layer_b.LinCombineLayerB`
- `Q_Sea_Battle.pr_assisted_layer.PRAssistedLayer`

## Changelog

- 0.1: Initial documented version from module header.