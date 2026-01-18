# LinTrainableAssistedModelA

> Role: Player A linear trainable-assisted baseline model that maps a field tensor to communication logits via measurement, assisted resource processing, and combination layers.

Location: `Q_Sea_Battle.lin_trainable_assisted_model_a.LinTrainableAssistedModelA`

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| field_size | int, constraint: convertible via `int(field_size)`, scalar | Field side length; used to derive $n2 = field\_size \times field\_size$. |
| comms_size | int, constraint: convertible via `int(comms_size)`, scalar | Output communication size; forwarded to `LinCombineLayerA(comms_size=...)`. |
| sr_mode | str, constraint: any string, scalar | Shared resource mode; forwarded to `PRAssistedLayer(mode=...)`; default `"expected"`. |
| seed | int \| None, constraint: any integer or `None`, scalar | Seed forwarded to `PRAssistedLayer(seed=...)`; default `0`. |
| p_high | float, constraint: convertible via `float(p_high)`, scalar | High probability forwarded to `PRAssistedLayer(p_high=...)`; default `0.9`. |
| resource_index | int, constraint: convertible via `int(resource_index)`, scalar | Resource index forwarded to `PRAssistedLayer(resource_index=...)`; default `0`. |
| hidden_units_meas | Sequence[int], constraint: sequence of ints, shape (m,) | Hidden units configuration forwarded to `LinMeasurementLayerA(hidden_units=...)`; default `(64,)`. |
| hidden_units_combine | int \| Sequence[int], constraint: int or sequence of ints, scalar or shape (m,) | Hidden units configuration forwarded to `LinCombineLayerA(hidden_units=...)`; default `(64, 64)`. |
| name | str \| None, constraint: any string or `None`, scalar | Keras model name; if `None`, uses `"LinTrainableAssistedModelA"`. |
| **kwargs | Any, constraint: arbitrary keyword args | Forwarded to `tf.keras.Model.__init__`. |

Preconditions: `field_size` and `comms_size` are provided and convertible to `int`; `field_size` should be meaningful for forming $n2 = field\_size^2$ (not otherwise validated in code); `field_batch` inputs provided later should be convertible to `tf.Tensor` of dtype float32 in `_ensure_batched`.  
Postconditions: Attributes are created: `field_size` (int, scalar), `comms_size` (int, scalar), `n2` (int, scalar), `measurement` (LinMeasurementLayerA), `pr_assisted` (PRAssistedLayer), `combine` (LinCombineLayerA), plus backward-compatible aliases `measure_layer`, `sr_layer`, `combine_layer` pointing to the same objects.  
Errors: Not specified; any exceptions raised by TensorFlow/Keras or the referenced layers’ constructors may propagate.  

!!! example "Example"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.lin_trainable_assisted_model_a import LinTrainableAssistedModelA

    model = LinTrainableAssistedModelA(field_size=5, comms_size=8)
    x = tf.zeros((2, 25), dtype=tf.float32)
    y = model(x, training=False)
    ```

## Public Methods

### _ensure_batched

Signature: `_ensure_batched(x: tf.Tensor) -> Tuple[tf.Tensor, bool]` (static method)

Arguments:
- x: tf.Tensor, dtype: any convertible to float32 via `tf.convert_to_tensor(..., dtype=tf.float32)`, shape (n2,) or (B, n2) where B is batch size

Returns:
- batched_x: tf.Tensor, dtype float32, shape (1, n2) if input rank is 1 else shape (B, n2)
- was_batched: bool, constraint: `False` if input rank is 1 else `True`, scalar

Behavior: Converts `x` to a float32 tensor; if `x.shape.rank == 1`, expands a batch dimension at axis 0 and returns `(expanded, False)`, otherwise returns `(x, True)`.  
Errors: Not specified; conversion/shape rank access errors may propagate from TensorFlow.

### compute_with_internal

Signature: `compute_with_internal(self, field_batch: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, List[tf.Tensor], List[tf.Tensor]]`

Arguments:
- field_batch: tf.Tensor, dtype: any convertible to float32, shape (n2,) or (B, n2)
- training: bool, constraint: boolean, scalar

Returns:
- comm_logits: tf.Tensor, dtype: Not specified (produced by `LinCombineLayerA`), shape (B, comms_size) if the combine layer follows the typical convention (exact shape not specified in this module)
- measurements: List[tf.Tensor], length 1; element 0 is `meas_for_resource`: tf.Tensor, dtype float32, shape (B, n2)
- internals: List[tf.Tensor], length 1; element 0 is `outcomes`: tf.Tensor, dtype: Not specified (produced by `PRAssistedLayer`), shape: Not specified

Behavior: Ensures `field_batch` is batched float32; computes measurement probabilities via `self.measurement(field_batch, training=training)`; if `getattr(self.pr_assisted, "mode", "expected") == "sample"`, binarizes measurements with threshold `>= 0.5` (float32), otherwise uses probabilities as-is; calls `self.pr_assisted` with a dict containing `current_measurement`, zero tensors for `previous_measurement` and `previous_outcome`, and `first_measurement` set to ones of shape (B, 1); combines assisted outcomes via `self.combine(outcomes, training=training)`; returns logits plus internal tensors wrapped in lists.  
Errors: Not specified; any exceptions raised by the underlying layers or TensorFlow ops may propagate.

### call

Signature: `call(self, field_batch: tf.Tensor, training: bool = False) -> tf.Tensor`

Arguments:
- field_batch: tf.Tensor, dtype: any convertible to float32, shape (n2,) or (B, n2)
- training: bool, constraint: boolean, scalar

Returns:
- comm_logits: tf.Tensor, dtype: Not specified (produced by `LinCombineLayerA`), shape: Not specified in this module

Behavior: Delegates to `compute_with_internal(..., training=training)` and returns only `comm_logits`.  
Errors: Not specified; propagates errors from `compute_with_internal`.

## Data & State

- field_size: int, constraint: `int(field_size)` conversion applied, scalar; used to derive `n2`.
- comms_size: int, constraint: `int(comms_size)` conversion applied, scalar.
- n2: int, constraint: equals `field_size * field_size`, scalar; flattened field length.
- measurement: LinMeasurementLayerA, constraint: constructed as `LinMeasurementLayerA(n2=self.n2, hidden_units=hidden_units_meas)`.
- pr_assisted: PRAssistedLayer, constraint: constructed as `PRAssistedLayer(length=self.n2, p_high=float(p_high), mode=str(sr_mode), resource_index=int(resource_index), seed=seed, name="PRAssistedLayerA")`.
- combine: LinCombineLayerA, constraint: constructed as `LinCombineLayerA(comms_size=self.comms_size, hidden_units=hidden_units_combine)`.
- measure_layer: LinMeasurementLayerA, alias of `measurement` (backward-compatible).
- sr_layer: PRAssistedLayer, alias of `pr_assisted` (backward-compatible).
- combine_layer: LinCombineLayerA, alias of `combine` (backward-compatible).

## Planned (design-spec)

Not specified.

## Deviations

Not specified.

## Notes for Contributors

- Keep input handling consistent with `_ensure_batched`: this module assumes rank-1 inputs represent a single flattened field and will be expanded to shape (1, n2).
- `compute_with_internal` uses `getattr(self.pr_assisted, "mode", "expected")` at runtime; changes to `PRAssistedLayer` mode handling can affect the measurement binarization path.

## Related

- `Q_Sea_Battle.lin_measurement_layer_a.LinMeasurementLayerA`
- `Q_Sea_Battle.pr_assisted_layer.PRAssistedLayer`
- `Q_Sea_Battle.lin_combine_layer_a.LinCombineLayerA`

## Changelog

- 0.1: Initial version documented from module text.