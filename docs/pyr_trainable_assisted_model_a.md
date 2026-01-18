# PyrTrainableAssistedModelA

> Role: Keras model implementing the Player-A pyramid assisted architecture with per-level measurement/combine layers and per-level shared-resource assisted layers, producing a single communication logit per batch element.

Location: `Q_Sea_Battle.pyr_trainable_assisted_model_a.PyrTrainableAssistedModelA`

## Derived constraints

- $n2$ (the flattened field length) is inferred from `game_layout` as either `int(game_layout.n2)` or `int(game_layout.field_size) * int(game_layout.field_size)`, and must be positive and a power of two.
- $m$ (communication size) is inferred as `int(game_layout.comms_size)` or `int(game_layout.M)` or defaults to `1`, and must satisfy $m = 1$.
- Model depth $K = \log_2(n2)$ (an integer), and the number of per-level measurement layers, combine layers, and shared-resource layers is exactly $K$.
- Input batch `field_batch` must be rank-2 with shape $(B, n2)$; $B$ is batch size.

## Constructor

| Parameter | Type | Description |
|---|---|---|
| `game_layout` | `Any`, GameLayout-like object | Must expose either `n2: int` or `field_size: int` (to derive $n2$), and may expose `comms_size: int` or `M: int` (to derive $m$). Constraints: derived $n2$ must be positive and a power of two; derived $m$ must equal 1. Shape: not applicable. |
| `p_high` | `float`, unconstrained | Correlation parameter forwarded to each `PRAssistedLayer`. |
| `sr_mode` | `str`, unconstrained | Mode forwarded to each `PRAssistedLayer` as `mode`. |
| `measure_layers` | `Optional[Sequence[tf.keras.layers.Layer]]`, length $K$ if provided | Optional per-level measurement layers. Constraint: if not `None`, `len(measure_layers) == K`. |
| `combine_layers` | `Optional[Sequence[tf.keras.layers.Layer]]`, length $K$ if provided | Optional per-level combine layers. Constraint: if not `None`, `len(combine_layers) == K`. |
| `name` | `Optional[str]`, unconstrained | Optional Keras model name. |

Preconditions

- `game_layout` provides enough attributes to infer $n2$ and $m$ as described under Derived constraints.
- Derived $n2$ is a power of two and greater than 0.
- Derived $m == 1$.
- If `measure_layers` is provided, its length equals $K$.
- If `combine_layers` is provided, its length equals $K$.

Postconditions

- `self.n2: int` is set to inferred $n2$.
- `self.M: int` is set to inferred $m$.
- `self.depth: int` is set to $K = \log_2(n2)$.
- `self.measure_layers: List[tf.keras.layers.Layer]` has length $K$; defaults to `PyrMeasurementLayerA()` repeated $K$ times if not provided.
- `self.combine_layers: List[tf.keras.layers.Layer]` has length $K$; defaults to `PyrCombineLayerA()` repeated $K$ times if not provided.
- `self.measure_layer` aliases `self.measure_layers[0]` and `self.combine_layer` aliases `self.combine_layers[0]` (legacy compatibility).
- `self.sr_layers: List[PRAssistedLayer]` has length $K$ with `resource_index` equal to the level index, and `length` equal to $n2 / 2^{(level+1)}$.

Errors

- Raises `ValueError` if derived $m \ne 1`.
- Raises `ValueError` if derived $n2 \le 0`.
- Raises `ValueError` if derived $n2$ is not a power of two.
- Raises `ValueError` if `measure_layers` is provided and `len(measure_layers) != K`.
- Raises `ValueError` if `combine_layers` is provided and `len(combine_layers) != K`.

!!! example "Example"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.pyr_trainable_assisted_model_a import PyrTrainableAssistedModelA

    class Layout:
        field_size = 4
        comms_size = 1

    model = PyrTrainableAssistedModelA(game_layout=Layout(), p_high=0.9, sr_mode="sample")
    x = tf.zeros((2, model.n2), dtype=tf.float32)
    logits = model(x)
    ```

## Public Methods

### call

- Signature: `call(self, field_batch: tf.Tensor) -> tf.Tensor`

Parameters

- `field_batch`: `tf.Tensor`, dtype not specified, shape $(B, n2)$, rank must be 2.

Returns

- `comm_logits`: `tf.Tensor`, dtype float32, shape $(B, 1)$.

Errors

- Propagates `ValueError` from `compute_with_internal` if `field_batch` is not rank-2.

Notes

- Delegates computation to `compute_with_internal` and returns only the first element (communication logits).

### compute_with_internal

- Signature: `compute_with_internal(self, field_batch: tf.Tensor) -> Tuple[tf.Tensor, List[tf.Tensor], List[tf.Tensor]]`

Parameters

- `field_batch`: `tf.Tensor`, dtype not specified, shape $(B, n2)$, rank must be 2.

Returns

- `comm_logits`: `tf.Tensor`, dtype float32, shape $(B, 1)$; computed as `((clip(state,0,1) * 2 - 1) * 10)` after the final level, where final `state` is expected to be shape $(B, 1)$.
- `measurements`: `List[tf.Tensor]`, length $K$, each element `tf.Tensor` dtype float32, shape $(B, L/2)$ for that level (exact $L$ per level is not validated in code).
- `outcomes`: `List[tf.Tensor]`, length $K$, each element `tf.Tensor` dtype float32, shape not specified (depends on `PRAssistedLayer` output).

Errors

- Raises `ValueError` if `field_batch` is not rank-2, i.e., `x.shape.rank != 2`.

## Data & State

- `n2`: `int`, inferred field length; constraint: positive power of two.
- `M`: `int`, inferred communication size; constraint: equals 1.
- `depth`: `int`, $K = \log_2(n2)$.
- `measure_layers`: `List[tf.keras.layers.Layer]`, length $K$; per-level measurement layers.
- `combine_layers`: `List[tf.keras.layers.Layer]`, length $K$; per-level combine layers.
- `measure_layer`: `tf.keras.layers.Layer`, alias to `measure_layers[0]` (legacy compatibility).
- `combine_layer`: `tf.keras.layers.Layer`, alias to `combine_layers[0]` (legacy compatibility).
- `sr_layers`: `List[PRAssistedLayer]`, length $K$; per-level shared-resource assisted layers with `resource_index=level`.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- Input validation in `compute_with_internal` checks only tensor rank, not the second dimension size; mismatched $(B, n2)$ sizes may fail later inside measurement/combine layers.
- The final conversion from the last-level `state` to `comm_logits` is a hard mapping to logits $\{-10, +10\}$ after clipping to $[0,1]$; this is intentionally non-trainable per module docstring.
- The shared-resource layer invocation always passes `first_measurement` as ones and passes zero tensors for previous measurement/outcome, regardless of level.

## Related

- `Q_Sea_Battle.pyr_measurement_layer_a.PyrMeasurementLayerA`
- `Q_Sea_Battle.pyr_combine_layer_a.PyrCombineLayerA`
- `Q_Sea_Battle.pr_assisted_layer.PRAssistedLayer`

## Changelog

- Not specified.