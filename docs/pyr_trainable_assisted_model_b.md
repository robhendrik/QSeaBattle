# PyrTrainableAssistedModelB

> Role: Player-B pyramid assisted Keras model that applies per-level measurement, assisted processing, and combination layers to produce a shooting logit.

Location: `Q_Sea_Battle.pyr_trainable_assisted_model_b.PyrTrainableAssistedModelB`

## Derived constraints

- Symbols: n2 = field_size, m = comms_size.
- n2 and m are inferred from `game_layout` via `_infer_n2_and_m(game_layout)`.
- Constraint: m must equal 1; otherwise construction raises `ValueError`.
- Constraint: n2 must be a power of two; enforced via `_validate_power_of_two(n2)` which returns `depth` (number of pyramid levels).

## Constructor

| Parameter | Type | Description |
|---|---|---|
| game_layout | Any, constraints: must be accepted by `_infer_n2_and_m(game_layout)`, shape: not specified | Game layout object used to infer n2 and m. |
| p_high | float, constraints: not specified, shape: scalar | Passed to each `PRAssistedLayer` as `p_high`. |
| sr_mode | str, constraints: not specified, shape: scalar | Passed to each `PRAssistedLayer` as `mode`. Default `"sample"`. |
| measure_layers | Optional[Sequence[tf.keras.layers.Layer]], constraints: if provided then `len(measure_layers) == depth`, shape: sequence length `depth` | Per-level measurement layers; if `None`, defaults to `depth` instances of `PyrMeasurementLayerB()`. |
| combine_layers | Optional[Sequence[tf.keras.layers.Layer]], constraints: if provided then `len(combine_layers) == depth`, shape: sequence length `depth` | Per-level combine layers; if `None`, defaults to `depth` instances of `PyrCombineLayerB()`. |
| name | Optional[str], constraints: Keras model name semantics, shape: scalar | Passed to `tf.keras.Model` constructor. |

Preconditions

- `_infer_n2_and_m(game_layout)` must succeed and return `(n2, m)`.
- m must be 1.
- n2 must be a power of two (as validated by `_validate_power_of_two`).
- If `measure_layers` is provided, it must be a sequence of length `depth`.
- If `combine_layers` is provided, it must be a sequence of length `depth`.

Postconditions

- `self.n2: int` and `self.M: int` are set from `_infer_n2_and_m(game_layout)`.
- `self.depth: int` is set from `_validate_power_of_two(self.n2)`.
- `self.measure_layers: List[tf.keras.layers.Layer]` and `self.combine_layers: List[tf.keras.layers.Layer]` are populated with length `depth`.
- Backward-compat aliases `self.measure_layer` and `self.combine_layer` reference the first element of the corresponding per-level lists.
- `self.sr_layers: List[PRAssistedLayer]` is populated with length `depth`, with decreasing `length` values per level: `n2/2, n2/4, ..., 1`.

Errors

- `ValueError`: if `m != 1` (message indicates `comms_size==1` is required).
- `ValueError`: if `measure_layers` is provided and its length is not `depth`.
- `ValueError`: if `combine_layers` is provided and its length is not `depth`.

Example

!!! example "Instantiate with default per-level layers"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.pyr_trainable_assisted_model_b import PyrTrainableAssistedModelB

    game_layout = ...  # must be accepted by _infer_n2_and_m
    model = PyrTrainableAssistedModelB(game_layout=game_layout, p_high=0.9, sr_mode="sample", name="player_b")

    # Example call signature (see call() for tensor shapes)
    B = 4
    n2 = model.n2
    depth = model.depth
    gun = tf.zeros((B, n2), dtype=tf.float32)
    comm = tf.zeros((B, 1), dtype=tf.float32)
    prev_measurements = [tf.zeros((B, n2 // (2 ** (level + 1))), dtype=tf.float32) for level in range(depth)]
    prev_outcomes = [tf.zeros((B, n2 // (2 ** (level + 1))), dtype=tf.float32) for level in range(depth)]
    y = model([gun, comm, prev_measurements, prev_outcomes], training=False)
    ```

## Public Methods

### call

Signature: `call(self, inputs: list, training: bool = False, **kwargs: Any) -> tf.Tensor`

Parameters

- inputs: list, constraints: must be list/tuple of length 4 in the order `[gun_batch, comm_batch, prev_measurements, prev_outcomes]`, shape: outer length 4.
- training: bool, constraints: not specified, shape: scalar; forwarded to per-level `measure_layer` and `combine_layer` if they accept a `training` keyword argument.
- **kwargs: Any, constraints: accepted but not used in implementation, shape: not applicable.

Returns

- shoot_logit: tf.Tensor, dtype float32, constraints: derived from clipped comm in $[0,1]$ then mapped to logits, shape (B, 1).

Behavior

- Validates `inputs` is a list/tuple of length 4 and unpacks it into `gun_batch`, `comm_batch`, `prev_measurements`, `prev_outcomes`.
- Converts `gun_batch` and `comm_batch` to float32 tensors and validates shapes: `gun` rank 2 with shape (B, n2) and `comm` rank 2 with shape (B, 1).
- Validates `prev_measurements` and `prev_outcomes` are Python lists/tuples with length `depth`.
- Iterates levels `0..depth-1`: computes a per-level measurement `meas_b` from current `state` using `measure_layers[level]`, then runs `PRAssistedLayer` to produce `out_b`, then applies `combine_layers[level]` to update `(state, c)`.
- Clips `c` to $[0,1]$ and maps to logits: `shoot_logit = (c * 2.0 - 1.0) * 10.0`.

Errors

- `ValueError`: if `inputs` is not list/tuple length 4.
- `ValueError`: if `gun_batch` does not convert to a rank-2 tensor.
- `ValueError`: if `comm_batch` does not convert to a rank-2 tensor of trailing dimension 1.
- `TypeError`: if `prev_measurements` or `prev_outcomes` is not a Python list/tuple.
- `ValueError`: if `prev_measurements` or `prev_outcomes` does not have length `depth`.
- `ValueError`: if per-level `prev_out` or `meas_b` is not rank-2.
- `ValueError`: if per-level last-dimension lengths of `meas_b`, `prev_meas`, and `prev_out` do not match.

## Data & State

- n2: int, constraints: inferred from `game_layout`, expected power of two, shape: scalar; field_size.
- M: int, constraints: must equal 1, shape: scalar; comms_size (m).
- depth: int, constraints: returned by `_validate_power_of_two(n2)`, shape: scalar; number of pyramid levels.
- measure_layers: List[tf.keras.layers.Layer], constraints: length `depth`, shape: list length `depth`; per-level measurement layers.
- combine_layers: List[tf.keras.layers.Layer], constraints: length `depth`, shape: list length `depth`; per-level combine layers.
- measure_layer: tf.keras.layers.Layer, constraints: alias of `measure_layers[0]`, shape: scalar reference.
- combine_layer: tf.keras.layers.Layer, constraints: alias of `combine_layers[0]`, shape: scalar reference.
- sr_layers: List[PRAssistedLayer], constraints: length `depth`, shape: list length `depth`; per-level assisted processing layers with decreasing `length`.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- The forward pass attempts to call measurement/combine sublayers with `training=training` and falls back to calling without `training` on `TypeError`; changes to sublayer call signatures should preserve this robustness behavior.
- `**kwargs` is accepted by `call` but unused; if adding behavior, ensure compatibility with Keras calling conventions.

## Related

- `Q_Sea_Battle.pyr_measurement_layer_b.PyrMeasurementLayerB`
- `Q_Sea_Battle.pyr_combine_layer_b.PyrCombineLayerB`
- `Q_Sea_Battle.pr_assisted_layer.PRAssistedLayer`
- `Q_Sea_Battle.pyr_trainable_assisted_model_a._infer_n2_and_m`
- `Q_Sea_Battle.pyr_trainable_assisted_model_a._validate_power_of_two`

## Changelog

- 0.1: Initial version; includes robustness patch to accept and forward the Keras `training` kwarg in `call` where supported by sublayers.