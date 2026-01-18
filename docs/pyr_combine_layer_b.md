# PyrCombineLayerB

> Role: Keras layer that combines Player B's gun state, shared-randomness outcome, and comm bit to produce the next pyramid-level gun state and updated comm bit.

Location: `Q_Sea_Battle.pyr_combine_layer_b.PyrCombineLayerB`

## Derived constraints

- Let $L$ be the last-dimension length of `gun_batch`; $L$ must be even.
- Let $B$ be the batch size (first dimension).
- `sr_outcome_batch` must have last-dimension length $L/2$.
- `comm_batch` must have last-dimension length $1$.

## Constructor

| Parameter | Type | Description |
| --- | --- | --- |
| name | Optional[str], constraints: None; shape: scalar | Optional Keras layer name. |

Preconditions: None specified.

Postconditions: The layer is created with `trainable=False`.

Errors: Not specified.

!!! example "Example"
    ```python
    import tensorflow as tf
    from Q_Sea_Battle.pyr_combine_layer_b import PyrCombineLayerB

    layer = PyrCombineLayerB(name="pyr_b")
    gun = tf.constant([[1, 0, 0, 1]], dtype=tf.float32)          # shape (B=1, L=4)
    sr = tf.constant([[1, 0]], dtype=tf.float32)                # shape (1, L/2=2)
    comm = tf.constant([[0]], dtype=tf.float32)                 # shape (1, 1)
    next_gun, next_comm = layer(gun, sr, comm)
    ```

## Public Methods

### call

**Signature:** `call(gun_batch, sr_outcome_batch, comm_batch) -> Tuple[tf.Tensor, tf.Tensor]`

Parameters:

- `gun_batch`: tf.Tensor, dtype float32 (converted via `tf.convert_to_tensor`), constraints: values intended in {0,1}; shape (B, L).
- `sr_outcome_batch`: tf.Tensor, dtype float32 (converted via `tf.convert_to_tensor`), constraints: values intended in {0,1}; shape (B, L/2).
- `comm_batch`: tf.Tensor, dtype float32 (converted via `tf.convert_to_tensor`), constraints: values intended in {0,1}; shape (B, 1).

Returns:

- `next_gun`: tf.Tensor, dtype float32, constraints: computed modulo 2; shape (B, L/2).
- `next_comm`: tf.Tensor, dtype float32, constraints: computed modulo 2; shape (B, 1).

Semantics:

- Let `even = gun_batch[..., ::2]` and `odd = gun_batch[..., 1::2]`.
- Computes `next_gun = (even + sr_outcome_batch) mod 2`.
- Computes parity contribution `parity = (sum(odd * sr_outcome_batch) mod 2)` along the last axis, keeping dimension, then `next_comm = (comm_batch + parity) mod 2`.

Preconditions:

- `L = tf.shape(gun_batch)[-1]` is even.
- `tf.shape(sr_outcome_batch)[-1] == tf.shape(gun_batch[..., ::2])[-1]` (i.e., `sr_outcome_batch` last dimension equals `L/2`).
- `tf.shape(comm_batch)[-1] == 1`.

Postconditions:

- Outputs satisfy the return shapes given above.

Errors:

- Raises `tf.errors.InvalidArgumentError` (via `tf.debugging.assert_equal`) if `L` is not even.
- Raises `tf.errors.InvalidArgumentError` (via `tf.debugging.assert_equal`) if `sr_outcome_batch` last dimension is not `L/2`.
- Raises `tf.errors.InvalidArgumentError` (via `tf.debugging.assert_equal`) if `comm_batch` last dimension is not `1`.

## Data & State

- Trainable state: None (`trainable=False`).
- Persistent variables: None specified.
- Runtime-derived tensors: `even`, `odd`, `gated`, `parity` are computed inside `call` and not stored.

## Planned (design-spec)

- Not specified.

## Deviations

- Not specified.

## Notes for Contributors

- The implementation converts all inputs to `tf.float32` and uses `tf.math.floormod(..., 2.0)`; if strict boolean/integer dtypes are desired, this is a potential future refactor but is not specified in the current code.

## Related

- `tf.keras.layers.Layer`
- `tf.debugging.assert_equal`
- `tf.math.floormod`

## Changelog

- 0.1: Initial implementation of Player B pyramid combine logic with even/odd split, XOR-by-mod-2, and gated parity comm update.