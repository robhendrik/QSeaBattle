# Module pyr_trainable_assisted_imitation_utilities

**Module import path**: `Q_Sea_Battle.pyr_trainable_assisted_imitation_utilities`

Per-level dataset generation and training utilities for **pyramid**
trainable-assisted models.

This module provides *level-aware* imitation-learning utilities for the
pyramid architecture, where each level maps an input of length `L` to
outputs of length `L/2`, with `L` a power of two.

All generated data uses `np.float32` with values in `(0.0, 1.0)`.

---

## Overview

This module provides:

- Enumeration of pyramid levels.
- Teacher functions for measurement and combine operations (A and B players).
- Per-level dataset generators for pyramid imitation learning.
- Conversion utilities to `tf.data.Dataset`.
- Lightweight helpers for training individual Keras layers.
- Weight-transfer helpers for assembling full pyramid models.

Training orchestration across levels is intentionally left to notebooks
and scripts.

---

## Functions

### pyramid_levels

Return input sizes per pyramid level.

#### Signature

```
pyramid_levels(
    field_size: int,
) -> list[int]
```

#### Parameters

- **field_size**  
  Initial size `L`, must be a power of two and `>= 2`.  
  Type: `int`

#### Returns

- Input sizes per level.  
  Type: `list[int]`, e.g. `16 -> [16, 8, 4, 2]`

#### Preconditions

- `field_size >= 2`.
- `field_size` is a power of two.

#### Postconditions

- Returned list is strictly decreasing by factor 2.
- Last element equals 2.

#### Errors

- `ValueError` if `field_size` is invalid.

#### Example

```python
levels = pyramid_levels(16)
```

---

### generate_measurement_dataset_a

Generate a dataset for pyramid Measurement-A at a single level.

#### Signature

```
generate_measurement_dataset_a(
    L: int,
    num_samples: int,
    seed: int = 0,
) -> dict[str, np.ndarray]
```

#### Parameters

- **L**  
  Input length at this level (power of two, `>= 2`).  
  Type: `int`

- **num_samples**  
  Number of samples.  
  Type: `int`

- **seed**  
  RNG seed.  
  Type: `int`

#### Returns

- Dictionary with:  
  - `field`: `np.ndarray, dtype float32 (0.0, 1.0), shape (N, L)`  
  - `meas_target`: `np.ndarray, dtype float32 (0.0, 1.0), shape (N, L/2)`

#### Preconditions

- `L` is a power of two.
- `num_samples > 0`.

#### Postconditions

- Targets equal pairwise XOR of input bits.

#### Errors

- `ValueError` if `L` is invalid.

#### Example

```python
ds = generate_measurement_dataset_a(L=8, num_samples=1024)
```

---

### generate_combine_dataset_a

Generate a dataset for pyramid Combine-A at a single level.

#### Signature

```
generate_combine_dataset_a(
    L: int,
    num_samples: int,
    seed: int = 0,
) -> dict[str, np.ndarray]
```

#### Returns

- Dictionary with:  
  - `field`: `np.ndarray, dtype float32, shape (N, L)`  
  - `sr_outcome`: `np.ndarray, dtype float32, shape (N, L/2)`  
  - `next_field_target`: `np.ndarray, dtype float32, shape (N, L/2)`

#### Preconditions

- `L` is a power of two.
- `num_samples > 0`.

#### Postconditions

- Targets equal XOR of even field bits with SR outcome.

---

### generate_measurement_dataset_b

Generate a dataset for pyramid Measurement-B at a single level.

#### Signature

```
generate_measurement_dataset_b(
    L: int,
    num_samples: int,
    seed: int = 0,
) -> dict[str, np.ndarray]
```

#### Returns

- Dictionary with:  
  - `gun`: `np.ndarray, dtype float32 (0.0, 1.0), shape (N, L)` (one-hot)  
  - `meas_target`: `np.ndarray, dtype float32 (0.0, 1.0), shape (N, L/2)`

#### Preconditions

- `L` is a power of two.
- `num_samples > 0`.

#### Postconditions

- `gun` vectors are one-hot.
- Targets follow teacher Measurement-B rule.

---

### generate_combine_dataset_b

Generate a dataset for pyramid Combine-B at a single level.

#### Signature

```
generate_combine_dataset_b(
    L: int,
    num_samples: int,
    seed: int = 0,
) -> dict[str, np.ndarray]
```

#### Returns

- Dictionary with:  
  - `gun`: `np.ndarray, dtype float32, shape (N, L)` (one-hot)  
  - `sr_outcome`: `np.ndarray, dtype float32, shape (N, L/2)`  
  - `comm`: `np.ndarray, dtype float32, shape (N, 1)`  
  - `next_gun_target`: `np.ndarray, dtype float32, shape (N, L/2)`  
  - `next_comm_target`: `np.ndarray, dtype float32, shape (N, 1)`

#### Preconditions

- `L` is a power of two.
- `num_samples > 0`.

#### Postconditions

- `next_gun_target` is one-hot if `gun` is one-hot.
- `next_comm_target` matches teacher Combine-B rule.

---

### to_tf_dataset

Convert a dict-of-arrays dataset to a `tf.data.Dataset`.

#### Signature

```
to_tf_dataset(
    ds: Mapping[str, np.ndarray],
    x_keys: Sequence[str],
    y_key: str,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
) -> tf.data.Dataset
```

#### Preconditions

- TensorFlow must be available.
- `y_key` exists in `ds`.

---

### train_layer

Train a single Keras layer as a standalone model.

#### Signature

```
train_layer(
    layer: Any,
    ds: tf.data.Dataset,
    loss: Any,
    epochs: int,
    metrics: Sequence[Any] | None = None,
    verbose: int = 1,
) -> tf.keras.Model
```

---

### transfer_pyr_model_a_layer_weights

Copy per-level trained weights into Pyramid Model A.

---

### transfer_pyr_model_b_layer_weights

Copy per-level trained weights into Pyramid Model B.

---

## Testing Hooks

Suggested invariants:

- All `L` values are powers of two.
- For one-hot `gun`, `next_gun_target` remains one-hot.
- Layer-wise datasets preserve shapes `(L -> L/2)`.
- Weight transfer preserves parameter counts per layer.

---

## Notes for Contributors

- Do not embed multi-level training loops in this module.
- Teacher rules must remain deterministic and stateless.
- Any change here must be mirrored in pyramid model documentation.

---

## Changelog

- 2026-01-16 — Initial specification page. (Rob Hendriks)
