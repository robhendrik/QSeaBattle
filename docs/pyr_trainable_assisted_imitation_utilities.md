# Q_Sea_Battle.pyr_trainable_assisted_imitation_utilities

> Role: Per-level (level-aware) dataset generation and training/weight-transfer utilities for pyramid trainable-assisted imitation models; produces dict-of-arrays datasets and optional TensorFlow `tf.data.Dataset` wrappers.

Location: `Q_Sea_Battle.pyr_trainable_assisted_imitation_utilities`

## Overview

This module provides utilities to generate supervised training datasets for pyramid-style trainable-assisted models, where each pyramid level maps an input vector of length `L` to outputs of length `L/2` (with `L` a power of two and `L >= 2`). It includes dataset generators for “measurement” and “combine” tasks for model variants A and B, helper conversion to `tf.data.Dataset`, a wrapper for training a Keras layer standalone, and utilities to transfer per-level trained weights back into a multi-level model.

TensorFlow is optional at import time; functions requiring TensorFlow will raise `ModuleNotFoundError` if TensorFlow is not available.

## Public API

### Functions

#### `pyramid_levels`

Signature: `pyramid_levels(field_size: int) -> List[int]`

Purpose: Compute the input sizes per pyramid level for a power-of-two field size.

Arguments:
- `field_size` (`int`): Initial size `N` (must be power of two, and `>= 2`).

Returns:
- `List[int]`: Input sizes per level, e.g. `16 -> [16, 8, 4, 2]`.

Errors:
- `ValueError`: If `field_size < 2`.
- `ValueError`: If `field_size` is not a power of two.

Example:
```python
from Q_Sea_Battle.pyr_trainable_assisted_imitation_utilities import pyramid_levels

levels = pyramid_levels(16)  # [16, 8, 4, 2]
```

#### `generate_measurement_dataset_a`

Signature: `generate_measurement_dataset_a(L: int, num_samples: int, seed: int = 0) -> Dict[str, np.ndarray]`

Purpose: Generate a per-level dataset for “measurement A” targets.

Arguments:
- `L` (`int`): Level input length (must be power of two, and `>= 2`).
- `num_samples` (`int`): Number of samples to generate.
- `seed` (`int`, default: `0`): Random seed.

Returns:
- `Dict[str, np.ndarray]`: Dict-of-arrays with keys:
  - `field`: shape `(num_samples, L)`, dtype `np.float32`, values in `{0.0, 1.0}`.
  - `meas_target`: shape `(num_samples, L//2)`, dtype `np.float32`, values in `{0.0, 1.0}`.

Errors:
- `ValueError`: If `L < 2`.
- `ValueError`: If `L` is not a power of two.

Example:
```python
from Q_Sea_Battle.pyr_trainable_assisted_imitation_utilities import generate_measurement_dataset_a

ds = generate_measurement_dataset_a(L=8, num_samples=1024, seed=1)
# ds["field"].shape == (1024, 8)
# ds["meas_target"].shape == (1024, 4)
```

#### `generate_combine_dataset_a`

Signature: `generate_combine_dataset_a(L: int, num_samples: int, seed: int = 0) -> Dict[str, np.ndarray]`

Purpose: Generate a per-level dataset for “combine A” targets.

Arguments:
- `L` (`int`): Level input length (must be power of two, and `>= 2`).
- `num_samples` (`int`): Number of samples to generate.
- `seed` (`int`, default: `0`): Random seed (also influences SR sampling via `seed + 12345`).

Returns:
- `Dict[str, np.ndarray]`: Dict-of-arrays with keys:
  - `field`: shape `(num_samples, L)`, dtype `np.float32`, values in `{0.0, 1.0}`.
  - `sr_outcome`: shape `(num_samples, L//2)`, dtype `np.float32`, values in `{0.0, 1.0}`.
  - `next_field_target`: shape `(num_samples, L//2)`, dtype `np.float32`, values in `{0.0, 1.0}`.

Errors:
- `ValueError`: If `L < 2`.
- `ValueError`: If `L` is not a power of two.

Example:
```python
from Q_Sea_Battle.pyr_trainable_assisted_imitation_utilities import generate_combine_dataset_a

ds = generate_combine_dataset_a(L=8, num_samples=512, seed=0)
# ds["next_field_target"].shape == (512, 4)
```

#### `generate_measurement_dataset_b`

Signature: `generate_measurement_dataset_b(L: int, num_samples: int, seed: int = 0)`

Purpose: Generate a per-level dataset for “measurement B” targets, using one-hot `gun` vectors to match the game/inference setting.

Arguments:
- `L` (`int`): Level input length. Constraints are not validated in this function (not specified), but other utilities assume `L` is a power of two and `>= 2`.
- `num_samples` (`int`): Number of samples to generate.
- `seed` (`int`, default: `0`): Random seed.

Returns:
- `dict`: Dict-of-arrays with keys:
  - `gun`: shape `(num_samples, L)`, one-hot, dtype `np.float32`.
  - `meas_target`: shape `(num_samples, L//2)`, dtype `np.float32`, values in `{0.0, 1.0}` computed per-sample.

Errors:
- Not specified. (No explicit validation; NumPy may raise on invalid shapes.)

Example:
```python
from Q_Sea_Battle.pyr_trainable_assisted_imitation_utilities import generate_measurement_dataset_b

ds = generate_measurement_dataset_b(L=8, num_samples=128, seed=42)
# ds["gun"].shape == (128, 8)
# ds["meas_target"].shape == (128, 4)
```

#### `generate_combine_dataset_b`

Signature: `generate_combine_dataset_b(L: int, num_samples: int, seed: int = 0)`

Purpose: Generate a per-level dataset for “combine B” targets, using one-hot `gun`, binary SR outcomes, and a binary `comm` bit.

Arguments:
- `L` (`int`): Level input length. Constraints are not validated in this function (not specified), but target shapes assume `L//2` exists.
- `num_samples` (`int`): Number of samples to generate.
- `seed` (`int`, default: `0`): Random seed.

Returns:
- `dict`: Dict-of-arrays with keys:
  - `gun`: shape `(num_samples, L)`, one-hot, dtype `np.float32`.
  - `sr_outcome`: shape `(num_samples, L//2)`, dtype `np.float32`, values in `{0.0, 1.0}`.
  - `comm`: shape `(num_samples, 1)`, dtype `np.float32`, values in `{0.0, 1.0}`.
  - `next_gun_target`: shape `(num_samples, L//2)`, dtype `np.float32`.
  - `next_comm_target`: shape `(num_samples, 1)`, dtype `np.float32`.

Errors:
- Not specified. (No explicit validation; NumPy may raise on invalid shapes.)

Example:
```python
from Q_Sea_Battle.pyr_trainable_assisted_imitation_utilities import generate_combine_dataset_b

ds = generate_combine_dataset_b(L=8, num_samples=256, seed=7)
# ds["next_gun_target"].shape == (256, 4)
# ds["next_comm_target"].shape == (256, 1)
```

#### `to_tf_dataset`

Signature: `to_tf_dataset(ds: Mapping[str, np.ndarray], x_keys: Sequence[str], y_key: str, batch_size: int, shuffle: bool = True, seed: int = 0) -> "tf.data.Dataset"`

Purpose: Convert a dict-of-NumPy-arrays dataset into a `tf.data.Dataset` yielding `(x, y)`, where `x` is either a single tensor or a tuple of tensors.

Arguments:
- `ds` (`Mapping[str, np.ndarray]`): Dict-like dataset of arrays.
- `x_keys` (`Sequence[str]`): Keys from `ds` to use as model inputs.
- `y_key` (`str`): Key from `ds` to use as the target.
- `batch_size` (`int`): Batch size for `.batch()`.
- `shuffle` (`bool`, default: `True`): Whether to shuffle.
- `seed` (`int`, default: `0`): Shuffle seed.

Returns:
- `tf.data.Dataset`: A batched, prefetched dataset. Prefetch uses `tf.data.AUTOTUNE`.

Errors:
- `ModuleNotFoundError`: If TensorFlow is required but not available.
- `KeyError`: If keys are missing in `ds` (raised by dict access).
- Other errors: Not specified (TensorFlow/NumPy may raise for incompatible shapes).

Example:
```python
from Q_Sea_Battle.pyr_trainable_assisted_imitation_utilities import generate_measurement_dataset_a, to_tf_dataset

np_ds = generate_measurement_dataset_a(L=8, num_samples=1024, seed=0)
tf_ds = to_tf_dataset(np_ds, x_keys=["field"], y_key="meas_target", batch_size=32, shuffle=True, seed=0)
```

#### `train_layer`

Signature: `train_layer(layer: Any, ds: "tf.data.Dataset", loss: Any, epochs: int, metrics: Optional[Sequence[Any]] = None, verbose: int = 1) -> "tf.keras.Model"`

Purpose: Train a Keras layer as a standalone model by wrapping it in a `tf.keras.Model` with inferred input signatures from the dataset.

Arguments:
- `layer` (`Any`): A callable Keras layer (or similar) supporting `layer(inp)` or `layer(*inputs)`.
- `ds` (`tf.data.Dataset`): Dataset yielding `(x, y)`, where `x` is a tensor or a tuple/list of tensors.
- `loss` (`Any`): Loss passed to `model.compile(loss=...)`.
- `epochs` (`int`): Number of epochs.
- `metrics` (`Optional[Sequence[Any]]`, default: `None`): Metrics passed to `model.compile(metrics=...)`.
- `verbose` (`int`, default: `1`): Verbosity passed to `model.fit(...)`.

Returns:
- `tf.keras.Model`: The compiled and fitted wrapper model.

Errors:
- `ModuleNotFoundError`: If TensorFlow is required but not available.
- `StopIteration`: If `ds` is empty (when sampling `next(iter(ds.take(1)))`).
- Other errors: Not specified (TensorFlow/Keras may raise for incompatible shapes/types).

Example:
```python
import tensorflow as tf
from Q_Sea_Battle.pyr_trainable_assisted_imitation_utilities import generate_measurement_dataset_a, to_tf_dataset, train_layer

np_ds = generate_measurement_dataset_a(L=8, num_samples=1024, seed=0)
ds = to_tf_dataset(np_ds, x_keys=["field"], y_key="meas_target", batch_size=32)

layer = tf.keras.layers.Dense(4, activation="sigmoid")
model = train_layer(layer, ds=ds, loss="binary_crossentropy", epochs=3, metrics=[tf.keras.metrics.BinaryAccuracy()])
```

#### `transfer_pyr_model_a_layer_weights`

Signature: `transfer_pyr_model_a_layer_weights(model_a: Any, measure_layers_a: Sequence[Any], combine_layers_a: Sequence[Any]) -> None`

Purpose: Copy per-level trained weights into a Model A instance that exposes `measure_layers` and `combine_layers` sequences.

Arguments:
- `model_a` (`Any`): Destination model; must have attributes `measure_layers` and `combine_layers` (both sequences).
- `measure_layers_a` (`Sequence[Any]`): Source per-level measurement layers to copy from.
- `combine_layers_a` (`Sequence[Any]`): Source per-level combine layers to copy from.

Returns:
- `None`

Errors:
- `ValueError`: If `model_a` does not have required attributes.
- `ValueError`: If `measure_layers_a` length does not match `len(model_a.measure_layers)`.
- `ValueError`: If `combine_layers_a` length does not match `len(model_a.combine_layers)`.
- `ValueError`: If `len(model_a.measure_layers) != len(model_a.combine_layers)`.

Example:
```python
from Q_Sea_Battle.pyr_trainable_assisted_imitation_utilities import transfer_pyr_model_a_layer_weights

# model_a must have .measure_layers and .combine_layers, and each element must support get_weights()/set_weights()
transfer_pyr_model_a_layer_weights(model_a, measure_layers_a=trained_meas_layers, combine_layers_a=trained_comb_layers)
```

#### `transfer_pyr_model_b_layer_weights`

Signature: `transfer_pyr_model_b_layer_weights(model_b: Any, measure_layers_b: Sequence[Any], combine_layers_b: Sequence[Any]) -> None`

Purpose: Copy per-level trained weights into a Model B instance that exposes `measure_layers` and `combine_layers` sequences.

Arguments:
- `model_b` (`Any`): Destination model; must have attributes `measure_layers` and `combine_layers` (both sequences).
- `measure_layers_b` (`Sequence[Any]`): Source per-level measurement layers to copy from.
- `combine_layers_b` (`Sequence[Any]`): Source per-level combine layers to copy from.

Returns:
- `None`

Errors:
- `ValueError`: If `model_b` does not have required attributes.
- `ValueError`: If `measure_layers_b` length does not match `len(model_b.measure_layers)`.
- `ValueError`: If `combine_layers_b` length does not match `len(model_b.combine_layers)`.
- `ValueError`: If `len(model_b.measure_layers) != len(model_b.combine_layers)`.

Example:
```python
from Q_Sea_Battle.pyr_trainable_assisted_imitation_utilities import transfer_pyr_model_b_layer_weights

transfer_pyr_model_b_layer_weights(model_b, measure_layers_b=trained_meas_layers, combine_layers_b=trained_comb_layers)
```

### Constants

Not specified.

### Types

#### `ArrayLike`

Definition: `ArrayLike = Union[np.ndarray, "tf.Tensor"]`

Description: Convenience union type representing either a NumPy array or a TensorFlow tensor (as a forward reference).

## Dependencies

- `numpy` (imported as `np`)
- `tensorflow` (optional; imported as `tf` if available; required for `to_tf_dataset` and `train_layer`)
- Standard library: `typing` (`Any`, `Dict`, `List`, `Mapping`, `Optional`, `Sequence`, `Tuple`, `Union`)

## Planned (design-spec)

Not specified.

## Deviations

- Several dataset generators validate that `L` is a power of two (`generate_measurement_dataset_a`, `generate_combine_dataset_a`) via a shared checker, while the B-variant generators do not perform the same validation (constraints are implied by shape math but not enforced).
- TensorFlow is treated as an optional dependency at import time; TensorFlow-required functions raise at call time rather than import time.

## Notes for Contributors

- Keep outputs as `np.float32` with binary values `{0.0, 1.0}` to match existing generators and teachers.
- If adding new TensorFlow-dependent utilities, follow the existing pattern of `_require_tf()` to provide a consistent error when TensorFlow is unavailable.
- Maintain the dict-of-arrays convention so callers can consistently convert to `tf.data.Dataset` using `to_tf_dataset`.

## Related

- `Q_Sea_Battle` package (project context)
- TensorFlow `tf.data.Dataset` and `tf.keras` APIs (used by `to_tf_dataset` and `train_layer`)

## Changelog

- 0.1: Initial version (as stated in module docstring).