# lin_trainable_assisted_imitation_utilities
> Role: Generate reproducible synthetic supervised imitation datasets (parity-prototype targets), convert them to `tf.data.Dataset`, and provide minimal Keras weight-transfer + single-layer training helpers for linear trainable assisted layers in QSeaBattle.
Location: `Q_Sea_Battle.lin_trainable_assisted_imitation_utilities`

## Overview

This module provides utilities to imitation-train linear assisted layers by generating synthetic supervised datasets with parity-based targets, converting datasets into TensorFlow input pipelines, and transferring trained weights into other layer/model instances. Generated datasets are NumPy array dictionaries with leading dimension `num_samples`; arrays are float32 with values in `{0.0, 1.0}` for TensorFlow friendliness. All generators are reproducible via an explicit RNG seed.

Terminology used by the module: SR (shared resource) is a pre-shared auxiliary resource available without communication; `PRAssistedLayer` is referenced as a specific SR type (not defined in this module).

## Public API

### Functions

#### `_rng(seed: Optional[int]) -> np.random.Generator`

**Purpose:** Create a reproducible NumPy random generator from an optional seed.

**Arguments:**
- `seed`: Optional RNG seed.

**Returns:**
- A `np.random.Generator` instance.

**Errors:**
- Not specified.

**Example:**
```python
from Q_Sea_Battle.lin_trainable_assisted_imitation_utilities import _rng

r = _rng(123)
x = r.integers(0, 10, size=(3,))
```

#### `_n2_from_layout(layout: Any) -> int`

**Purpose:** Compute `n²` (field size squared) from a layout-like object or raw integer.

**Arguments:**
- `layout`: Either an object with attribute `field_size`, or a raw integer interpreted as `field_size`.

**Returns:**
- `n2`: An integer equal to `field_size * field_size`.

**Errors:**
- Not specified.

**Example:**
```python
from Q_Sea_Battle.lin_trainable_assisted_imitation_utilities import _n2_from_layout

n2 = _n2_from_layout(5)  # 25
```

#### `_m_from_layout(layout: Any) -> int`

**Purpose:** Extract `m` (communication bits size) from a layout-like object, defaulting to `1` if unspecified.

**Arguments:**
- `layout`: Layout-like object; if it has `comms_size`, that value is used.

**Returns:**
- `m`: Integer comms size.

**Errors:**
- Not specified.

**Example:**
```python
from types import SimpleNamespace
from Q_Sea_Battle.lin_trainable_assisted_imitation_utilities import _m_from_layout

layout = SimpleNamespace(comms_size=3)
m = _m_from_layout(layout)  # 3
```

#### `_as_float01(x: np.ndarray) -> np.ndarray`

**Purpose:** Ensure an array is `float32` (intended to represent values in `{0.0, 1.0}`).

**Arguments:**
- `x`: NumPy array.

**Returns:**
- A NumPy array cast to `np.float32` (or returned unchanged if already `float32`).

**Errors:**
- Not specified.

**Example:**
```python
import numpy as np
from Q_Sea_Battle.lin_trainable_assisted_imitation_utilities import _as_float01

x = np.array([0, 1, 1], dtype=np.int64)
y = _as_float01(x)  # float32
```

#### `_parity_bits(x01: np.ndarray) -> np.ndarray`

**Signature:** `_parity_bits(x01: np.ndarray) -> np.ndarray`

**Purpose:** Compute parity (XOR reduction) over the last dimension.

**Arguments:**
- `x01`: Array with last dimension `N`, values intended in `{0, 1}` (int/bool/float accepted; floats are thresholded at `> 0.5`).

**Returns:**
- `parity`: Array with shape `x01.shape[:-1]`, dtype `int64`, values in `{0, 1}`.

**Errors:**
- Not specified.

**Example:**
```python
import numpy as np
from Q_Sea_Battle.lin_trainable_assisted_imitation_utilities import _parity_bits

x = np.array([[0, 1, 1], [1, 1, 1]], dtype=np.float32)
p = _parity_bits(x)  # array([0, 1])
```

#### `_layer_is_built(layer: tf.keras.layers.Layer) -> bool`

**Purpose:** Check whether a Keras layer is built and has weights.

**Arguments:**
- `layer`: A `tf.keras.layers.Layer`.

**Returns:**
- `True` if `layer.built` is truthy and `layer.weights` is not `None`, else `False`.

**Errors:**
- Not specified.

**Example:**
```python
import tensorflow as tf
from Q_Sea_Battle.lin_trainable_assisted_imitation_utilities import _layer_is_built

layer = tf.keras.layers.Dense(4)
built = _layer_is_built(layer)  # likely False until first call/build
```

#### `generate_measurement_dataset_a(layout: Any, num_samples: int, p_one: float = 0.5, seed: Optional[int] = None) -> ArrayDict`

**Purpose:** Generate supervised pairs for `LinMeasurementLayerA` where the target equals the field (`meas_target == field`).

**Arguments:**
- `layout`: Layout-like object or int; used to determine `n2 = field_size²`.
- `num_samples`: Number of samples to generate.
- `p_one`: Probability of a `1` in each field bit (Bernoulli/binomial).
- `seed`: Optional RNG seed for reproducibility.

**Returns:**
- A dict with keys:
  - `"field"`: shape `(num_samples, n2)`, float32 in `{0.0, 1.0}`
  - `"meas_target"`: shape `(num_samples, n2)`, identical copy of `"field"`

**Errors:**
- Not specified.

**Example:**
```python
from Q_Sea_Battle.lin_trainable_assisted_imitation_utilities import generate_measurement_dataset_a

ds = generate_measurement_dataset_a(layout=5, num_samples=128, p_one=0.3, seed=1)
field = ds["field"]
target = ds["meas_target"]
```

#### `generate_measurement_dataset_b(layout: Any, num_samples: int, seed: Optional[int] = None) -> ArrayDict`

**Purpose:** Generate supervised pairs for `LinMeasurementLayerB` where the input gun vector is one-hot and the target equals the gun (`meas_target == gun`).

**Arguments:**
- `layout`: Layout-like object or int; used to determine `n2 = field_size²`.
- `num_samples`: Number of samples to generate.
- `seed`: Optional RNG seed for reproducibility.

**Returns:**
- A dict with keys:
  - `"gun"`: shape `(num_samples, n2)`, one-hot float32 vectors
  - `"meas_target"`: shape `(num_samples, n2)`, identical copy of `"gun"`

**Errors:**
- Not specified.

**Example:**
```python
from Q_Sea_Battle.lin_trainable_assisted_imitation_utilities import generate_measurement_dataset_b

ds = generate_measurement_dataset_b(layout=5, num_samples=64, seed=42)
gun = ds["gun"]
target = ds["meas_target"]
```

#### `generate_combine_dataset_a(layout: Any, num_samples: int, seed: Optional[int] = None) -> ArrayDict`

**Purpose:** Generate supervised pairs for `LinCombineLayerA` where `comm_target` is the parity of `outcomes_a`, replicated to `m` bits.

**Arguments:**
- `layout`: Layout-like object or int; used to determine `n2 = field_size²` and `m = comms_size` (default `1`).
- `num_samples`: Number of samples to generate.
- `seed`: Optional RNG seed for reproducibility.

**Returns:**
- A dict with keys:
  - `"outcomes_a"`: shape `(num_samples, n2)`, random float32 in `{0.0, 1.0}`
  - `"comm_target"`: shape `(num_samples, m)`, float32 where each row is `parity(outcomes_a[row])` replicated `m` times

**Errors:**
- Not specified.

**Example:**
```python
from types import SimpleNamespace
from Q_Sea_Battle.lin_trainable_assisted_imitation_utilities import generate_combine_dataset_a

layout = SimpleNamespace(field_size=5, comms_size=3)
ds = generate_combine_dataset_a(layout=layout, num_samples=256, seed=7)
```

#### `generate_combine_dataset_b(layout: Any, num_samples: int, seed: Optional[int] = None) -> ArrayDict`

**Purpose:** Generate supervised triples for `LinCombineLayerB` where `shoot_target` is `parity(outcomes_b) XOR parity(comm)`.

**Arguments:**
- `layout`: Layout-like object or int; used to determine `n2 = field_size²` and `m = comms_size` (default `1`).
- `num_samples`: Number of samples to generate.
- `seed`: Optional RNG seed for reproducibility.

**Returns:**
- A dict with keys:
  - `"outcomes_b"`: shape `(num_samples, n2)`, random float32 in `{0.0, 1.0}`
  - `"comm"`: shape `(num_samples, m)`, random float32 in `{0.0, 1.0}`
  - `"shoot_target"`: shape `(num_samples, 1)`, float32 in `{0.0, 1.0}`

**Errors:**
- Not specified.

**Example:**
```python
from types import SimpleNamespace
from Q_Sea_Battle.lin_trainable_assisted_imitation_utilities import generate_combine_dataset_b

layout = SimpleNamespace(field_size=5, comms_size=2)
ds = generate_combine_dataset_b(layout=layout, num_samples=128, seed=99)
```

#### `to_tf_dataset(dataset: Union[ArrayDict, Sequence[Mapping[str, Any]]], x_keys: Sequence[str], y_key: str, batch_size: int = 32, shuffle: bool = True, seed: Optional[int] = None) -> tf.data.Dataset`

**Purpose:** Convert a generated dataset (dict-of-arrays or list-of-rows) into a batched `tf.data.Dataset` yielding `(x, y)` for Keras training.

**Arguments:**
- `dataset`: Either a mapping of arrays with leading dimension `N`, or a sequence of row mappings (e.g. `list[dict]`).
- `x_keys`: Keys used as model inputs; if length is `1`, `x` is a single tensor; if length > `1`, `x` is a tuple of tensors in this same order.
- `y_key`: Key used as the target tensor.
- `batch_size`: Batch size used by `.batch(...)`.
- `shuffle`: Whether to shuffle before batching.
- `seed`: Shuffle seed for reproducible ordering.

**Returns:**
- A `tf.data.Dataset` yielding `(x, y)` batches, with tensors cast to float32 via `_as_float01`.

**Errors:**
- `ImportError`: If TensorFlow is required but unavailable (guard present as `if tf is None`).
- `ValueError`: If dict-of-arrays has inconsistent leading dimensions.
- `ValueError`: If sequence-of-rows is empty.

**Example:**
```python
from Q_Sea_Battle.lin_trainable_assisted_imitation_utilities import generate_combine_dataset_b, to_tf_dataset

np_ds = generate_combine_dataset_b(layout=5, num_samples=1000, seed=0)
tf_ds = to_tf_dataset(np_ds, x_keys=["outcomes_b", "comm"], y_key="shoot_target", batch_size=64, shuffle=True, seed=0)
```

#### `transfer_layer_weights(source_layer: tf.keras.layers.Layer, target_layer: tf.keras.layers.Layer) -> None`

**Purpose:** Copy trained weights from one built Keras layer into another, with explicit validation and clear errors on mismatch.

**Arguments:**
- `source_layer`: Built layer providing weights.
- `target_layer`: Built layer receiving weights.

**Returns:**
- `None`.

**Errors:**
- `ValueError`: If either layer is not built or has no weights.
- `ValueError`: If number of weights differs.
- `ValueError`: If any corresponding weight shapes differ (error message includes index and weight names when available).

**Example:**
```python
import tensorflow as tf
from Q_Sea_Battle.lin_trainable_assisted_imitation_utilities import transfer_layer_weights

src = tf.keras.layers.Dense(8)
tgt = tf.keras.layers.Dense(8)
_ = src(tf.zeros((1, 4)))
_ = tgt(tf.zeros((1, 4)))
transfer_layer_weights(src, tgt)
```

#### `transfer_assisted_model_a_layer_weights(trained_measure_layer: tf.keras.layers.Layer, trained_combine_layer: tf.keras.layers.Layer, model_a: Any) -> None`

**Purpose:** Copy trained measurement and combine layer weights into an object representing a full `LinTrainableAssistedModelA`-like model.

**Arguments:**
- `trained_measure_layer`: Trained measurement layer to copy from.
- `trained_combine_layer`: Trained combine layer to copy from.
- `model_a`: Target model object expected to have `measure_layer` and `combine_layer` attributes.

**Returns:**
- `None`.

**Errors:**
- `AttributeError`: If `model_a` lacks `measure_layer` or `combine_layer`.
- Propagates `ValueError` from `transfer_layer_weights` for build/shape mismatches.

**Example:**
```python
from Q_Sea_Battle.lin_trainable_assisted_imitation_utilities import transfer_assisted_model_a_layer_weights

# model_a must expose .measure_layer and .combine_layer; exact class is not specified in this module.
transfer_assisted_model_a_layer_weights(trained_measure_layer, trained_combine_layer, model_a)
```

#### `transfer_assisted_model_b_layer_weights(trained_measure_layer: tf.keras.layers.Layer, trained_combine_layer: tf.keras.layers.Layer, model_b: Any) -> None`

**Purpose:** Symmetric helper to copy trained measurement and combine layer weights into a `LinTrainableAssistedModelB`-like object.

**Arguments:**
- `trained_measure_layer`: Trained measurement layer to copy from.
- `trained_combine_layer`: Trained combine layer to copy from.
- `model_b`: Target model object expected to have `measure_layer` and `combine_layer` attributes.

**Returns:**
- `None`.

**Errors:**
- `AttributeError`: If `model_b` lacks `measure_layer` or `combine_layer`.
- Propagates `ValueError` from `transfer_layer_weights` for build/shape mismatches.

**Example:**
```python
from Q_Sea_Battle.lin_trainable_assisted_imitation_utilities import transfer_assisted_model_b_layer_weights

transfer_assisted_model_b_layer_weights(trained_measure_layer, trained_combine_layer, model_b)
```

#### `train_layer(layer, ds, loss, epochs: int, metrics=None)`

**Signature:** `train_layer(layer, ds, loss, epochs: int, metrics=None)`

**Purpose:** Train a single Keras layer (or callable) via supervised imitation by wrapping it into a minimal `tf.keras.Model` and running `model.fit(...)`.

**Arguments:**
- `layer`: `tf.keras.layers.Layer` or callable; must accept either a single tensor or multiple tensor inputs.
- `ds`: `tf.data.Dataset` yielding `(x, y)`, where `x` is a tensor or tuple/list of tensors.
- `loss`: Keras loss object or string.
- `epochs`: Number of training epochs.
- `metrics`: Optional list of Keras metrics; defaults to empty list.

**Returns:**
- A compiled and trained `tf.keras.Model` wrapping the provided `layer`.

**Errors:**
- Not specified; underlying TensorFlow/Keras errors may occur (e.g., incompatible shapes, invalid loss).

**Example:**
```python
import tensorflow as tf
from Q_Sea_Battle.lin_trainable_assisted_imitation_utilities import generate_measurement_dataset_a, to_tf_dataset, train_layer

np_ds = generate_measurement_dataset_a(layout=5, num_samples=512, seed=0)
ds = to_tf_dataset(np_ds, x_keys=["field"], y_key="meas_target", batch_size=32, shuffle=True, seed=0)

layer = tf.keras.layers.Dense(25, activation="sigmoid")
model = train_layer(layer=layer, ds=ds, loss="binary_crossentropy", epochs=3, metrics=[tf.keras.metrics.BinaryAccuracy()])
```

### Constants

- None.

### Types

- `ArrayDict`: Type alias for `Dict[str, np.ndarray]`.

## Dependencies

- `numpy` (`np`): random generation, array creation, parity computation, shape checks.
- `tensorflow` (`tf`): `tf.data.Dataset`, tensor conversion, Keras layers/models/training.
- `dataclasses.asdict`, `dataclasses.is_dataclass`: Imported but not used in this module.
- `typing` utilities: `Any`, `Dict`, `Iterable`, `List`, `Mapping`, `MutableMapping`, `Optional`, `Sequence`, `Tuple`, `Union` (only a subset is used).

## Planned (design-spec)

- Unknown (no design notes provided).

## Deviations

- `to_tf_dataset` contains a guard `if tf is None`, but `tensorflow` is imported unconditionally as `tf` in this module; behavior when TensorFlow is missing is not specified beyond the raised `ImportError`.
- `asdict` and `is_dataclass` are imported but unused.
- `_as_float01` enforces `float32` but does not clamp/validate values beyond dtype conversion; values are assumed to be in `{0.0, 1.0}` by construction/caller.

## Notes for Contributors

- Keep dataset generators reproducible by routing randomness through `_rng(seed)` and ensuring all returned arrays use leading dimension `(num_samples, ...)` with `np.float32`.
- When adding new dataset generators, preserve the dict-of-arrays contract so `to_tf_dataset` can consume them.
- Weight transfer helpers require layers to be built; ensure callers build layers (e.g., via a forward pass) before calling `transfer_layer_weights`.
- `train_layer` infers input structure from the first batch of `ds`; ensure datasets yield consistent structures and fully-defined non-batch shapes.

## Related

- Referenced (not defined here): `LinMeasurementLayerA`, `LinMeasurementLayerB`, `LinCombineLayerA`, `LinCombineLayerB`, `LinTrainableAssistedModelA`, `LinTrainableAssistedModelB`, `PRAssistedLayer`.

## Changelog

- 0.1: Initial version (as stated in module docstring).