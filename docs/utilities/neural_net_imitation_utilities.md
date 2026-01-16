# Module neural_net_imitation_utilities

**Module import path**: `Q_Sea_Battle.neural_net_imitation_utilities`

Utilities for generating imitation-learning datasets for
`NeuralNetPlayers.model_a` and `NeuralNetPlayers.model_b` based on the
MajorityPlayers teacher strategy.

This module is used to synthesize supervised datasets for training neural
agents to imitate analytically defined majority-based communication and
decision rules.

---

## Overview

This module provides utilities to:

- Partition the flattened game field into communication segments.
- Compute majority communication bits per segment.
- Generate imitation-learning datasets for Model A (field → comm).
- Generate imitation-learning datasets for Model B (field + comm + gun → shoot).
- Generate paired datasets with controlled randomness.

All datasets are generated synthetically and deterministically given the
random seed.

---

## Functions

### make_segments

Compute contiguous segments over the flattened field.

#### Signature

```
make_segments(
    layout: GameLayout,
) -> list[tuple[int, int]]
```

#### Parameters

- **layout**  
  Game layout defining field and communication sizes.  
  Type: `GameLayout`

#### Returns

- List of `(start, end)` index pairs (slice-style, end exclusive).  
  Type: `list[tuple[int, int]]`, length `m = comms_size`

#### Preconditions

- `field_size >= 1`.
- `1 <= comms_size <= n2`.

#### Postconditions

- Segments cover `[0, n2)` without gaps or overlaps.
- Segment lengths differ by at most 1.

#### Errors

- `ValueError` if layout parameters are invalid.
- `RuntimeError` if full coverage is not achieved.

#### Example

```python
segments = make_segments(layout)
```

---

### compute_majority_comm

Compute teacher majority communication bits for a batch of fields.

#### Signature

```
compute_majority_comm(
    fields: np.ndarray,
    layout: GameLayout,
) -> np.ndarray
```

#### Parameters

- **fields**  
  Flattened binary fields.  
  Type: `np.ndarray, dtype int {0,1}, shape (N, n2)`

- **layout**  
  Game layout defining segmentation.  
  Type: `GameLayout`

#### Returns

- Majority communication bits.  
  Type: `np.ndarray, dtype float32 {0.0,1.0}, shape (N, m)`

#### Preconditions

- `fields.ndim == 2`.
- `fields.shape[1] == n2`.

#### Postconditions

- Each output bit equals the majority value of its segment.

#### Errors

- `ValueError` if input shape is inconsistent with `layout`.

#### Example

```python
comms = compute_majority_comm(fields, layout)
```

---

### generate_majority_dataset_model_a

Generate an imitation-learning dataset for Model A (field → comm).

#### Signature

```
generate_majority_dataset_model_a(
    layout: GameLayout,
    num_samples: int,
    p_one: float = 0.5,
    seed: int | None = None,
) -> pd.DataFrame
```

#### Parameters

- **layout**  
  Game layout.  
  Type: `GameLayout`

- **num_samples**  
  Number of samples to generate.  
  Type: `int`

- **p_one**  
  Probability that a field cell equals 1.  
  Type: `float`

- **seed**  
  Optional RNG seed.  
  Type: `int | None`

#### Returns

- Dataset with columns:  
  - `field`: `np.ndarray, dtype float32, shape (n2,)`  
  - `comm`: `np.ndarray, dtype float32, shape (m,)`  
  Type: `pd.DataFrame`

#### Preconditions

- `num_samples > 0`.

#### Postconditions

- Dataset length equals `num_samples`.

#### Errors

- `ValueError` if `num_samples <= 0`.

#### Example

```python
df_a = generate_majority_dataset_model_a(layout, 1024)
```

---

### generate_majority_dataset_model_b

Generate an imitation-learning dataset for Model B
(comm + gun → shoot).

#### Signature

```
generate_majority_dataset_model_b(
    layout: GameLayout,
    num_samples: int,
    p_one: float = 0.5,
    seed: int | None = None,
) -> pd.DataFrame
```

#### Parameters

- **layout**  
  Game layout.  
  Type: `GameLayout`

- **num_samples**  
  Number of samples to generate.  
  Type: `int`

- **p_one**  
  Probability that a field cell equals 1.  
  Type: `float`

- **seed**  
  Optional RNG seed.  
  Type: `int | None`

#### Returns

- Dataset with columns:  
  - `field`: `np.ndarray, dtype float32, shape (n2,)`  
  - `comm`: `np.ndarray, dtype float32, shape (m,)`  
  - `gun`: `np.ndarray, dtype float32, shape (n2,)` (one-hot)  
  - `shoot`: `float32 {0.0,1.0}`  
  Type: `pd.DataFrame`

#### Preconditions

- `num_samples > 0`.

#### Postconditions

- `shoot` equals the majority bit of the segment containing `gun`.

#### Errors

- `ValueError` if `num_samples <= 0`.

#### Example

```python
df_b = generate_majority_dataset_model_b(layout, 1024)
```

---

### generate_majority_imitation_datasets

Generate paired imitation-learning datasets for Model A and Model B.

#### Signature

```
generate_majority_imitation_datasets(
    layout: GameLayout,
    num_samples_a: int,
    num_samples_b: int,
    p_one: float = 0.5,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]
```

#### Parameters

- **layout**  
  Game layout.  
  Type: `GameLayout`

- **num_samples_a**  
  Number of samples for Model A.  
  Type: `int`

- **num_samples_b**  
  Number of samples for Model B.  
  Type: `int`

- **p_one**  
  Probability that a field cell equals 1.  
  Type: `float`

- **seed**  
  Optional RNG seed.  
  Type: `int | None`

#### Returns

- `(dataset_a, dataset_b)`  
  Type: `tuple[pd.DataFrame, pd.DataFrame]`

#### Preconditions

- `num_samples_a > 0`.
- `num_samples_b > 0`.

#### Postconditions

- Datasets are reproducible given the same seed.
- Seeds for A and B differ by 1 if a base seed is provided.

#### Errors

- Propagates errors from underlying generator functions.

#### Example

```python
df_a, df_b = generate_majority_imitation_datasets(layout, 1024, 1024, seed=0)
```

---

## Testing Hooks

Suggested invariants for testing:

- `sum(segment lengths) == n2`.
- Majority comm bits equal manual majority counts on small examples.
- Gun index always maps to exactly one segment.
- Dataset lengths match requested sample counts.

---

## Notes for Contributors

- Keep dataset schemas stable; downstream training code depends on column names.
- Do not introduce TensorFlow dependencies in this module.
- Any change in segmentation logic must be reflected consistently across all generators.

---

## Changelog

- 2026-01-16 — Initial specification page. (Rob Hendriks)
