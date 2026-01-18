# Algorithms

## Purpose

This page defines the **normative** algorithms for QSeaBattle player families and the shared resource (SR) interface. The goal is to make the reference behavior independently re-implementable without reading Python source.

!!! note
If any behavior in code contradicts this page, the code is considered incorrect.

## Scope

This page specifies:

* Core game-level semantics needed by algorithms.
* Deterministic classical baselines (Simple, Majority).
* Assisted algorithms using **SR (shared resource)**.
* Trainable assisted algorithms (Lin and Pyr) as constrained implementations of the same information flow.
* SR semantics and sampling modes.

This page does not specify:

* Training procedures (see Training pages).
* Implementation details of TensorFlow/Keras layers.
* Logging, tournaments, or visualization.

## Notation and symbols

* `field_size`: integer $n \ge 1$.
* `n2`: integer $n^2$.
* `comms_size`: integer $m$ with $1 \le m \le n2$.
* `field`: binary vector of length `n2`, flattened in a fixed order.
* `gun`: one-hot vector of length `n2` indicating the queried cell index.
* `comm`: communicated message from Player A to Player B, binary vector of length `m` (Lin) or a single bit (Pyr).
* `sr`: shared resource value(s) available to both players without signaling.

Shapes used throughout:

* `field`: `np.ndarray, dtype int {0,1}, shape (n2,)` or `tf.Tensor, dtype float32, shape (B, n2)`.
* `gun`: `np.ndarray, dtype int {0,1}, shape (n2,)` one-hot or `tf.Tensor, dtype float32, shape (B, n2)` one-hot.
* `comm`: `np.ndarray, dtype int {0,1}, shape (m,)` or `tf.Tensor, dtype float32, shape (B, m)`.
* `shoot`: scalar decision bit `np.ndarray, dtype int {0,1}, shape (1,)` or `tf.Tensor, dtype float32, shape (B, 1)`.

!!! warning
All math symbols MUST be interpreted using the variable names above. Do not mix different meanings of `m`, `n2`, `field_size`.

## Shared resource (SR)

### Definition

**SR (shared resource)** is any pre-established auxiliary resource accessible to both Player A and Player B without communication and without signaling. SR may be classical, post-quantum, or simulated.

`PRAssistedLayer` is one concrete SR mechanism that provides structured correlations.

### SR interface contract

SR MUST satisfy:

* **No signaling**: Player B MUST NOT obtain information about `field` except via `comm` and SR that is independent of `field` given the chosen SR mode.
* **Symmetry**: Player A and Player B MUST interpret SR values in the same indexing convention.
* **Mode control**: Any `sr_mode` parameter MUST be defined as a choice of shared resource mechanism, not "randomness".

SR MAY be used in one of two modes:

* **Expected-mode**: SR outcomes are replaced by their expectation under the SR distribution (deterministic, differentiable).
* **Sample-mode**: SR outcomes are sampled (stochastic).

Preconditions:

* SR configuration is fixed before a game begins.
* SR does not depend on runtime observations (`field`, `gun`) except through allowed conditional selection rules described below.

Postconditions:

* SR usage preserves the information flow constraints described in Invariants.

Errors:

* Any SR mechanism that can encode `field` into SR outcomes is invalid.

## Core game semantics used by all algorithms

### Game inputs and outputs

For each game instance:

* Player A receives `field`.
* Player B receives `gun`.
* Player A sends `comm` to Player B.
* Player B outputs `shoot` as a guess of the battlefield value at the gun index.

Success condition (conceptual):

* Let `k = argmax(gun)` for one-hot `gun`.
* The game is won if `shoot == field[k]`.

!!! note
Algorithms below specify how `comm` and `shoot` are computed; the environment defines win/loss bookkeeping.

## Deterministic baseline algorithms

### Simple (coverage) strategy

Intent: communicate `m` cells directly and guess randomly outside coverage.

Algorithm:

* Partition indices into a fixed agreed set `C` of size `m` and its complement.
* Player A sets `comm[j] = field[C[j]]` for $j = 0..m-1$.
* Player B computes `k = argmax(gun)`.
* If `k` is in `C`, Player B outputs the matching communicated bit.
* Else Player B outputs `0` or `1` using a fixed baseline rule (for example, always `0`, or a fixed prior).

Preconditions:

* `1 <= m <= n2`.
* Both players share the same ordered index set `C`.

Postconditions:

* If `k` is in coverage, `shoot` matches `field[k]` in noiseless communication.

Errors:

* `ValueError` if `m` invalid or coverage mapping inconsistent.

### Majority (segment-majority) strategy

Intent: communicate one bit per segment of the field.

Algorithm:

* Partition the flattened field indices into `m` contiguous segments of equal length `L = n2 / m` (requires `m | n2`).
* Player A computes, for each segment, the majority bit (ties map to `1`).
* Player A sends these `m` bits as `comm`.
* Player B computes `k = argmax(gun)` and determines which segment contains `k`.
* Player B outputs the corresponding segment bit.

Preconditions:

* `field_size >= 1`.
* `1 <= m <= n2`.
* `m | n2`.

Postconditions:

* `comm` is deterministic given `field`.
* `shoot` depends only on `comm` and `gun`.

Errors:

* `ValueError` if `m` does not divide `n2`.

## Assisted algorithms with SR

This section specifies algorithms that may outperform purely classical deterministic baselines by using SR correlations while respecting no-signaling constraints.

### Assisted (Lin) algorithm family

This family uses `m`-bit communication with linear-style primitives.

#### Lin teacher primitives

Teacher layers define reference transformations:

* Measurement-A: produces `meas_a` from `field`.
* Combine-A: produces `comm` from `meas_a` and SR outcomes.
* Measurement-B: produces `meas_b` from `gun`.
* Combine-B: produces `shoot` from `meas_b`, SR outcomes, and `comm`.

Types and shapes (batch form):

* `field`: `tf.Tensor, dtype float32, shape (B, n2)`.
* `gun`: `tf.Tensor, dtype float32, shape (B, n2)` one-hot.
* `meas_a`: `tf.Tensor, dtype float32, shape (B, n2)` (Lin design default).
* `meas_b`: `tf.Tensor, dtype float32, shape (B, n2)` (Lin design default).
* `comm`: `tf.Tensor, dtype float32, shape (B, m)`.
* `shoot`: `tf.Tensor, dtype float32, shape (B, 1)`.

!!! note
Lin exact measurement/combine semantics depend on the selected Lin reference strategy (for example, parity prototype). The teacher primitives MUST be the single source of truth for these semantics.

#### Lin end-to-end reference flow

Algorithm (single sample):

* Player A:

  * Compute `meas_a = MeasureA(field)`.
  * Obtain SR outcomes required by Combine-A according to `sr_mode`.
  * Compute `comm = CombineA(meas_a, sr_outcomes_a)`.
* Player B:

  * Compute `meas_b = MeasureB(gun)`.
  * Obtain SR outcomes required by Combine-B according to `sr_mode`.
  * Compute `shoot = CombineB(meas_b, sr_outcomes_b, comm)`.

Preconditions:

* `field_size >= 1`.
* `1 <= m <= n2`.
* SR configuration is compatible with the Combine rules.

Postconditions:

* `comm` depends on `field` only through MeasureA and CombineA.
* `shoot` depends on `field` only through `comm` and SR outcomes.

Errors:

* `ValueError` on shape mismatch or invalid SR configuration.

### Assisted (Pyr) algorithm family

This family uses a pyramid reduction structure and typically enforces `comms_size = 1`.

#### Pyramid structural constraints

* `n2 = field_size^2` MUST be a power of two.
* `comms_size` MUST equal `1`.
* The pyramid has `K = log2(n2)` levels.
* Level $l$ has active length $L_l = n2 / 2^l$ for $l = 0..K-1$.
* Each level maps length $L$ to $L/2$.

Preconditions:

* `field_size >= 1`.
* `n2` is a power of two.
* `m = 1`.

Errors:

* `ValueError` if `n2` not power of two or `m != 1`.

#### Pyr teacher primitives

At each level:

* Measurement-A: `meas_a_l = MeasureA_l(field_l)` where `field_l` has length $L$ and output has length $L/2$.
* Combine-A: `field_{l+1} = CombineA_l(field_l, sr_outcome_l)` output length $L/2$.
* Measurement-B: `meas_b_l = MeasureB_l(gun_l)` input length $L$, output length $L/2$.
* Combine-B: `(gun_{l+1}, comm_{l+1}) = CombineB_l(gun_l, sr_outcome_l, comm_l)` where `comm_l` is a single bit.

Types and shapes (single sample):

* `field_l`: `np.ndarray, dtype int {0,1}, shape (L,)`.
* `gun_l`: `np.ndarray, dtype int {0,1}, shape (L,)` one-hot.
* `sr_outcome_l`: `np.ndarray, dtype int {0,1}, shape (L/2,)`.
* `comm_l`: `np.ndarray, dtype int {0,1}, shape (1,)`.

#### Pyr end-to-end reference flow

Algorithm (single sample):

* Initialize:

  * `field_0 = field` reshaped/flattened to length `n2`.
  * `gun_0 = gun` one-hot length `n2`.
  * `comm_0` is a single bit initialized by Player A at level 0 (teacher-defined rule).
* For each level $l = 0..K-1$:

  * Player A:

    * Compute `meas_a_l = MeasureA_l(field_l)`.
    * Obtain SR outcomes `sr_outcome_l` for this level.
    * Compute `field_{l+1} = CombineA_l(field_l, sr_outcome_l)`.
    * Update `comm_l` according to the Pyr-A teacher rule and send the current `comm_l` (or final `comm_K`) to Player B depending on the protocol definition.
  * Player B:

    * Compute `meas_b_l = MeasureB_l(gun_l)`.
    * Obtain SR outcomes `sr_outcome_l` for this level.
    * Compute `(gun_{l+1}, comm_{l+1}) = CombineB_l(gun_l, sr_outcome_l, comm_l)`.
* Output:

  * Player B outputs `shoot = comm_K` or the protocol-defined final bit.

!!! warning
The exact rule for how `comm` is initialized and when it is transmitted MUST match the Pyr teacher implementation used for dataset generation.

## Trainable assisted models as constrained implementations

### Teacher vs trainable relationship

For both Lin and Pyr:

* Teacher primitives define the **reference mapping** used to generate supervised targets.
* Trainable models MUST be architectures that cannot violate the information flow constraints and are trained to approximate the teacher mapping.

Normative requirement:

* A trainable assisted model MUST NOT have any input path that bypasses:

  * `field -> comm` only through Player A,
  * `comm + gun -> shoot` only through Player B,
  * SR usage only through SR interfaces.

### Equivalence targets

When trained on the corresponding imitation datasets:

* `LinTrainableAssistedModelA` SHOULD approximate `Lin teacher A` mapping: `field -> comm`.
* `LinTrainableAssistedModelB` SHOULD approximate `Lin teacher B` mapping: `(gun, comm) -> shoot`.
* `PyrTrainableAssistedModelA` SHOULD approximate `Pyr teacher A` per-level mappings.
* `PyrTrainableAssistedModelB` SHOULD approximate `Pyr teacher B` per-level mappings.

!!! note
This page does not claim optimality of learned models. It specifies the intended target behavior and constraints.

## Preconditions, postconditions, errors summary

### Global preconditions

* `field_size >= 1`.
* `n2 = field_size^2`.
* `1 <= comms_size <= n2` unless specified otherwise.
* For Pyr: `n2` power of two and `comms_size = 1`.
* All bit-vectors are binary in `{0,1}` (or float32 equivalents `{0.0,1.0}` for TF).

### Global postconditions

* Player B output `shoot` is a single bit.
* Player B uses only `gun`, received `comm`, and SR outcomes.
* SR does not violate no-signaling constraints.

### Global errors

* `ValueError` for invalid shapes, invalid parameter domains, or violated structural constraints.
* `RuntimeError` for detected internal inconsistency (for example, mismatched level lists between Player A and Player B).

## Testing hooks

Suggested invariants to test algorithm correctness without training:

* Majority segmentation covers `[0, n2)` with no overlaps and total length `n2`.
* For Pyr: level sizes are `[n2, n2/2, ..., 2]` and count is `log2(n2)`.
* For Pyr: each level halves the active length for both field and gun representations.
* For Lin: `comm` has shape `(m,)` and `shoot` has shape `(1,)` for single-sample execution.
* No-signaling smoke test: holding `comm` fixed, varying `field` MUST NOT change Player B output distribution in expected-mode SR.

## Planned (design-spec)

* A fully explicit pseudocode listing for each teacher primitive (Measure/Combine for Lin and Pyr) may be added if the reference teachers are updated or extended.

## Deviations

* None.

## Changelog

* 2026-01-16 (Rob Hendriks): Initial `algorithms.md` drafted as normative baseline for assisted and trainable-assisted algorithm families.
