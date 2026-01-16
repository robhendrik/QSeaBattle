# Invariants

## Purpose
This page lists global invariants: one-sentence truths that MUST hold across the QSeaBattle codebase.

These are intended to be:
- easy to audit during code review,
- easy to test,
- stable over time.

## Game and environment
- `field_size` is the board side length and `n2 = field_size * field_size`.
- The field and gun vectors MUST have identical shape `(n2,)` (or `(B, n2)` when batched).
- The gun vector MUST be one-hot, for example `gun.sum() == 1` for a single sample.
- `n2` MUST be both a perfect square and a power of two.
- `comms_size = m` MUST divide `n2`.
- Channel noise MUST be applied only to `comm`, never to `field` or `gun`.
- Rewards MUST be deterministic given `(field, gun, shoot)`.

## Communication
- Communication bandwidth is exactly `comms_size` bits per game.
- If `comms_size == 1`, communication MUST still be represented as shape `(m,)` or `(B, 1)` (never a scalar).

## Assisted and shared resources
- SR (shared resource) MUST NOT be treated as an extra communication channel.
- Each SR instance MAY be measured at most once by each party.
- For pyramid strategies, exactly one SR measurement is consumed per layer per party.

## Trainable assisted models (Lin and Pyr)
- Player A models MUST NOT access gun information.
- Player B models MUST NOT access the full field information.
- Any learned model MUST preserve the same input and output interfaces as the classical policy it approximates.
- Model state required for reconstruction (for example, per-layer outcomes) MUST be captured during Player A's decision
  and used by Player B; it MUST NOT be recomputed using hidden information.

## Data generation and training
- Imitation-learning targets MUST be produced by a spec-compliant classical policy.
- Training utilities MUST NOT introduce new information paths (no shortcuts, no extra inputs).
- Weight transfer utilities MUST NOT change topology; only parameter values may be copied.

## Logging and evaluation
- Each tournament log row MUST correspond to exactly one game.
- Evaluation utilities MUST NOT mutate player behavior beyond documented resets.

## Testability expectation
- Every invariant above SHOULD have at least one unit test or property test that fails if the invariant is violated.

## Changelog
- 2026-01-16 - Rob Hendriks: Update SR terminology and module naming conventions.
