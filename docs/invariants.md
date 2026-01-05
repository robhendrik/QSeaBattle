# Invariants

## Purpose
This page lists **global invariants**: one-sentence truths that MUST hold across the QSeaBattle codebase.
These are intended to be:
- easy to audit during code review,
- easy to test,
- and stable over time.


## Game and environment
- The `field_size = n` is the board side length. The total number of cells is $n^2$.
- The field and gun vectors MUST have identical shape `(n^2,)` (or `(B, n^2)` batched).
- The gun vector MUST be one-hot: `gun.sum() == 1`.
- The total number of cells is to be a square and a power of 2 (so values 4, 16, 64, 256 are allowed and for example 8 and 32 not (not a square) and 25, 49 not (not a power of 2)).
- The comms_size (`m`) must divide the total number of cells $n^2$. This means we can always split the flattened input array of size $n^2$ in equal blocks size $L$ such that $mL = n^2$.
- Channel noise MUST be applied only to `comm`, never to `field` or `gun`.
- Rewards MUST be deterministic given `(field, gun, shoot)`.


## Communication
- Communication bandwidth is exactly `comms_size` bits per game.
- If `comms_size == 1`, communication MUST still be represented as shape `(B, 1)` (never a scalar).


## Assisted / shared randomness
- Shared randomness MUST NOT be treated as an extra communication channel.
- Each shared-randomness resource MAY be measured at most once by each party.
- For pyramid strategies, exactly one shared-randomness measurement is consumed per layer per party.


## Trainable assisted models (Lin/Pyr)
- Player A models MUST NOT access gun information.
- Player B models MUST NOT access the full field information.
- Any learned model MUST preserve the same input/output interfaces as the classical policy it approximates.
- Model state required for reconstruction (e.g., per-layer outcomes) MUST be captured during Player Aâ€™s decision
  and used by Player B; it MUST NOT be recomputed using hidden information.


## Data generation and training
- Imitation-learning targets MUST be produced by a spec-compliant classical policy.
- Training utilities MUST NOT introduce new information paths (no shortcuts, no extra inputs).
- Weight transfer utilities MUST NOT change topology; only parameter values may be copied.


## Logging and evaluation
- Each tournament log row MUST correspond to exactly one game.
- Evaluation utilities MUST NOT mutate player behavior beyond documented resets.


## Testability expectation (recommendation)
- Every invariant above SHOULD have at least one unit test or property test that would fail if the invariant is violated.