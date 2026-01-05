# Imitation learning for assisted models

## Purpose
Train Lin (and later Pyr) assisted models to imitate the optimal classical assisted strategy using supervised learning.

## Data generation
- Generate `(field, gun)` samples using `GameEnv`
- Generate optimal `(comm, shoot)` targets using classical AssistedPlayers

## Behavioral contract
- Training data **MUST** be generated from a spec-compliant classical policy.
- Training **MUST NOT** introduce additional information channels.

## Losses
- Binary cross-entropy for communication bit
- Binary cross-entropy for shooting decision

## Invariants
- Dataset inputs and targets **MUST** match the shapes defined in Chapter 7.
- Training **MUST NOT** alter the runtime contracts of the models.