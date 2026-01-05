# Training considerations for Pyramid (Pyr) models

## Purpose
Specify additional constraints and recommendations for training Pyramid assisted models.

## Training regimes
- Supervised imitation (from classical pyramid policy)
- Layer-wise pretraining (optional)
- End-to-end fine-tuning (allowed but constrained)

## Behavioral constraints
- Training **MUST NOT** introduce cross-layer shortcuts
- Shared-randomness usage **MUST** remain one-per-layer
- Communication bandwidth **MUST** remain 1 bit

## Diagnostics (recommended)
- Per-layer loss monitoring
- Consistency checks between adjacent layers
- Ablation of individual layers

## Invariants
- Runtime behavior **MUST** match Chapter 8 contracts
- Learned weights **MUST NOT** change input/output interfaces