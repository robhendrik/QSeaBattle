# Utility module: Imitation training

## Purpose
Define helper utilities to train trainable assisted models using supervised imitation learning,
while preserving all runtime behavioral contracts.

## Core utilities

### `train_layer(layer, dataset, loss, epochs, metrics=None)`
Train a single neural layer in isolation.

**Contract**
- The utility MUST NOT alter layer input/output signatures
- The utility MUST NOT introduce cross-layer information flow


### `train_model(model, datasets)`
Orchestrate training of full Lin or Pyr models.

**Contract**
- Training MUST respect per-layer constraints (Chapter 7, 8)
- Runtime behavior MUST remain unchanged after training

## Invariants
- Training utilities MUST NOT modify game logic
- All randomness MUST be controlled and logged