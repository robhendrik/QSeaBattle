# Utility module: Layer and weight transfer

## Purpose
Provide safe utilities for transferring trained weights from standalone layers
into composite assisted models.

## Core utilities

### `transfer_assisted_model_a_layer_weights(...)`
Copy trained Player A layers into a composite model.

### `transfer_assisted_model_b_layer_weights(...)`
Copy trained Player B layers into a composite model.

## Behavioral contract
- Only weights may be transferred
- Model topology MUST remain unchanged
- Any shape mismatch MUST raise an error

## Invariants
- Post-transfer model MUST satisfy original spec contracts