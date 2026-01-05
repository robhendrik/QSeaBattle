# Trainable assisted models -- Pyramid (Pyr)

## Purpose
Define a hierarchical, multi-layer trainable assisted model that scales the classical assisted strategy
to large fields by iteratively reducing the problem size while preserving all assisted-player contracts.

The Pyramid (Pyr) models generalize the Lin models by composing **repeated measurement-combine stages**
in a strictly structured hierarchy.

## Conceptual overview

Let the field have size $n^2$, with $n^2 = 2^k$.
The Pyr architecture performs $k$ iterations.  
At each iteration:
- The effective field size is halved
- One shared-randomness measurement is consumed
- Intermediate outcomes are stored for later reconstruction

Exactly **one communication bit** is produced at the final layer.

## Class PyrTrainableAssistedModelA

## Purpose
Approximate the classical AssistedPlayerA pyramid strategy using trainable neural layers,
producing a single communication bit from a large field.

## Location
- **Module:** `src/Q_Sea_Battle/pyr_trainable_assisted_models.py`
- **Class:** `PyrTrainableAssistedModelA`

## Inputs
| Name | Shape | Description |
|------|-------|-------------|
| `field` | `(B, n^2)` | Binary field vector |

## Outputs
| Name | Shape | Description |
|------|-------|-------------|
| `comm` | `(B, 1)` | Communication bit |

## Internal state

The model **MUST** store the following per decision:

- `measurements_per_layer[i]`: measurement tensor at layer *i*
- `outcomes_per_layer[i]`: shared-randomness outcomes at layer *i*

where:
- Layer 0 operates on size $n^2$
- Layer $i+1$ operates on size $n^2 / 2^{i+1}$

## Iterative structure (normative)

For layer $i = 0, \dots, k-1$:

1. **Measurement layer**
   - Input size: $n^2 / 2^i$
   - Output size: $n^2 / 2^{i+1}$

2. **Shared randomness**
   - One measurement is performed
   - Outcome size: $n^2 / 2^{i+1}$

3. **Combine layer**
   - Inputs: measurement output + shared randomness
   - Output: reduced field for next layer

At the final layer, the reduced field has size 1 and is emitted as `comm`.

## Behavioral contract
- Exactly **one shared-randomness measurement per layer**
- Exactly **one communication bit total**
- No access to gun information
- No skipping or reordering of layers

## Invariants
- Input size **MUST** be a power of two
- Number of layers = $\log_2(n^2)$
- Lengths of `measurements_per_layer` and `outcomes_per_layer` **MUST** equal number of layers

## Failure modes
- `ValueError` if input size is not a power of two
- `RuntimeError` if layers are executed inconsistently

## Class PyrTrainableAssistedModelB

## Purpose
Reconstruct the relevant field bit using hierarchical outcomes, gun position, and a single communication bit.

## Location
- **Module:** `src/Q_Sea_Battle/pyr_trainable_assisted_models.py`
- **Class:** `PyrTrainableAssistedModelB`

## Inputs
| Name | Shape | Description |
|------|-------|-------------|
| `gun` | `(B, n^2)` | One-hot gun vector |
| `comm` | `(B, 1)` | Communication bit |

## Outputs
| Name | Shape | Description |
|------|-------|-------------|
| `shoot` | `(B, 1)` | Binary decision |

## Internal state
- `outcomes_per_layer[i]`: outcomes received from Player A
- Optional learned reconstruction layers per hierarchy level

## Reconstruction procedure (normative)

For layer $i = k-1, \dots, 0$:

1. Use gun position to select the relevant subtree
2. Combine:
   - Corresponding shared-randomness outcome
   - Current reconstruction state
   - Communication bit (only at final stage)

The final output is a scalar decision.

## Behavioral contract
- Exactly **one shared-randomness measurement per layer**
- Gun input **MUST** remain one-hot throughout reconstruction
- Output **MUST** be binary at inference time

## Invariants
- Reconstruction depth equals measurement depth
- No access to full field information

## Notes / rationale
The Pyr architecture:
- Scales logarithmically with field size
- Mirrors the classical pyramid protocol
- Makes information flow explicit and inspectable

Any deviation from the layer structure constitutes a **spec violation**.