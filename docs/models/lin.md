# Trainable assisted models -- Linear (Lin)

## Purpose
Provide a trainable approximation of the classical assisted strategy using linear neural network layers, preserving all behavioral contracts defined for assisted players.

## Scope
This chapter specifies the **Lin** family of trainable assisted models. These models replace fixed logical operations with learned linear mappings while maintaining the same information-theoretic constraints.


## Class LinTrainableAssistedModelA

## Purpose
Approximate the classical AssistedPlayerA strategy using a single linear measurement layer and a linear combine layer.

## Location
- **Module:** `src/Q_Sea_Battle/lin_trainable_assisted_models.py`
- **Class:** `LinTrainableAssistedModelA`

## Inputs
| Name | Shape | Description |
|----|----|----|
| `field` | `(B, n^2)` | Binary field vector |

## Outputs
| Name | Shape | Description |
|----|----|----|
| `comm` | `(B, 1)` | Communication bit |

## Internal state
- `measurements_per_layer`: list of tensors
- `outcomes_per_layer`: list of tensors

## Behavioral contract
- Exactly **one measurement** is performed per decision.
- Exactly **one communication bit** is produced.
- The model **MUST NOT** use gun information.
- Shared randomness is accessed via a differentiable proxy.

## Invariants
- Input length `n^2` **MUST** be a power of two.
- `comms_size == 1`.


## Class LinTrainableAssistedModelB

## Purpose
Approximate the classical AssistedPlayerB strategy using linear layers to reconstruct the relevant field bit.

## Location
- **Module:** `src/Q_Sea_Battle/lin_trainable_assisted_models.py`
- **Class:** `LinTrainableAssistedModelB`

## Inputs
| Name | Shape | Description |
|----|----|----|
| `gun` | `(B, n^2)` | One-hot gun vector |
| `comm` | `(B, 1)` | Communication bit |

## Outputs
| Name | Shape | Description |
|----|----|----|
| `shoot` | `(B, 1)` | Binary decision |

## Behavioral contract
- Exactly **one measurement** is performed per decision.
- Output **MUST** be binary at inference time.
- The model **MUST NOT** access the field directly.

## Invariants
- Gun input **MUST** be one-hot.
- Decision depends only on `(gun, comm, shared randomness)`.


## Notes / rationale
The Lin models serve as the minimal trainable baseline. They are intentionally capacity-limited to:
- Test learnability under strict information constraints
- Provide a reference point for deeper architectures (Pyr)

All violations of classical-assisted invariants are considered specification errors.