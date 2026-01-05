# Assisted players (shared-randomness assisted)

## Purpose
Define Player A and Player B strategies that use a shared-randomness resource to outperform
unassisted baselines under strict communication constraints.

This chapter specifies the assisted-player family and its dependency on the shared randomness contract.


## Shared randomness dependency
Assisted players MUST use a resource that satisfies:

- `docs/shared_randomness/shared_randomness.md` (normative)

The resource may be implemented as:
- a pure Python class (`SharedRandomness`), or
- a differentiable Keras layer (`SharedRandomnessLayer`),

but the externally observable behavior MUST comply with the same contract.


## Class SharedRandomness()

### Purpose
Reference class name for the shared randomness resource used by assisted players.

> The authoritative behavior is defined in `docs/shared_randomness/shared_randomness.md`.


## Class AssistedPlayers(Players)

### Purpose
Factory and manager for assisted `(AssistedPlayerA, AssistedPlayerB)` pairs that share correlated resources.

### Location
- **Module:** `src/Q_Sea_Battle/assisted_players.py`
- **Class:** `AssistedPlayers`

### Parameters
| Name | Type | Constraints |
|------|------|-------------|
| `p_high` | `float` | `[0,1]` |
| `layout` | `GameLayout` | `comms_size == 1` (recommended baseline) |

### Behavioral contract
- Communication bandwidth MUST be respected (`comms_size` bits).
- Assisted players MUST NOT treat shared randomness as extra communication.
- Per game, the shared randomness resources MUST be reset.


## Class AssistedPlayerA(PlayerA)

### Purpose
Observe the field and compute `comm` using shared randomness according to a fixed hierarchical reduction scheme.

### Behavioral contract (normative)
- MUST NOT access gun information.
- MUST consume shared randomness in the same sequence that Player B will mirror.
- MUST produce exactly `comms_size` bits (typically 1).

### Algorithmic outline (informative)
- Reduce the field iteratively (pairwise grouping) while consuming one SR measurement per reduction stage.
- Store per-stage outcomes needed for Player B reconstruction.


## Class AssistedPlayerB(PlayerB)

### Purpose
Use gun position, received `comm`, and shared randomness outcomes to reconstruct the relevant field bit.

### Behavioral contract (normative)
- MUST NOT access full field information.
- MUST mirror Player Aâ€™s SR consumption order (one SR measurement per stage).
- Gun input MUST remain one-hot throughout any internal transformations.
- Output MUST be binary at inference time.

### Algorithmic outline (informative)
- Traverse the reduction hierarchy guided by gun position.
- Combine SR outcomes and `comm` to compute `shoot`.


## Invariants
- For the standard assisted protocol: `comms_size == 1` and `n^2` is a power of two.
- Each SR resource is measured at most once per party per game.