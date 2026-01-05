# Deterministic and baseline players

## Purpose
Define simple, non-learning baseline player families used for:
- sanity checks,
- debugging,
- performance lower bounds,
- and validating tournament logging.

This chapter specifies two baseline families:
1. **SimplePlayers** (a minimal baseline; Player B includes a stochastic fallback)
2. **MajorityPlayers** (a deterministic heuristic baseline)


## Class SimplePlayers(Players)

### Purpose
Factory that returns a matched `(SimplePlayerA, SimplePlayerB)` pair.

### Behavioral contract
- `SimplePlayers.players()` MUST generate instances of `SimplePlayerA` and `SimplePlayerB`.
- The family MUST respect communication bandwidth: output length equals `layout.comms_size`.


## Class SimplePlayerA(PlayerA)

### Purpose
Send the values of the first `m` field cells directly, where `m = layout.comms_size`.

### Algorithm (normative)
Let `m = comms_size`. Let `field` be the flattened field vector of length `n^2`.

1. Read the first `m` entries: `field[0:m]`.
2. Output `comm = field[0:m]`.

### Invariants
- MUST NOT access gun information.
- MUST output exactly `m` bits.


## Class SimplePlayerB(PlayerB)

### Purpose
If the gun points at one of the first `m` cells, use the communicated value for that cell.
Otherwise, fall back to a simple prior-based stochastic guess using `layout.enemy_probability`.

### Algorithm (normative)
Let `m = comms_size`. Let `gun` be one-hot of length `n^2`. Let `idx = argmax(gun)`.

1. If `idx < m`:
   - Output `shoot = comm[idx]`.
2. Else:
   - Sample `shoot ~ Bernoulli(layout.enemy_probability)`.

### Notes
This baseline is **not fully deterministic** due to the stochastic fallback in step 2.

### Invariants
- MUST NOT access full field information.
- Output MUST be binary (`0` or `1`).
- If `idx < m`, the output MUST equal the communicated bit for that index.


## Class MajorityPlayers(Players)

### Purpose
Factory that returns a matched `(MajorityPlayerA, MajorityPlayerB)` pair implementing a majority heuristic over partitions.

### Behavioral contract
- `MajorityPlayers.players()` MUST generate instances of `MajorityPlayerA` and `MajorityPlayerB`.
- The family MUST respect communication bandwidth: output length equals `layout.comms_size`.
- The family MUST be deterministic given inputs.


## Class MajorityPlayerA(PlayerA)

### Purpose
Split the flattened field into `m` pieces and send one majority bit per piece.

### Algorithm (normative)
Let `m = comms_size`. Let `N = n^2` be the flattened field length.

1. Partition indices `{0, 1, ..., N-1}` into `m` contiguous blocks:
   - Block `j` contains indices from `start = floor(j*N/m)` up to (but excluding) `end = floor((j+1)*N/m)`.
2. For each block `j`:
   - Let `ones_j = sum(field[i] for i in block j)`.
   - Let `zeros_j = |block j| - ones_j`.
   - Set `comm[j] = 1` if `ones_j >= zeros_j`, else `comm[j] = 0`.
3. Output `comm` (length `m`).

### Invariants
- MUST NOT access gun information.
- MUST output exactly `m` bits.
- MUST be deterministic given `field`.


## Class MajorityPlayerB(PlayerB)

### Purpose
Determine which subset (block) the gun points to and return the corresponding communicated majority bit.

### Algorithm (normative)
Let `m = comms_size`. Let `N = n^2`. Let `idx = argmax(gun)`.

1. Compute the block id `j` such that `idx` lies in block `j` under the same partition rule used by `MajorityPlayerA`.
2. Output `shoot = comm[j]`.

### Invariants
- MUST NOT access full field information.
- Output MUST be binary.
- MUST be deterministic given `(gun, comm)`.


## Related
- Base interfaces: `docs/players/base.md`
- Game parameters: `docs/game/GameLayout.md`