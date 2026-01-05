# Base player interfaces

## Purpose
Define the abstract interfaces for Player A and Player B, plus the factory interface that returns a matched pair.
Unless otherwise specified by a concrete implementation, the **default baseline behavior is random**.


## Class Players

### Purpose
Factory and lifecycle manager for paired `PlayerA` / `PlayerB` instances.

### Public methods
- `players() -> tuple[PlayerA, PlayerB]`  
  Return a matched `(player_a, player_b)` pair.
- `reset() -> None`  
  Reset any internal per-game state (if applicable).

### Invariants
- `players()` MUST always return a matched A/B pair.
- Both players MUST share the same `GameLayout`.


## Class PlayerA

### Purpose
Observe the field and produce a communication vector.

### Method
```python
decide(field: np.ndarray, supp: Any | None = None) -> np.ndarray
```

### Contract
- Output MUST be a binary array of length `comms_size`.
- Even when `comms_size == 1`, output MUST be an array (shape `(1,)` or `(B,1)`), not a scalar.
- Player A MUST NOT access gun information.

### Default baseline behavior (random)
If a concrete implementation does not specify a different decision rule, `PlayerA.decide` is interpreted as:
- Sample each communication bit independently and uniformly from `{0,1}`.

This provides a well-defined **random baseline** for debugging and sanity checks.


## Class PlayerB

### Purpose
Observe the gun position and received communication, and decide whether to shoot.

### Method
```python
decide(gun: np.ndarray, comm: np.ndarray, supp: Any | None = None) -> int
```

### Contract
- Output MUST be scalar `0` or `1` (or `(B,1)` for batched implementations).
- Input `gun` MUST be one-hot.
- Player B MUST NOT access the full field.

### Default baseline behavior (random)
If a concrete implementation does not specify a different decision rule, `PlayerB.decide` is interpreted as:
- Sample `shoot` uniformly from `{0,1}`, independent of `(gun, comm)`.


## Related
- Implemented by: deterministic, assisted, neural players
- Algorithms: `docs/algorithms.md` (informative)