# Class PRAssistedPlayerB

> **Role**: Player B implementation that combines a one-hot gun index, PR-assisted measurements, and a single communication bit to decide whether to shoot.

**Module path**: `Q_Sea_Battle.pr_assisted_player_b`
**Status**: stable · **Version**: 0.2 · **Owner**: Rob Hendriks

## Purpose & Scope

* **Goal**: Implement the Player B side of the PR-assisted strategy as specified in the QSeaBattle design, using PR-assisted resources to resolve the gun index through iterative halving.
* **Non-goals**: Learning, batching, or stochastic exploration; this class is deterministic given PR-assisted outcomes and inputs.

## Public Interface (Summary)

* **Classes**: `PRAssistedPlayerB`
* **Key methods**: `decide(gun, comm, supp=None)`
* **External contracts**: Depends on `GameLayout`, `PlayersBase.PlayerB`, and `PRAssistedPlayers` for access to PR-assisted resources.

## Types & Shapes

* Let `field_size` be the linear field dimension.
* Let `n2 = field_size^2`.
* Let `m = comms_size`.

Constraints derived from `GameLayout` and the PR-assisted protocol:

* `m = 1`
* `n2` must be a power of two.
* Arrays are NumPy `np.ndarray`, dtype `int` in `{0,1}`.

## Class: PRAssistedPlayerB

### Constructor

`PRAssistedPlayerB(game_layout, parent)`

**Parameters**

* `game_layout` — `GameLayout`
  Game configuration; must satisfy `comms_size = 1` and `n2 = field_size^2` power-of-two.
* `parent` — `PRAssistedPlayers`
  Owning factory that provides access to PR-assisted resources per level.

**Preconditions**

* `isinstance(parent, PRAssistedPlayers)`
* `game_layout.comms_size = 1`
* `n2 = field_size^2` is a power of two.

**Postconditions**

* Player is bound to `game_layout`.
* `self.parent` references the owning `PRAssistedPlayers` instance.
* No mutable shared state is created beyond this reference.

**Errors**

* `TypeError` if `parent` is not a `PRAssistedPlayers` instance.

## Public Methods

### `decide(gun, comm, supp=None)`

**Purpose**: Decide whether to shoot by iteratively resolving the gun index using PR-assisted measurements and combining outcomes with the communication bit.

**Args**

* `gun` — `np.ndarray`, dtype `int {0,1}`, shape `(n2,)`
  One-hot gun vector; exactly one entry equals `1`.
* `comm` — `np.ndarray`, dtype `int {0,1}`, shape `(1,)`
  Single communication bit from Player A.
* `supp` — `Any | None`
  Optional supporting information; unused.

**Returns**

* `int`, scalar in `{0,1}`
  `1` indicates shoot, `0` indicates do not shoot.

**Algorithm**

1. Validate `gun` as one-hot and `comm` as a single bit.
2. Initialize `intermediate_gun = gun` and an empty list of PR-assisted outcomes.
3. While `intermediate_gun` has length greater than 1:

   * Partition into adjacent pairs.
   * Identify the unique active pair `(0,1)` or `(1,0)`.
   * Build a measurement string of length `n2 / 2`.
   * Query the PR-assisted resource at the current level via `measurement_b`.
   * Record the outcome corresponding to the active pair.
   * Collapse to a new one-hot vector of half the length.
4. Append the communication bit `comm[0]` to the outcomes.
5. Return the parity (mod 2 sum) of all recorded bits.

**Raises**

* `ValueError` when:

  * `gun` does not have shape `(n2,)`.
  * `gun` contains values outside `{0,1}`.
  * `gun` is not one-hot.
  * `comm` does not have shape `(1,)` or contains values outside `{0,1}`.
  * Intermediate vectors violate even-length or one-hot invariants.
  * Measurement construction violates protocol constraints.

!!! example "Minimal usage"
```python
from Q_Sea_Battle.pr_assisted_players import PRAssistedPlayers
from Q_Sea_Battle.game_layout import GameLayout
import numpy as np

````
layout = GameLayout(field_size=4, comms_size=1)
players = PRAssistedPlayers(layout, p_high=1.0)
_, player_b = players.players()

gun = np.zeros(layout.field_size**2, dtype=int)
gun[3] = 1
comm = np.array([1], dtype=int)

shoot = player_b.decide(gun, comm)
```
````

## Data & State

* Attributes (public):

  * `parent` — `PRAssistedPlayers` — provides access to PR-assisted resources.
* Side effects: None beyond querying PR-assisted resources.
* Thread-safety: Not thread-safe; assumes single-game, single-thread usage.

## Testing Hooks

* Invariant: At every iteration, `intermediate_gun.sum() == 1`.
* Invariant: At every iteration, `len(intermediate_gun)` is even until it reaches 1.
* Invariant: Exactly one active pair `(0,1)` or `(1,0)` exists per level.
* Property test: Flipping `comm[0]` flips the final decision parity.

## Notes for Contributors

* Keep terminology consistent: use "PR-assisted" rather than "shared randomness".
* Do not vectorize across levels; level-by-level semantics are part of the protocol.
* Any optimization must preserve one-hot and parity invariants exactly.

## Deviations

* Design documents may describe outcomes in abstract bit terms; the implementation explicitly encodes parity as `sum(results) % 2`.

## Planned (design-spec)

* Batched evaluation over multiple guns is discussed in the design but not implemented here.

## Changelog

* 2026-01-16 — Rob Hendriks — Initial specification-grade documentation for `PRAssistedPlayerB`.
