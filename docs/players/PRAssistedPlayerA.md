# Class PRAssistedPlayerA

> **Role**: Player A implementation that compresses the field into a single communication bit using PR-assisted resources.

**Module path**: `Q_Sea_Battle.pr_assisted_player_a`
**Status**: stable · **Version**: 0.2 · **Owner**: Rob Hendriks

## Purpose & Scope

* **Goal**: Implement the Player A side of the PR-assisted strategy by iteratively compressing the input field into one bit using PR-assisted measurements.
* **Non-goals**: Learning, batching, stochastic exploration, or alternative compression strategies.

## Public Interface (Summary)

* **Classes**: `PRAssistedPlayerA`
* **Key methods**: `decide(field, supp=None)`
* **External contracts**: Depends on `GameLayout`, `PlayersBase.PlayerA`, and `PRAssistedPlayers` for PR-assisted resources.

## Types & Shapes

* Let `field_size` be the linear field dimension.
* Let `n2 = field_size^2`.
* Let `m = comms_size`.

Constraints derived from `GameLayout` and the PR-assisted protocol:

* `m = 1`
* `n2` must be a power of two.
* Arrays are NumPy `np.ndarray`, dtype `int` in `{0,1}`.

## Class: PRAssistedPlayerA

### Constructor

`PRAssistedPlayerA(game_layout, parent)`

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

### `decide(field, supp=None)`

**Purpose**: Compress the input field into a single communication bit using PR-assisted measurements.

**Args**

* `field` — `np.ndarray`, dtype `int {0,1}`, shape `(n2,)`
  Flattened field array representing the full battlefield.
* `supp` — `Any | None`
  Optional supporting information; unused.

**Returns**

* `np.ndarray`, dtype `int {0,1}`, shape `(1,)`
  Single-element array containing the communication bit.

**Algorithm**

1. Validate `field` shape and binary values.
2. Initialize `intermediate_field = field`.
3. For each level while `len(intermediate_field) > 1`:

   * Partition into adjacent pairs.
   * Build a measurement string where each entry encodes equality of a pair.
   * Query the PR-assisted resource at the current level via `measurement_a`.
   * Combine original bits and PR-assisted outcomes into auxiliary pairs.
   * Collapse each auxiliary pair into a single bit.
4. Return the final remaining bit as the communication bit.

**Raises**

* `ValueError` when:

  * `field` does not have shape `(n2,)`.
  * `field` contains values outside `{0,1}`.
  * Intermediate field length is not even at any level.
* `RuntimeError` if the final compressed field does not have length 1.

!!! example "Minimal usage"
```python
from Q_Sea_Battle.pr_assisted_players import PRAssistedPlayers
from Q_Sea_Battle.game_layout import GameLayout
import numpy as np

````
layout = GameLayout(field_size=4, comms_size=1)
players = PRAssistedPlayers(layout, p_high=1.0)
player_a, _ = players.players()

field = np.random.randint(0, 2, size=layout.field_size**2)
comm = player_a.decide(field)
```
````

## Data & State

* Attributes (public):

  * `parent` — `PRAssistedPlayers` — provides access to PR-assisted resources.
* Side effects: None beyond querying PR-assisted resources.
* Thread-safety: Not thread-safe; assumes single-game, single-thread usage.

## Testing Hooks

* Invariant: At every level, `len(intermediate_field)` is even until it reaches 1.
* Invariant: `intermediate_field` always contains only `{0,1}` values.
* Property test: For deterministic PR-assisted resources, repeated calls with identical `field` produce identical output.

## Notes for Contributors

* Preserve the exact level-by-level semantics; reordering operations breaks the protocol.
* Keep terminology consistent: use "PR-assisted" rather than "shared randomness".
* Any refactoring must preserve array shapes and collapse logic exactly.

## Deviations

* The design document describes the protocol abstractly; the implementation makes array-level pairing and collapse explicit.

## Planned (design-spec)

* Batched compression over multiple fields is discussed in the design but not implemented here.

## Changelog

* 2026-01-16 — Rob Hendriks — Initial specification-grade documentation for `PRAssistedPlayerA`.
