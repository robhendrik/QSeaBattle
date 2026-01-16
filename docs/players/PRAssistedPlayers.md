# Class PRAssistedPlayers

> **Role**: Factory and owner of PR-assisted resources that provides paired Player A and Player B instances for PR-assisted play.

**Module path**: `Q_Sea_Battle.pr_assisted_players`
**Status**: stable · **Version**: 0.2 · **Owner**: Rob Hendriks

## Purpose & Scope

* **Goal**: Construct and manage the hierarchy of PR-assisted resources required by the PR-assisted protocol and expose paired players that consume them.
* **Non-goals**: Learning, batching, or dynamic resizing of PR-assisted resources during a game.

## Public Interface (Summary)

* **Classes**: `PRAssistedPlayers`
* **Key methods**: `players()`, `reset()`, `pr_assisted(index)`
* **External contracts**: Depends on `GameLayout`, `Players`, `PRAssisted`, `PRAssistedPlayerA`, and `PRAssistedPlayerB`.

## Types & Shapes

* Let `field_size` be the linear field dimension.
* Let `n2 = field_size^2`.
* Let `m = comms_size`.

Constraints derived from `GameLayout` and the PR-assisted protocol:

* `m = 1`
* `n2` must be a power of two.
* Arrays used internally by PR-assisted resources are NumPy `np.ndarray`, dtype `int {0,1}`.

## Class: PRAssistedPlayers

### Constructor

`PRAssistedPlayers(game_layout, p_high)`

**Parameters**

* `game_layout` — `GameLayout`
  Game configuration; must satisfy `comms_size = 1` and `n2 = field_size^2` power-of-two.
* `p_high` — `float`
  Correlation parameter passed to all PR-assisted resources.

**Preconditions**

* `game_layout.comms_size = 1`
* `n2 = field_size^2 > 0`
* `n2` is a power of two.

**Postconditions**

* A hierarchy of PR-assisted resources is created and stored internally.
* No player instances are created until `players()` is called.

**Errors**

* `ValueError` if `comms_size != 1`.
* `ValueError` if `field_size <= 0`.
* `ValueError` if `n2` is not a power of two.

## Public Methods

### `players()`

**Purpose**: Create or return the cached pair of PR-assisted players.

**Args**

* None.

**Returns**

* `Tuple[PlayerA, PlayerB]`
  A tuple `(player_a, player_b)` where:

  * `player_a` is an instance of `PRAssistedPlayerA`.
  * `player_b` is an instance of `PRAssistedPlayerB`.

**Preconditions**

* Constructor preconditions hold.

**Postconditions**

* Exactly one instance of each player type is cached and reused on subsequent calls.

**Errors**

* None raised directly.

!!! example "Minimal usage"
```python
from Q_Sea_Battle.pr_assisted_players import PRAssistedPlayers
from Q_Sea_Battle.game_layout import GameLayout

````
layout = GameLayout(field_size=4, comms_size=1)
players = PRAssistedPlayers(layout, p_high=1.0)
player_a, player_b = players.players()
```
````

### `reset()`

**Purpose**: Reset internal PR-assisted resources between games.

**Args**

* None.

**Returns**

* `None`.

**Postconditions**

* All PR-assisted resources are re-created.
* Cached player instances remain valid but will query fresh resources.

**Errors**

* None raised directly.

### `pr_assisted(index)`

**Purpose**: Access a PR-assisted resource at a given protocol level.

**Args**

* `index` — `int`
  Level index into the PR-assisted resource hierarchy.

**Returns**

* `PRAssisted`
  PR-assisted resource corresponding to the given level.

**Errors**

* `IndexError` if `index` is out of bounds.

### `shared_randomness(index)`

**Purpose**: Deprecated compatibility alias for `pr_assisted(index)`.

**Args**

* `index` — `int`
  Level index into the PR-assisted resource hierarchy.

**Returns**

* `PRAssisted`
  Same object as returned by `pr_assisted(index)`.

**Errors**

* `IndexError` if `index` is out of bounds.

!!! warning
This method is deprecated and retained only for backward compatibility. Use `pr_assisted()` instead.

## Data & State

* Attributes (internal):

  * `_pr_assisted_array` — `list[PRAssisted]` — ordered by decreasing length.
  * `_playerA` — `PRAssistedPlayerA | None` — cached instance.
  * `_playerB` — `PRAssistedPlayerB | None` — cached instance.
* Side effects: None beyond object construction.
* Thread-safety: Not thread-safe; intended for single-game, single-thread usage.

## Testing Hooks

* Invariant: `len(_pr_assisted_array) = log2(n2)`.
* Invariant: Lengths of PR-assisted resources follow `2**(k)` with strictly decreasing powers of two.
* Invariant: `players()` returns the same object instances on repeated calls.
* Property test: After `reset()`, new PR-assisted resources are distinct objects.

## Notes for Contributors

* Do not change the ordering or number of PR-assisted resources; Player A and B rely on strict level alignment.
* Avoid introducing side effects in `pr_assisted()`; it must remain a pure accessor.
* The deprecated `shared_randomness()` should eventually be removed only with a major version bump.

## Deviations

* The design document refers generically to shared resources; the implementation names these explicitly as PR-assisted resources with a compatibility alias.

## Planned (design-spec)

* Support for alternative assisted resource types selectable via configuration is discussed in the design but not implemented here.

## Changelog

* 2026-01-16 — Rob Hendriks — Initial specification-grade documentation for `PRAssistedPlayers`.
