# PlayersA

> **Role**: Planned class name for Player A factory/wrapper (not implemented in current code).

**Location**: `Q_Sea_Battle.players_base.PlayersA` (planned)

!!! warning "Not implemented"
    The current implementation does **not** define a `PlayersA` class. The base implementation defines `PlayerA`.

## Constructor

| Parameter | Type | Description |
|---|---|---|
| game_layout | `GameLayout`, scalar | Planned: shared configuration for the Player A implementation. |

**Preconditions**

- Planned: `game_layout` is a valid `GameLayout`, scalar.

**Postconditions**

- Planned: an instance is initialised and ready to produce `comm` decisions.

**Errors**

- Planned: may raise `TypeError`/`ValueError` propagated from `GameLayout` validation when constructed upstream.

## Public Methods

!!! note "Planned (design-spec)"
    The design document specifies a Player A role but does not name it `PlayersA`.
    This page exists to match the requested documentation structure.

- Planned methods: not specified under the name `PlayersA`.

## Planned (design-spec)

- `PlayersA` is not explicitly present in the design document.
- The design specifies a Player A role that decides a communication vector `comm`, shape `(m,)`.

## Deviations

- Requested name vs implementation:
  - Requested documentation target: `PlayersA`.
  - Implementation provides: `PlayerA` with method:
    - `decide(field, supp=None) -> comm`
      - `field`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)`
      - `supp`: any or `None`, scalar
      - `comm`: `np.ndarray`, dtype `int` {0,1}, shape `(m,)`

## Data & State

- Planned attributes (public):
  - `game_layout` — `GameLayout`, scalar.

- Side effects:
  - Planned: none beyond producing decisions.

- Thread-safety:
  - Depends on concrete implementation; base `PlayerA` uses NumPy global RNG and is not thread-safe.

## Notes for Contributors

- If `PlayersA` is introduced, document whether it is:
  - an alias of `PlayerA`, or
  - a higher-level wrapper that constructs `PlayerA` instances.
- Keep method naming consistent with orchestration (`decide(field, supp=None)` returning `comm`, shape `(m,)`).
- If you rename classes, update both docs and any import paths used by notebooks and scripts.

## Related

- See also: `PlayerA` in `Q_Sea_Battle.players_base` (implemented), and `Players` (factory/container).

## Changelog

- 2026-01-07 — Author: Rob Hendriks — Initial placeholder specification page for `PlayersA`.
