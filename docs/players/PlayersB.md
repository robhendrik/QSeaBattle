# PlayersB

> **Role**: Planned class name for Player B factory/wrapper (not implemented in current code).

**Location**: `Q_Sea_Battle.players_base.PlayersB` (planned)

!!! warning "Not implemented"
    The current implementation does **not** define a `PlayersB` class. The base implementation defines `PlayerB`.

## Constructor

| Parameter | Type | Description |
|---|---|---|
| game_layout | `GameLayout`, scalar | Planned: shared configuration for the Player B implementation. |

**Preconditions**

- Planned: `game_layout` is a valid `GameLayout`, scalar.

**Postconditions**

- Planned: an instance is initialised and ready to produce `shoot` decisions.

**Errors**

- Planned: may raise `TypeError`/`ValueError` propagated from `GameLayout` validation when constructed upstream.

## Public Methods

!!! note "Planned (design-spec)"
    The design document specifies a Player B role but does not name it `PlayersB`.
    This page exists to match the requested documentation structure.

- Planned methods: not specified under the name `PlayersB`.

## Planned (design-spec)

- `PlayersB` is not explicitly present in the design document.
- The design specifies a Player B role that decides `shoot`, scalar `int` {0,1}, given `gun` and `comm`.

## Deviations

- Requested name vs implementation:
  - Requested documentation target: `PlayersB`.
  - Implementation provides: `PlayerB` with method:
    - `decide(gun, comm, supp=None) -> shoot`
      - `gun`: `np.ndarray`, dtype `int` {0,1}, shape `(n2,)`, one-hot
      - `comm`: `np.ndarray`, dtype `int` {0,1}, shape `(m,)`
      - `supp`: any or `None`, scalar
      - `shoot`: `int` {0,1}, scalar

## Data & State

- Planned attributes (public):
  - `game_layout` — `GameLayout`, scalar.

- Side effects:
  - Planned: none beyond producing decisions.

- Thread-safety:
  - Depends on concrete implementation; base `PlayerB` uses NumPy global RNG and is not thread-safe.

## Notes for Contributors

- If `PlayersB` is introduced, document whether it is:
  - an alias of `PlayerB`, or
  - a higher-level wrapper that constructs `PlayerB` instances.
- Keep method naming consistent with orchestration (`decide(gun, comm, supp=None)` returning `shoot`, scalar).
- If you rename classes, update both docs and any import paths used by notebooks and scripts.

## Related

- See also: `PlayerB` in `Q_Sea_Battle.players_base` (implemented), and `Players` (factory/container).

## Changelog

- 2026-01-07 — Author: Rob Hendriks — Initial placeholder specification page for `PlayersB`.
