# Class GameEnv

## Purpose
Implement the physical rules of QSeaBattle: field generation, gun placement, reward evaluation, and communication noise.

## Location
- **Module:** `src/Q_Sea_Battle/game_env.py`
- **Class:** `GameEnv`

## Public methods
### `reset() -> None`
Generate a new random field and gun.

### `provide() -> tuple[np.ndarray, np.ndarray]`
Return `(field, gun)` as flattened binary arrays.

### `evaluate(shoot: int) -> float`
Return 1.0 if decision matches field value at gun index, else 0.0.

### `apply_channel_noise(comm: np.ndarray) -> np.ndarray`
Apply independent bit flips with probability `channel_noise`.

## Invariants
- `gun` MUST be one-hot.
- `field.shape == gun.shape == (n^2,)`