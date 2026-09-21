"""QSeaBattle: Linear (depth=1) per-game dataset generation utilities.

This module generates and validates the canonical storage format for "linear"
teacher traces (single-step/depth=1). Each generated game is represented as
binary float32 arrays describing:

- The hidden field bits (the "board").
- A one-hot gun position (the selected target index).
- Communication bits shared across both players.
- Per-player measurement inputs and outputs.
- The resulting `shoot` label (hit/miss bit at the gun index).

All arrays use float32 with logical bits encoded as {0.0, 1.0}.

Canonical shapes (single game vs. dataset):
    Single game (no leading batch dimension):
        field_bits:      (2, n2)
        gun_bits:        (2, n2)
        comms_bits:      (2, m)
        meas_in_a_bits:  (1, n2)
        meas_out_a_bits: (1, n2)
        meas_in_b_bits:  (1, n2)
        meas_out_b_bits: (1, n2)
        shoot:           (1,)

    Dataset (N games):
        field_bits:      (N, 2, n2)
        gun_bits:        (N, 2, n2)
        comms_bits:      (N, 2, m)
        meas_in_a_bits:  (N, 1, n2)
        meas_out_a_bits: (N, 1, n2)
        meas_in_b_bits:  (N, 1, n2)
        meas_out_b_bits: (N, 1, n2)
        shoot:           (N, 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class LinSizes:
    """Dimension bundle for linear (depth=1) traces.

    Attributes:
        n2: Number of field/gun bits.
        m: Number of communication bits.
        depth: Trace depth (fixed to 1 for this module).
    """

    n2: int
    m: int
    depth: int = 1


def lin_sizes(n2: int, m: int) -> LinSizes:
    """Create a validated :class:`LinSizes` for linear traces.

    Args:
        n2: Number of field/gun bits. Must be >= 1.
        m: Number of communication bits. Must be >= 1.

    Returns:
        A :class:`LinSizes` with ``depth=1``.

    Raises:
        ValueError: If ``n2 < 1`` or ``m < 1``.
    """
    if n2 < 1:
        raise ValueError("n2 must be >= 1.")
    if m < 1:
        raise ValueError("m must be >= 1.")
    return LinSizes(n2=int(n2), m=int(m), depth=1)


def _rng(seed: int) -> np.random.Generator:
    """Construct a NumPy RNG from an integer seed."""
    return np.random.default_rng(int(seed))


def _assert_binary(x: np.ndarray, name: str) -> None:
    """Validate that an array contains only {0.0, 1.0} values."""
    u = np.unique(x)
    if not np.all((u == 0.0) | (u == 1.0)):
        raise ValueError(f"{name} must be binary {{0,1}} float32; got unique values {u!r}.")


def _assert_one_hot(x: np.ndarray, name: str) -> None:
    """Validate that an array is binary and contains exactly one '1'."""
    _assert_binary(x, name)
    if int(np.sum(x)) != 1:
        raise ValueError(f"{name} must be one-hot; got sum={int(np.sum(x))}.")


def _parity(x: np.ndarray) -> float:
    """Compute XOR parity of an array interpreted as bits via thresholding.

    Args:
        x: Array whose entries are interpreted as bits via ``x > 0.5``.

    Returns:
        0.0 or 1.0 as a Python float.
    """
    return float(np.bitwise_xor.reduce((x > 0.5).astype(np.int32)))


def _pr_rule_out_b(out_a: np.ndarray, meas_in_a: np.ndarray, meas_in_b: np.ndarray) -> np.ndarray:
    """Compute B's output under the PR-assisted correlation rule.

    This implements a simple "PR-assisted" correlation: for indices where both
    measurement inputs are 1, B's output is flipped relative to A's output.

    Args:
        out_a: A's output bits, interpreted via ``> 0.5``.
        meas_in_a: A's measurement input bits.
        meas_in_b: B's measurement input bits.

    Returns:
        B's output bits as float32 in {0.0, 1.0}.
    """
    flip = (meas_in_a > 0.5) & (meas_in_b > 0.5)
    return np.logical_xor(out_a > 0.5, flip).astype(np.float32)


def generate_one_game_trace_lin(
    n2: int,
    m: int,
    *,
    seed: int = 0,
    validate: bool = True,
) -> Dict[str, np.ndarray]:
    """Generate a single linear (depth=1) teacher trace.

    The generated trace uses binary float32 arrays. The gun is a one-hot vector
    choosing a single index; `shoot` is the field bit at that index.

    Communication bits are constructed as a length-``m`` constant vector equal to
    the parity of A's measurement output.

    Args:
        n2: Number of field/gun bits.
        m: Number of communication bits.
        seed: RNG seed for this game.
        validate: If True, run :func:`validate_one_game_trace_lin` on the result.

    Returns:
        A dict of NumPy arrays with the canonical per-game shapes documented in
        the module docstring.

    Raises:
        ValueError: If validation is enabled and the generated trace fails
            validation (should not occur for this generator).
    """
    s = lin_sizes(n2, m)
    rng = _rng(seed)

    field_bits = np.zeros((2, s.n2), dtype=np.float32)
    gun_bits = np.zeros((2, s.n2), dtype=np.float32)
    comms_bits = np.zeros((2, s.m), dtype=np.float32)
    meas_in_a_bits = np.zeros((1, s.n2), dtype=np.float32)
    meas_out_a_bits = np.zeros((1, s.n2), dtype=np.float32)
    meas_in_b_bits = np.zeros((1, s.n2), dtype=np.float32)
    meas_out_b_bits = np.zeros((1, s.n2), dtype=np.float32)

    field0 = rng.integers(0, 2, size=(s.n2,), dtype=np.int32).astype(np.float32)
    idx0 = int(rng.integers(0, s.n2))
    gun0 = np.zeros((s.n2,), dtype=np.float32)
    gun0[idx0] = 1.0

    shoot = np.array([field0[idx0]], dtype=np.float32)

    # A receives the full field as its measurement input and emits an arbitrary
    # output string; B receives the one-hot gun index and responds using the
    # PR-assisted rule based on A's output and both measurement inputs.
    meas_in_a = field0.copy()
    out_a = rng.integers(0, 2, size=(s.n2,), dtype=np.int32).astype(np.float32)
    comm_bit = _parity(out_a)
    comm0 = np.full((s.m,), comm_bit, dtype=np.float32)

    meas_in_b = gun0.copy()
    out_b = _pr_rule_out_b(out_a, meas_in_a, meas_in_b)

    # Duplicate values across the player axis to match the canonical (2, *)
    # convention used by downstream consumers.
    field_bits[0] = field0
    field_bits[1] = field0
    gun_bits[0] = gun0
    gun_bits[1] = gun0
    comms_bits[0] = comm0
    comms_bits[1] = comm0

    meas_in_a_bits[0] = meas_in_a
    meas_out_a_bits[0] = out_a
    meas_in_b_bits[0] = meas_in_b
    meas_out_b_bits[0] = out_b

    if validate:
        validate_one_game_trace_lin(
            n2,
            m,
            field_bits=field_bits,
            gun_bits=gun_bits,
            comms_bits=comms_bits,
            meas_in_a_bits=meas_in_a_bits,
            meas_out_a_bits=meas_out_a_bits,
            meas_in_b_bits=meas_in_b_bits,
            meas_out_b_bits=meas_out_b_bits,
            shoot=shoot,
        )

    return dict(
        field_bits=field_bits,
        gun_bits=gun_bits,
        comms_bits=comms_bits,
        meas_in_a_bits=meas_in_a_bits,
        meas_out_a_bits=meas_out_a_bits,
        meas_in_b_bits=meas_in_b_bits,
        meas_out_b_bits=meas_out_b_bits,
        shoot=shoot,
    )


def validate_one_game_trace_lin(
    n2: int,
    m: int,
    *,
    field_bits: np.ndarray,
    gun_bits: np.ndarray,
    comms_bits: np.ndarray,
    meas_in_a_bits: np.ndarray,
    meas_out_a_bits: np.ndarray,
    meas_in_b_bits: np.ndarray,
    meas_out_b_bits: np.ndarray,
    shoot: np.ndarray,
) -> None:
    """Validate shapes and bit encodings for a single linear (depth=1) trace.

    This checks:
      - Exact array shapes for the per-game format.
      - Binary encoding (values in {0.0, 1.0}) for all arrays.
      - One-hot encoding for both players' gun vectors.

    Args:
        n2: Number of field/gun bits.
        m: Number of communication bits.
        field_bits: Array of shape (2, n2).
        gun_bits: Array of shape (2, n2).
        comms_bits: Array of shape (2, m).
        meas_in_a_bits: Array of shape (1, n2).
        meas_out_a_bits: Array of shape (1, n2).
        meas_in_b_bits: Array of shape (1, n2).
        meas_out_b_bits: Array of shape (1, n2).
        shoot: Array of shape (1,).

    Raises:
        ValueError: If any shape constraint or bit constraint is violated.
    """
    s = lin_sizes(n2, m)

    if field_bits.shape != (2, s.n2):
        raise ValueError(f"field_bits shape {field_bits.shape} != (2,{s.n2}).")
    if gun_bits.shape != (2, s.n2):
        raise ValueError(f"gun_bits shape {gun_bits.shape} != (2,{s.n2}).")
    if comms_bits.shape != (2, s.m):
        raise ValueError(f"comms_bits shape {comms_bits.shape} != (2,{s.m}).")

    for name, arr in [
        ("meas_in_a_bits", meas_in_a_bits),
        ("meas_out_a_bits", meas_out_a_bits),
        ("meas_in_b_bits", meas_in_b_bits),
        ("meas_out_b_bits", meas_out_b_bits),
    ]:
        if arr.shape != (1, s.n2):
            raise ValueError(f"{name} shape {arr.shape} != (1,{s.n2}).")

    if shoot.shape != (1,):
        raise ValueError(f"shoot shape {shoot.shape} != (1,).")

    _assert_binary(field_bits, "field_bits")
    _assert_binary(gun_bits, "gun_bits")
    _assert_binary(comms_bits, "comms_bits")
    _assert_binary(meas_in_a_bits, "meas_in_a_bits")
    _assert_binary(meas_out_a_bits, "meas_out_a_bits")
    _assert_binary(meas_in_b_bits, "meas_in_b_bits")
    _assert_binary(meas_out_b_bits, "meas_out_b_bits")
    _assert_binary(shoot.astype(np.float32), "shoot")

    _assert_one_hot(gun_bits[0], "gun_bits[0]")
    _assert_one_hot(gun_bits[1], "gun_bits[1]")


def generate_lin_dataset(
    n2: int,
    m: int,
    num_games: int,
    *,
    seed: int = 0,
    validate: bool = True,
) -> Dict[str, np.ndarray]:
    """Generate a dataset of independent linear (depth=1) game traces.

    Each game uses seed ``seed + i`` for game index ``i``.

    Args:
        n2: Number of field/gun bits.
        m: Number of communication bits.
        num_games: Number of games to generate. Must be > 0.
        seed: Base RNG seed.
        validate: If True, validate each per-game trace as it is generated.

    Returns:
        A dict of NumPy arrays with the canonical dataset shapes documented in
        the module docstring.

    Raises:
        ValueError: If ``num_games <= 0`` or if validation fails for any game.
    """
    s = lin_sizes(n2, m)
    N = int(num_games)
    if N <= 0:
        raise ValueError("num_games must be > 0.")

    field_bits = np.zeros((N, 2, s.n2), dtype=np.float32)
    gun_bits = np.zeros((N, 2, s.n2), dtype=np.float32)
    comms_bits = np.zeros((N, 2, s.m), dtype=np.float32)
    meas_in_a_bits = np.zeros((N, 1, s.n2), dtype=np.float32)
    meas_out_a_bits = np.zeros((N, 1, s.n2), dtype=np.float32)
    meas_in_b_bits = np.zeros((N, 1, s.n2), dtype=np.float32)
    meas_out_b_bits = np.zeros((N, 1, s.n2), dtype=np.float32)
    shoot = np.zeros((N, 1), dtype=np.float32)

    for i in range(N):
        g = generate_one_game_trace_lin(s.n2, s.m, seed=seed + i, validate=validate)
        field_bits[i] = g["field_bits"]
        gun_bits[i] = g["gun_bits"]
        comms_bits[i] = g["comms_bits"]
        meas_in_a_bits[i] = g["meas_in_a_bits"]
        meas_out_a_bits[i] = g["meas_out_a_bits"]
        meas_in_b_bits[i] = g["meas_in_b_bits"]
        meas_out_b_bits[i] = g["meas_out_b_bits"]
        shoot[i, 0] = float(g["shoot"][0])

    return dict(
        field_bits=field_bits,
        gun_bits=gun_bits,
        comms_bits=comms_bits,
        meas_in_a_bits=meas_in_a_bits,
        meas_out_a_bits=meas_out_a_bits,
        meas_in_b_bits=meas_in_b_bits,
        meas_out_b_bits=meas_out_b_bits,
        shoot=shoot,
    )


def save_npz(path: str, ds: Dict[str, np.ndarray]) -> None:
    """Save a generated dataset dict as a compressed ``.npz`` file.

    Args:
        path: Output file path.
        ds: Dataset dict mapping names to NumPy arrays (e.g., from
            :func:`generate_lin_dataset`).
    """
    np.savez_compressed(path, **ds)