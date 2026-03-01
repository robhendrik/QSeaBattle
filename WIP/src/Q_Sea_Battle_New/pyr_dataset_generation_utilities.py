"""QSeaBattle WIP: Pyramid per-game dataset generation utilities.

This module generates *per-game* binary traces in the **canonical storage format**
defined in the frozen WIP spec/acceptance documents.

Key properties (normative):
- One sample == one full game.
- On disk: *binary* {0,1} only (stored as float32).
- Dense, right-padded vectors to length n2.
- Shapes:
    field_bits:      (N, depth+1, n2)
    gun_bits:        (N, depth+1, n2)
    comms_bits:      (N, depth+1, 1)
    meas_in_a_bits:  (N, depth,   n2)
    meas_out_a_bits: (N, depth,   n2)
    meas_in_b_bits:  (N, depth,   n2)
    meas_out_b_bits: (N, depth,   n2)
    shoot:           (N, 1)

Implementation note
-------------------
The *logic* is aligned with the existing teacher utilities in
`pyr_trainable_assisted_imitation_utilities.py` (follow code over docstrings
if they disagree).

This file is deliberately dependency-light (NumPy only), so it can run in data
generation scripts without TensorFlow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# --- Self-contained teacher logic (authoritative: follow code over external docstrings) ---
def _pairs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return even/odd indexed halves: x[0::2], x[1::2]."""
    return x[0::2], x[1::2]

def teacher_measure_a(field: np.ndarray) -> np.ndarray:
    """Measurement A teacher: pairwise XOR of (even, odd)."""
    even, odd = _pairs(field)
    return np.logical_xor(even > 0.5, odd > 0.5).astype(np.float32)

def teacher_combine_a(field: np.ndarray, sr_outcome: np.ndarray) -> np.ndarray:
    """Combine A teacher: even bits XOR SR outcome."""
    even, _ = _pairs(field)
    return np.logical_xor(even > 0.5, sr_outcome > 0.5).astype(np.float32)

def teacher_measure_b(gun: np.ndarray) -> np.ndarray:
    """Measurement B teacher: (NOT even) AND odd."""
    even, odd = _pairs(gun)
    return (np.logical_not(even > 0.5) & (odd > 0.5)).astype(np.float32)

def teacher_combine_b(gun: np.ndarray, sr_outcome: np.ndarray, comm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Combine B teacher producing (next_gun, next_comm)."""
    even, odd = _pairs(gun)
    next_gun = (np.logical_xor(even > 0.5, odd > 0.5)).astype(np.float32)
    idx = int(np.argmax(next_gun))
    next_comm = np.array([float((comm[0] > 0.5) ^ (sr_outcome[idx] > 0.5))], dtype=np.float32)
    return next_gun, next_comm




@dataclass(frozen=True)
class PyrSizes:
    """Derived pyramid sizes for a given n2."""

    n2: int
    depth: int
    L: Tuple[int, ...]  # per level input widths (length depth), L[d] = n2 / 2^d
    k: Tuple[int, ...]  # per level half widths (length depth), k[d] = L[d] / 2


def pyr_sizes(n2: int) -> PyrSizes:
    """Compute pyramid sizes.

    Parameters
    ----------
    n2:
        Total field cells. Must be a power of two and >= 2.
    """
    if n2 < 2:
        raise ValueError("n2 must be >= 2.")
    if n2 & (n2 - 1) != 0:
        raise ValueError("n2 must be a power of two.")
    depth = int(np.log2(n2))
    L = tuple(int(n2 // (2**d)) for d in range(depth))
    k = tuple(int(Ld // 2) for Ld in L)
    if k[-1] != 1:
        raise ValueError("Invalid pyramid sizing: expected k_{depth-1} == 1.")
    return PyrSizes(n2=n2, depth=depth, L=L, k=k)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _pad(prefix: np.ndarray, n2: int) -> np.ndarray:
    """Right-pad a 1D prefix vector to length n2 with zeros."""
    out = np.zeros((n2,), dtype=np.float32)
    L = int(prefix.shape[0])
    out[:L] = prefix.astype(np.float32)
    return out


def _assert_binary(x: np.ndarray, name: str) -> None:
    u = np.unique(x)
    if not np.all((u == 0.0) | (u == 1.0)):
        raise ValueError(f"{name} must be binary {{0,1}} float32; got unique values {u!r}.")


def _assert_one_hot_prefix(x: np.ndarray, L: int, name: str) -> None:
    prefix = x[:L]
    _assert_binary(prefix.astype(np.float32), name)
    s = int(np.sum(prefix))
    if s != 1:
        raise ValueError(f"{name} must be one-hot within [0:{L}); got sum={s}.")


def _pr_rule_out_b(out_a: np.ndarray, meas_in_a: np.ndarray, meas_in_b: np.ndarray) -> np.ndarray:
    """Binary PR correlation rule for the *second* measurement.

    In replay-mode, the second outcome matches the first, except when *both*
    measurement inputs are high.

    Here, the binary teacher convention is: "high" == 1.

    Therefore:
        out_b = out_a XOR (meas_in_a AND meas_in_b)
    """
    flip = (meas_in_a > 0.5) & (meas_in_b > 0.5)
    out_b = np.logical_xor(out_a > 0.5, flip).astype(np.float32)
    return out_b


def generate_one_game_trace_pyr(
    n2: int,
    *,
    seed: int = 0,
    validate: bool = True,
) -> Dict[str, np.ndarray]:
    """Generate one per-game trace for the pyramid teacher.

    Returns a dict with *unstacked* arrays for a single game:
    - field_bits:      (depth+1, n2)
    - gun_bits:        (depth+1, n2)
    - comms_bits:      (depth+1, 1)
    - meas_in_a_bits:  (depth,   n2)
    - meas_out_a_bits: (depth,   n2)
    - meas_in_b_bits:  (depth,   n2)
    - meas_out_b_bits: (depth,   n2)
    - shoot:           (1,)
    """
    s = pyr_sizes(n2)
    rng = _rng(seed)

    # Allocate.
    field_bits = np.zeros((s.depth + 1, n2), dtype=np.float32)
    gun_bits = np.zeros((s.depth + 1, n2), dtype=np.float32)
    comms_bits = np.zeros((s.depth + 1, 1), dtype=np.float32)
    meas_in_a_bits = np.zeros((s.depth, n2), dtype=np.float32)
    meas_out_a_bits = np.zeros((s.depth, n2), dtype=np.float32)
    meas_in_b_bits = np.zeros((s.depth, n2), dtype=np.float32)
    meas_out_b_bits = np.zeros((s.depth, n2), dtype=np.float32)

    # --- Initial state ---
    # Field: random binary bits of length n2.
    field0 = rng.integers(0, 2, size=(n2,), dtype=np.int32).astype(np.float32)
    field_bits[0] = field0

    # Gun: one-hot over [0:n2).
    idx0 = int(rng.integers(0, n2))
    gun0 = np.zeros((n2,), dtype=np.float32)
    gun0[idx0] = 1.0
    gun_bits[0] = gun0

    # Shoot label: whether the chosen cell contains a ship (one-shot hit).
    shoot = np.array([field0[idx0]], dtype=np.float32)

    # --- Model A trace: field reduction + comm generation ---
    field_prefix = field0.copy()
    out_a_prefix_prev = None
    for d in range(s.depth):
        Ld = s.L[d]
        kd = s.k[d]
        fp = field_prefix[:Ld]

        # Measurement input (teacher).
        meas_in_a = teacher_measure_a(fp)
        if meas_in_a.shape != (kd,):
            raise ValueError(f"teacher_measure_a returned shape {meas_in_a.shape}, expected ({kd},).")
        meas_in_a_bits[d] = _pad(meas_in_a, n2)

        # First measurement outcome is prescribed by dataset (uniform SR).
        out_a = rng.integers(0, 2, size=(kd,), dtype=np.int32).astype(np.float32)
        meas_out_a_bits[d] = _pad(out_a, n2)
        out_a_prefix_prev = out_a

        # Next field (teacher combine A).
        next_field = teacher_combine_a(fp, out_a)
        if next_field.shape != (kd,):
            raise ValueError(f"teacher_combine_a returned shape {next_field.shape}, expected ({kd},).")
        field_prefix = _pad(next_field, n2)
        field_bits[d + 1] = field_prefix

    # Comm emitted by Model A is the final reduced field bit.
    comm0 = np.array([field_bits[s.depth][0]], dtype=np.float32)
    comms_bits[0] = comm0

    # --- Model B trace: gun reduction + comm update ---
    gun_prefix = gun0.copy()
    comm = comm0.copy()
    for d in range(s.depth):
        Ld = s.L[d]
        kd = s.k[d]
        gp = gun_prefix[:Ld]

        # Measurement input (teacher).
        meas_in_b = teacher_measure_b(gp)
        if meas_in_b.shape != (kd,):
            raise ValueError(f"teacher_measure_b returned shape {meas_in_b.shape}, expected ({kd},).")
        meas_in_b_bits[d] = _pad(meas_in_b, n2)

        # Second measurement outcome (PR correlation vs A-side outcome).
        # Uses the A-side first outcome at the same level and both measurement inputs.
        out_a = meas_out_a_bits[d, :kd]
        out_b = _pr_rule_out_b(out_a, meas_in_a_bits[d, :kd], meas_in_b)
        meas_out_b_bits[d] = _pad(out_b, n2)

        # Combine-B (teacher). Uses out_b as SR outcome.
        next_gun, next_comm = teacher_combine_b(gp, out_b, comm)
        if next_gun.shape != (kd,):
            raise ValueError(f"teacher_combine_b next_gun shape {next_gun.shape}, expected ({kd},).")
        if next_comm.shape != (1,):
            raise ValueError(f"teacher_combine_b next_comm shape {next_comm.shape}, expected (1,).")

        gun_prefix = _pad(next_gun, n2)
        gun_bits[d + 1] = gun_prefix
        comm = next_comm.astype(np.float32)
        comms_bits[d + 1] = comm

    if validate:
        validate_one_game_trace(
            n2,
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


def validate_one_game_trace(
    n2: int,
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
    """Validate a single-game trace against core acceptance invariants."""
    s = pyr_sizes(n2)

    # Shapes.
    if field_bits.shape != (s.depth + 1, n2):
        raise ValueError(f"field_bits shape {field_bits.shape} != ({s.depth+1},{n2}).")
    if gun_bits.shape != (s.depth + 1, n2):
        raise ValueError(f"gun_bits shape {gun_bits.shape} != ({s.depth+1},{n2}).")
    if comms_bits.shape != (s.depth + 1, 1):
        raise ValueError(f"comms_bits shape {comms_bits.shape} != ({s.depth+1},1).")
    for name, arr in [
        ("meas_in_a_bits", meas_in_a_bits),
        ("meas_out_a_bits", meas_out_a_bits),
        ("meas_in_b_bits", meas_in_b_bits),
        ("meas_out_b_bits", meas_out_b_bits),
    ]:
        if arr.shape != (s.depth, n2):
            raise ValueError(f"{name} shape {arr.shape} != ({s.depth},{n2}).")
    if shoot.shape != (1,):
        raise ValueError(f"shoot shape {shoot.shape} != (1,).")

    # Domains.
    _assert_binary(field_bits, "field_bits")
    _assert_binary(gun_bits, "gun_bits")
    _assert_binary(comms_bits, "comms_bits")
    _assert_binary(meas_in_a_bits, "meas_in_a_bits")
    _assert_binary(meas_out_a_bits, "meas_out_a_bits")
    _assert_binary(meas_in_b_bits, "meas_in_b_bits")
    _assert_binary(meas_out_b_bits, "meas_out_b_bits")
    _assert_binary(shoot.astype(np.float32), "shoot")

    # Gun one-hot per level (within meaningful prefix).
    for d in range(s.depth + 1):
        Ld = n2 if d == 0 else int(n2 // (2**d))
        _assert_one_hot_prefix(gun_bits[d], Ld, f"gun_bits[{d}]")

    # Padding must be zeros beyond the meaningful prefixes.
    for d in range(s.depth):
        Ld = s.L[d]
        kd = s.k[d]
        if np.any(field_bits[d, Ld:] != 0.0):
            raise ValueError(f"field_bits[{d}] has non-zero padding beyond L_d={Ld}.")
        if np.any(gun_bits[d, Ld:] != 0.0):
            raise ValueError(f"gun_bits[{d}] has non-zero padding beyond L_d={Ld}.")
        for name, arr in [
            ("meas_in_a_bits", meas_in_a_bits),
            ("meas_out_a_bits", meas_out_a_bits),
            ("meas_in_b_bits", meas_in_b_bits),
            ("meas_out_b_bits", meas_out_b_bits),
        ]:
            if np.any(arr[d, kd:] != 0.0):
                raise ValueError(f"{name}[{d}] has non-zero padding beyond k_d={kd}.")


def generate_pyr_dataset(
    n2: int,
    num_games: int,
    *,
    seed: int = 0,
    validate: bool = True,
) -> Dict[str, np.ndarray]:
    """Generate a stacked dataset for `num_games` pyramid games."""
    s = pyr_sizes(n2)
    N = int(num_games)
    if N <= 0:
        raise ValueError("num_games must be > 0.")

    field_bits = np.zeros((N, s.depth + 1, n2), dtype=np.float32)
    gun_bits = np.zeros((N, s.depth + 1, n2), dtype=np.float32)
    comms_bits = np.zeros((N, s.depth + 1, 1), dtype=np.float32)
    meas_in_a_bits = np.zeros((N, s.depth, n2), dtype=np.float32)
    meas_out_a_bits = np.zeros((N, s.depth, n2), dtype=np.float32)
    meas_in_b_bits = np.zeros((N, s.depth, n2), dtype=np.float32)
    meas_out_b_bits = np.zeros((N, s.depth, n2), dtype=np.float32)
    shoot = np.zeros((N, 1), dtype=np.float32)

    for i in range(N):
        g = generate_one_game_trace_pyr(n2, seed=seed + i, validate=validate)
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
    """Save a generated dataset dict to a `.npz` file."""
    np.savez_compressed(path, **ds)