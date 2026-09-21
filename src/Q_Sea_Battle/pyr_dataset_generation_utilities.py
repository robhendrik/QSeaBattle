"""QSeaBattle: Pyramid per-game dataset generation utilities.

This module generates *per-game* binary traces in the canonical on-disk storage
format used by QSeaBattle for the "pyramid" task.

Dataset conventions (as implemented)
------------------------------------
- One sample corresponds to one full game (one trace).
- All arrays are stored as float32 containing only binary values {0.0, 1.0}.
- Vectors are dense and right-padded with zeros to a fixed width ``n2``.
- Pyramid sizes are derived from ``n2`` (must be a power of two). At pyramid level
  ``d`` (0-indexed):
    - ``L[d] = n2 / 2**d`` is the meaningful prefix length for field/gun states.
    - ``k[d] = L[d] / 2`` is the meaningful prefix length for measurement inputs
      and outcomes at that level.

Stacked dataset shapes (N games)
-------------------------------
    field_bits:      (N, depth + 1, n2)
    gun_bits:        (N, depth + 1, n2)
    comms_bits:      (N, depth + 1, 1)
    meas_in_a_bits:  (N, depth,     n2)
    meas_out_a_bits: (N, depth,     n2)
    meas_in_b_bits:  (N, depth,     n2)
    meas_out_b_bits: (N, depth,     n2)
    shoot:           (N, 1)

Implementation note
-------------------
This module is intentionally dependency-light (NumPy only) so it can be used in
standalone data generation scripts without TensorFlow.

Notes on "teacher" logic
------------------------
The functions in the "teacher" section implement the authoritative reduction and
combination rules for the pyramid trace. Where external specifications disagree
with behavior, prefer the code here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# --- Self-contained teacher logic (authoritative: follow code over external docstrings) ---
def _pairs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 1D vector into its even- and odd-indexed elements.

    Args:
        x: 1D NumPy array.

    Returns:
        A tuple ``(x[0::2], x[1::2])``.
    """
    return x[0::2], x[1::2]


def teacher_measure_a(field: np.ndarray) -> np.ndarray:
    """Compute A-side measurement inputs from the current field prefix.

    The teacher defines the A measurement input as the pairwise XOR between the
    even-indexed and odd-indexed halves of the current prefix.

    Args:
        field: 1D float array containing binary values {0,1} in its meaningful
            prefix.

    Returns:
        1D float32 array of binary values {0,1} with length ``len(field) / 2``.
    """
    even, odd = _pairs(field)
    return np.logical_xor(even > 0.5, odd > 0.5).astype(np.float32)


def teacher_combine_a(field: np.ndarray, sr_outcome: np.ndarray) -> np.ndarray:
    """Compute the next reduced field prefix from the A-side outcome.

    The next field prefix is defined as:
        next_field = even(field) XOR sr_outcome

    Args:
        field: 1D float array for the current meaningful prefix.
        sr_outcome: 1D float array of binary outcomes (same length as
            ``even(field)``).

    Returns:
        1D float32 array of binary values {0,1} with length ``len(field) / 2``.
    """
    even, _ = _pairs(field)
    return np.logical_xor(even > 0.5, sr_outcome > 0.5).astype(np.float32)


def teacher_measure_b(gun: np.ndarray) -> np.ndarray:
    """Compute B-side measurement inputs from the current gun prefix.

    The teacher defines the B measurement input as:
        (NOT even(gun)) AND odd(gun)

    Args:
        gun: 1D float array containing a one-hot vector in its meaningful prefix.

    Returns:
        1D float32 array of binary values {0,1} with length ``len(gun) / 2``.
    """
    even, odd = _pairs(gun)
    return (np.logical_not(even > 0.5) & (odd > 0.5)).astype(np.float32)


def teacher_combine_b(
    gun: np.ndarray, sr_outcome: np.ndarray, comm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the next reduced gun prefix and updated comm bit.

    - The next gun prefix is pairwise XOR of even/odd entries.
    - The comm bit is updated by XORing the current comm bit with the SR outcome
      at the argmax position of the next gun.

    Args:
        gun: 1D float array containing a one-hot vector in its meaningful prefix.
        sr_outcome: 1D float array of binary outcomes aligned with ``even(gun)``.
        comm: Float32 array with shape (1,) representing the current comm bit.

    Returns:
        Tuple ``(next_gun, next_comm)`` where:
        - next_gun is a 1D float32 binary vector of length ``len(gun) / 2``.
        - next_comm is a float32 array with shape (1,).
    """
    even, odd = _pairs(gun)
    next_gun = (np.logical_xor(even > 0.5, odd > 0.5)).astype(np.float32)
    idx = int(np.argmax(next_gun))
    next_comm = np.array(
        [float((comm[0] > 0.5) ^ (sr_outcome[idx] > 0.5))], dtype=np.float32
    )
    return next_gun, next_comm


@dataclass(frozen=True)
class PyrSizes:
    """Derived pyramid sizes for a given ``n2``.

    Attributes:
        n2: Total padded width of all vectors.
        depth: Number of pyramid reduction steps (equals ``log2(n2)``).
        L: Per-level meaningful prefix widths for field/gun state vectors.
        k: Per-level meaningful prefix widths for measurement vectors.
    """

    n2: int
    depth: int
    L: Tuple[int, ...]  # per level input widths (length depth), L[d] = n2 / 2^d
    k: Tuple[int, ...]  # per level half widths (length depth), k[d] = L[d] / 2


def pyr_sizes(n2: int) -> PyrSizes:
    """Compute derived pyramid sizes from ``n2``.

    Args:
        n2: Total number of field cells (and padded vector width). Must be a power
            of two and >= 2.

    Returns:
        A ``PyrSizes`` instance describing per-level prefix lengths.

    Raises:
        ValueError: If ``n2`` is < 2, not a power of two, or yields an invalid
            final reduction size.
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
    """Create a NumPy RNG instance for a given integer seed."""
    return np.random.default_rng(int(seed))


def _pad(prefix: np.ndarray, n2: int) -> np.ndarray:
    """Right-pad a 1D prefix vector to length ``n2`` with zeros.

    Args:
        prefix: 1D array containing the meaningful prefix values.
        n2: Target length.

    Returns:
        1D float32 array of shape (n2,) where ``out[:len(prefix)] == prefix`` and
        the remainder is zero.
    """
    out = np.zeros((n2,), dtype=np.float32)
    L = int(prefix.shape[0])
    out[:L] = prefix.astype(np.float32)
    return out


def _assert_binary(x: np.ndarray, name: str) -> None:
    """Validate that an array contains only binary float values {0,1}."""
    u = np.unique(x)
    if not np.all((u == 0.0) | (u == 1.0)):
        raise ValueError(f"{name} must be binary {{0,1}} float32; got unique values {u!r}.")


def _assert_one_hot_prefix(x: np.ndarray, L: int, name: str) -> None:
    """Validate that ``x[:L]`` is one-hot (binary and sums to 1)."""
    prefix = x[:L]
    _assert_binary(prefix.astype(np.float32), name)
    s = int(np.sum(prefix))
    if s != 1:
        raise ValueError(f"{name} must be one-hot within [0:{L}); got sum={s}.")


def _pr_rule_out_b(
    out_a: np.ndarray, meas_in_a: np.ndarray, meas_in_b: np.ndarray
) -> np.ndarray:
    """Compute the PR rule for the B-side outcome in replay-mode.

    This implements a PR-assisted shared resource correlation between the A-side
    and B-side outcomes at the same pyramid level. The B-side outcome matches the
    A-side outcome except when both measurement inputs are 1 (p_rule condition),
    in which case the outcome is flipped.

    With the binary convention 1 == "high":
        out_b = out_a XOR (meas_in_a AND meas_in_b)

    Args:
        out_a: 1D float array of binary outcomes from side A.
        meas_in_a: 1D float array of binary measurement inputs for side A.
        meas_in_b: 1D float array of binary measurement inputs for side B.

    Returns:
        1D float32 array of binary outcomes for side B.
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

    Args:
        n2: Total padded width. Must be a power of two and >= 2.
        seed: RNG seed for reproducible trace generation.
        validate: If True, run `validate_one_game_trace` on the result.

    Returns:
        A dict of NumPy arrays for a single game (unstacked):
            - field_bits: (depth + 1, n2)
            - gun_bits: (depth + 1, n2)
            - comms_bits: (depth + 1, 1)
            - meas_in_a_bits: (depth, n2)
            - meas_out_a_bits: (depth, n2)
            - meas_in_b_bits: (depth, n2)
            - meas_out_b_bits: (depth, n2)
            - shoot: (1,)
    """
    s = pyr_sizes(n2)
    rng = _rng(seed)

    # Allocate fixed-size, right-padded trace arrays.
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

        # First measurement outcome is sampled uniformly (SR is stochastic here).
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

        # Second measurement outcome: replay-mode PR rule vs the A-side outcome at
        # the same level, conditioned on both measurement inputs.
        out_a = meas_out_a_bits[d, :kd]
        out_b = _pr_rule_out_b(out_a, meas_in_a_bits[d, :kd], meas_in_b)
        meas_out_b_bits[d] = _pad(out_b, n2)

        # Combine-B (teacher). Uses out_b as the SR outcome.
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
    """Validate a single-game trace against core invariants.

    This function checks:
    - Array shapes match the canonical per-game layout for ``n2``.
    - All values are binary {0,1} (stored as float32).
    - Gun vectors are one-hot within the meaningful prefix at each level.
    - Padding is strictly zero beyond each level's meaningful prefix length.

    Args:
        n2: Total padded width used to derive pyramid sizes.
        field_bits: Per-level field state, including the initial state.
        gun_bits: Per-level gun state, including the initial state.
        comms_bits: Per-level comm bit trace.
        meas_in_a_bits: A-side measurement inputs by level.
        meas_out_a_bits: A-side measurement outcomes by level.
        meas_in_b_bits: B-side measurement inputs by level.
        meas_out_b_bits: B-side measurement outcomes by level.
        shoot: One-shot hit label.

    Raises:
        ValueError: If any invariant is violated.
    """
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

    # Gun must remain one-hot per level within its meaningful prefix.
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
    """Generate a stacked dataset for ``num_games`` pyramid games.

    Args:
        n2: Total padded width. Must be a power of two and >= 2.
        num_games: Number of games (samples) to generate.
        seed: Base seed; each game uses ``seed + i``.
        validate: If True, validate each generated trace.

    Returns:
        A dict of stacked arrays with leading dimension N == ``num_games``.
    """
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
    """Save a generated dataset dict to a compressed ``.npz`` file.

    Args:
        path: Output file path.
        ds: Dataset dict as returned by `generate_pyr_dataset`.
    """
    np.savez_compressed(path, **ds)