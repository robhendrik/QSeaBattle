"""
dataset_conversion_utility.py

Training-view converters for canonical QSeaBattle pyramid datasets.

This module is **pure** and **deterministic**:
- It accepts a canonical dataset dict (loaded from .npz or produced by a generator).
- It returns cropped, representation-converted arrays matching what trainable layers/models expect.

Normative reference: converters_spec.md (cropped outputs to active widths L_d / k_d).
"""

from __future__ import annotations

from typing import Literal, TypedDict, Dict, Tuple, List, Sequence, Any, Union
import numpy as np


TrainRep = Literal["bits", "scaled", "hard_logit"]


class CanonicalPyrDataset(TypedDict):
    field_bits: np.ndarray        # (N, depth+1, n2)
    gun_bits: np.ndarray          # (N, depth+1, n2)
    comms_bits: np.ndarray        # (N, depth+1, 1)
    meas_in_a_bits: np.ndarray    # (N, depth,   n2)
    meas_out_a_bits: np.ndarray   # (N, depth,   n2)
    meas_in_b_bits: np.ndarray    # (N, depth,   n2)
    meas_out_b_bits: np.ndarray   # (N, depth,   n2)
    shoot: np.ndarray             # (N, 1)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def infer_depth_from_dataset(ds: CanonicalPyrDataset) -> int:
    """Return depth = ds["meas_in_a_bits"].shape[1]."""
    return int(ds["meas_in_a_bits"].shape[1])


def infer_n2_from_dataset(ds: CanonicalPyrDataset) -> int:
    """Return n2 = ds["field_bits"].shape[2]."""
    return int(ds["field_bits"].shape[2])


def level_sizes(n2: int, d: int) -> tuple[int, int]:
    """
    Return (L_d, k_d) for level d.

    L_d = n2 // 2**d
    k_d = L_d // 2

    Preconditions:
      - n2 is a power of 2
      - 0 <= d
    """
    if d < 0:
        raise ValueError(f"d must be >= 0, got {d}")
    L_d = n2 // (2 ** d)
    k_d = L_d // 2
    return int(L_d), int(k_d)


def _as_float32(x: np.ndarray) -> np.ndarray:
    if x.dtype != np.float32:
        return x.astype(np.float32, copy=False)
    return x



def apply_rep(
    x_bits: np.ndarray,
    rep: TrainRep,
    *,
    beta: Union[float, Sequence[float]],
) -> np.ndarray:
    """
    Convert a {0,1} float32 array to the requested representation.

    - bits: unchanged
    - scaled: x - 0.5
    - hard_logit: beta * (2x - 1)

    Beta may be:
      - scalar
      - sequence of scalars (mixed-beta mode)

    Mixed-beta behavior (normative):
      1) Deterministic contiguous split along sample axis.
      2) Apply per-chunk beta.
      3) Deterministic round-robin interleave.

    Notes:
      - Input is assumed float32 {0.0,1.0}
      - Output is float32
    """

    x_bits = _as_float32(x_bits)
    N = x_bits.shape[0]

    # ---- Normalize beta ----
    if isinstance(beta, (list, tuple, np.ndarray)):
        betas = [float(b) for b in beta]
        if len(betas) == 0:
            raise ValueError("beta sequence must not be empty.")
    else:
        betas = [float(beta)]

    # ---- No mixed case ----
    if len(betas) == 1:
        b = betas[0]
        if rep == "bits":
            return x_bits
        if rep == "scaled":
            return x_bits - np.float32(0.5)
        if rep == "hard_logit":
            return np.float32(b) * (np.float32(2.0) * x_bits - np.float32(1.0))
        raise ValueError(f"Unknown rep: {rep!r}")

    # ---- Mixed-beta case ----
    K = len(betas)

    # Deterministic contiguous split
    indices = np.array_split(np.arange(N), K)

    converted_chunks = []

    for idx, b in zip(indices, betas):
        chunk = x_bits[idx]

        if rep == "bits":
            out = chunk

        elif rep == "scaled":
            out = chunk - np.float32(0.5)

        elif rep == "hard_logit":
            out = np.float32(b) * (np.float32(2.0) * chunk - np.float32(1.0))

        else:
            raise ValueError(f"Unknown rep: {rep!r}")

        converted_chunks.append(out.astype(np.float32))

    # ---- Deterministic round-robin interleave ----
    max_len = max(len(c) for c in converted_chunks)
    interleaved = []

    for i in range(max_len):
        for c in converted_chunks:
            if i < len(c):
                interleaved.append(c[i])

    return np.stack(interleaved, axis=0)



def _require_keys(ds: dict[str, Any], keys: Sequence[str]) -> None:
    missing = [k for k in keys if k not in ds]
    if missing:
        raise KeyError(f"Dataset missing keys: {missing}")


def _validate_basic_shapes(ds: CanonicalPyrDataset) -> tuple[int, int, int]:
    """
    Validate canonical shape relations lightly and return (N, depth, n2).

    This is not a full acceptance validator; it guards against obvious misuse.
    """
    _require_keys(ds, [
        "field_bits", "gun_bits", "comms_bits",
        "meas_in_a_bits", "meas_out_a_bits", "meas_in_b_bits", "meas_out_b_bits",
        "shoot",
    ])
    field = ds["field_bits"]
    gun = ds["gun_bits"]
    comms = ds["comms_bits"]
    mi_a = ds["meas_in_a_bits"]
    mo_a = ds["meas_out_a_bits"]
    mi_b = ds["meas_in_b_bits"]
    mo_b = ds["meas_out_b_bits"]
    shoot = ds["shoot"]

    if field.ndim != 3:
        raise ValueError(f"field_bits must be (N, depth+1, n2), got {field.shape}")
    N, depth_p1, n2 = field.shape
    if gun.shape != (N, depth_p1, n2):
        raise ValueError(f"gun_bits shape mismatch: expected {(N, depth_p1, n2)}, got {gun.shape}")
    if comms.shape != (N, depth_p1, 1):
        raise ValueError(f"comms_bits shape mismatch: expected {(N, depth_p1, 1)}, got {comms.shape}")

    depth = depth_p1 - 1
    if mi_a.shape != (N, depth, n2):
        raise ValueError(f"meas_in_a_bits shape mismatch: expected {(N, depth, n2)}, got {mi_a.shape}")
    if mo_a.shape != (N, depth, n2):
        raise ValueError(f"meas_out_a_bits shape mismatch: expected {(N, depth, n2)}, got {mo_a.shape}")
    if mi_b.shape != (N, depth, n2):
        raise ValueError(f"meas_in_b_bits shape mismatch: expected {(N, depth, n2)}, got {mi_b.shape}")
    if mo_b.shape != (N, depth, n2):
        raise ValueError(f"meas_out_b_bits shape mismatch: expected {(N, depth, n2)}, got {mo_b.shape}")
    if shoot.shape != (N, 1):
        raise ValueError(f"shoot shape mismatch: expected {(N, 1)}, got {shoot.shape}")

    return int(N), int(depth), int(n2)


def _crop_field_like(x: np.ndarray, L_d: int) -> np.ndarray:
    """Crop last axis to L_d (expects x (..., n2))."""
    return x[..., :L_d]


def _crop_meas_like(x: np.ndarray, k_d: int) -> np.ndarray:
    """Crop last axis to k_d (expects x (..., n2))."""
    return x[..., :k_d]


# ---------------------------------------------------------------------
# private helpers: _normalize_betas, _contiguous_splits, _interleave_round_robin
# ---------------------------------------------------------------------

def _normalize_betas(beta: float | Sequence[float]) -> list[float]:
    # Backward-compat: scalar -> [scalar]
    if isinstance(beta, (int, float, np.floating)):
        return [float(beta)]
    # Sequence[float] -> list[float]
    betas = [float(b) for b in beta]  # may raise TypeError if not iterable; that's fine
    if len(betas) == 0:
        raise ValueError("beta sequence must be non-empty")
    return betas


def _contiguous_splits(N: int, K: int) -> list[tuple[int, int]]:
    """
    Deterministic contiguous splits equivalent to np.array_split(np.arange(N), K),
    represented as [(start,end), ...] with chunk sizes differing by at most 1.
    """
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    base = N // K
    rem = N % K
    ranges: list[tuple[int, int]] = []
    start = 0
    for j in range(K):
        size = base + (1 if j < rem else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def _interleave_round_robin(chunks: list[np.ndarray]) -> np.ndarray:
    """
    Round-robin interleave along axis 0:
      take row 0 of chunk0, row 0 of chunk1, ..., then row 1 of chunk0, ...
    Skips exhausted chunks. Pure/deterministic.
    """
    if not chunks:
        raise ValueError("chunks must be non-empty")
    lens = [int(c.shape[0]) for c in chunks]
    if sum(lens) == 0:
        # Return an empty array with correct shape/dtype inferred from first chunk
        return np.concatenate(chunks, axis=0)

    max_len = max(lens)
    rows: list[np.ndarray] = []
    for t in range(max_len):
        for j, c in enumerate(chunks):
            if t < lens[j]:
                rows.append(c[t])
    return np.stack(rows, axis=0)

# ---------------------------------------------------------------------
# Converters (frozen signatures)
# ---------------------------------------------------------------------

def convert_layer_measure_a(
    ds: CanonicalPyrDataset,
    *,
    rep_x: TrainRep = "scaled",
    rep_y: TrainRep = "hard_logit",
    beta: float | Sequence[float] = 10.0,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Per-level Measure A dataset.

    Returns:
      {d: (X, Y)} for d in [0 .. depth-1]

    Shapes (cropped):
      X: (N, L_d) from field_bits[:, d, :L_d]
      Y: (N, k_d) from meas_in_a_bits[:, d, :k_d]
    """
    N, depth, n2 = _validate_basic_shapes(ds)
    betas = _normalize_betas(beta)
    K = len(betas)

    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    # Backward-compatible fast path: bitwise identical outputs and ordering.
    if K == 1:
        b0 = betas[0]
        for d in range(depth):
            L_d, k_d = level_sizes(n2, d)
            X = apply_rep(_crop_field_like(ds["field_bits"][:, d, :], L_d), rep_x, beta=b0)
            Y = apply_rep(_crop_meas_like(ds["meas_in_a_bits"][:, d, :], k_d), rep_y, beta=b0)
            out[d] = (X, Y)
        return out

    splits = _contiguous_splits(N, K)

    for d in range(depth):
        L_d, k_d = level_sizes(n2, d)
        X_chunks: list[np.ndarray] = []
        Y_chunks: list[np.ndarray] = []
        for j, (s, e) in enumerate(splits):
            bj = betas[j]
            Xj = apply_rep(_crop_field_like(ds["field_bits"][s:e, d, :], L_d), rep_x, beta=bj)
            Yj = apply_rep(_crop_meas_like(ds["meas_in_a_bits"][s:e, d, :], k_d), rep_y, beta=bj)
            X_chunks.append(Xj)
            Y_chunks.append(Yj)

        X = _interleave_round_robin(X_chunks)
        Y = _interleave_round_robin(Y_chunks)
        out[d] = (X, Y)

    return out


def convert_layer_combine_a(
    ds: CanonicalPyrDataset,
    *,
    rep_field: TrainRep = "scaled",
    rep_outcome: TrainRep = "hard_logit",
    rep_target: TrainRep = "hard_logit",
    beta: float | Sequence[float] = 10.0,
) -> Dict[int, Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]]:
    """
    Per-level Combine A dataset.

    Returns:
      {d: ((field_d, out_a_d), field_d1)} for d in [0 .. depth-1]

    Shapes (cropped):
      field_d:  (N, L_d)
      out_a_d:  (N, k_d)
      field_d1: (N, L_{d+1})
    """
    N, depth, n2 = _validate_basic_shapes(ds)
    betas = _normalize_betas(beta)
    K = len(betas)

    out: Dict[int, Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = {}

    # Backward-compatible fast path: bitwise identical outputs and ordering.
    if K == 1:
        b0 = betas[0]
        for d in range(depth):
            L_d, k_d = level_sizes(n2, d)
            L_d1, _ = level_sizes(n2, d + 1)
            field_d = apply_rep(_crop_field_like(ds["field_bits"][:, d, :], L_d), rep_field, beta=b0)
            out_a_d = apply_rep(_crop_meas_like(ds["meas_out_a_bits"][:, d, :], k_d), rep_outcome, beta=b0)
            field_d1 = apply_rep(_crop_field_like(ds["field_bits"][:, d + 1, :], L_d1), rep_target, beta=b0)
            out[d] = ((field_d, out_a_d), field_d1)
        return out

    splits = _contiguous_splits(N, K)

    for d in range(depth):
        L_d, k_d = level_sizes(n2, d)
        L_d1, _ = level_sizes(n2, d + 1)

        field_chunks: list[np.ndarray] = []
        outa_chunks: list[np.ndarray] = []
        target_chunks: list[np.ndarray] = []

        for j, (s, e) in enumerate(splits):
            bj = betas[j]
            field_j = apply_rep(_crop_field_like(ds["field_bits"][s:e, d, :], L_d), rep_field, beta=bj)
            outa_j = apply_rep(_crop_meas_like(ds["meas_out_a_bits"][s:e, d, :], k_d), rep_outcome, beta=bj)
            target_j = apply_rep(_crop_field_like(ds["field_bits"][s:e, d + 1, :], L_d1), rep_target, beta=bj)
            field_chunks.append(field_j)
            outa_chunks.append(outa_j)
            target_chunks.append(target_j)

        field_d = _interleave_round_robin(field_chunks)
        out_a_d = _interleave_round_robin(outa_chunks)
        field_d1 = _interleave_round_robin(target_chunks)

        out[d] = ((field_d, out_a_d), field_d1)

    return out


def convert_layer_measure_b(
    ds: CanonicalPyrDataset,
    *,
    rep_x: TrainRep = "scaled",
    rep_y: TrainRep = "hard_logit",
    beta: float | Sequence[float] = 10.0,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Per-level Measure B dataset.

    Returns:
      {d: (X, Y)} for d in [0 .. depth-1]

    Shapes (cropped):
      X: (N, L_d) from gun_bits[:, d, :L_d]
      Y: (N, k_d) from meas_in_b_bits[:, d, :k_d]
    """
    N, depth, n2 = _validate_basic_shapes(ds)
    betas = _normalize_betas(beta)
    K = len(betas)

    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    # Backward-compatible fast path: bitwise identical outputs and ordering.
    if K == 1:
        b0 = betas[0]
        for d in range(depth):
            L_d, k_d = level_sizes(n2, d)
            X = apply_rep(_crop_field_like(ds["gun_bits"][:, d, :], L_d), rep_x, beta=b0)
            Y = apply_rep(_crop_meas_like(ds["meas_in_b_bits"][:, d, :], k_d), rep_y, beta=b0)
            out[d] = (X, Y)
        return out

    splits = _contiguous_splits(N, K)

    for d in range(depth):
        L_d, k_d = level_sizes(n2, d)
        X_chunks: list[np.ndarray] = []
        Y_chunks: list[np.ndarray] = []
        for j, (s, e) in enumerate(splits):
            bj = betas[j]
            Xj = apply_rep(_crop_field_like(ds["gun_bits"][s:e, d, :], L_d), rep_x, beta=bj)
            Yj = apply_rep(_crop_meas_like(ds["meas_in_b_bits"][s:e, d, :], k_d), rep_y, beta=bj)
            X_chunks.append(Xj)
            Y_chunks.append(Yj)

        X = _interleave_round_robin(X_chunks)
        Y = _interleave_round_robin(Y_chunks)
        out[d] = (X, Y)

    return out


def convert_layer_combine_b(
    ds: CanonicalPyrDataset,
    *,
    rep_gun: TrainRep = "scaled",
    rep_outcome_b: TrainRep = "hard_logit",
    rep_comm_in: TrainRep = "hard_logit",
    rep_gun_next: TrainRep = "hard_logit",
    rep_comm_next: TrainRep = "hard_logit",
    beta: float | Sequence[float] = 10.0,
) -> Dict[int, Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    """
    Per-level Combine B dataset.

    Returns:
      {d: ((gun_d, out_b_d, comm_d), (gun_d1, comm_d1))} for d in [0 .. depth-1]

    Shapes (cropped):
      gun_d:   (N, L_d)
      out_b_d: (N, k_d)
      comm_d:  (N, 1)
      gun_d1:  (N, L_{d+1})
      comm_d1: (N, 1)
    """
    N, depth, n2 = _validate_basic_shapes(ds)
    betas = _normalize_betas(beta)
    K = len(betas)

    out: Dict[int, Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = {}

    # Backward-compatible fast path: bitwise identical outputs and ordering.
    if K == 1:
        b0 = betas[0]
        for d in range(depth):
            L_d, k_d = level_sizes(n2, d)
            L_d1, _ = level_sizes(n2, d + 1)
            gun_d = apply_rep(_crop_field_like(ds["gun_bits"][:, d, :], L_d), rep_gun, beta=b0)
            out_b_d = apply_rep(_crop_meas_like(ds["meas_out_b_bits"][:, d, :], k_d), rep_outcome_b, beta=b0)
            comm_d = apply_rep(ds["comms_bits"][:, d, :], rep_comm_in, beta=b0)  # (N,1)
            gun_d1 = apply_rep(_crop_field_like(ds["gun_bits"][:, d + 1, :], L_d1), rep_gun_next, beta=b0)
            comm_d1 = apply_rep(ds["comms_bits"][:, d + 1, :], rep_comm_next, beta=b0)
            out[d] = ((gun_d, out_b_d, comm_d), (gun_d1, comm_d1))
        return out

    splits = _contiguous_splits(N, K)

    for d in range(depth):
        L_d, k_d = level_sizes(n2, d)
        L_d1, _ = level_sizes(n2, d + 1)

        gun_chunks: list[np.ndarray] = []
        outb_chunks: list[np.ndarray] = []
        comm_chunks: list[np.ndarray] = []
        gun1_chunks: list[np.ndarray] = []
        comm1_chunks: list[np.ndarray] = []

        for j, (s, e) in enumerate(splits):
            bj = betas[j]
            gun_j = apply_rep(_crop_field_like(ds["gun_bits"][s:e, d, :], L_d), rep_gun, beta=bj)
            outb_j = apply_rep(_crop_meas_like(ds["meas_out_b_bits"][s:e, d, :], k_d), rep_outcome_b, beta=bj)
            comm_j = apply_rep(ds["comms_bits"][s:e, d, :], rep_comm_in, beta=bj)  # (n_j,1)
            gun1_j = apply_rep(_crop_field_like(ds["gun_bits"][s:e, d + 1, :], L_d1), rep_gun_next, beta=bj)
            comm1_j = apply_rep(ds["comms_bits"][s:e, d + 1, :], rep_comm_next, beta=bj)

            gun_chunks.append(gun_j)
            outb_chunks.append(outb_j)
            comm_chunks.append(comm_j)
            gun1_chunks.append(gun1_j)
            comm1_chunks.append(comm1_j)

        gun_d = _interleave_round_robin(gun_chunks)
        out_b_d = _interleave_round_robin(outb_chunks)
        comm_d = _interleave_round_robin(comm_chunks)
        gun_d1 = _interleave_round_robin(gun1_chunks)
        comm_d1 = _interleave_round_robin(comm1_chunks)

        out[d] = ((gun_d, out_b_d, comm_d), (gun_d1, comm_d1))

    return out


def convert_internal_model_a(
    ds: CanonicalPyrDataset,
    *,
    rep_field: TrainRep = "scaled",
    rep_comm_target: TrainRep = "hard_logit",
    rep_meas_target: TrainRep = "hard_logit",
    rep_out_target: TrainRep = "hard_logit",
    beta: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Internal Model A supervised view (mandatory lists).

    Returns:
      (field_0, comm_target_final, meas_targets_list, out_targets_list)

    Shapes (cropped):
      field_0:           (N, n2)
      comm_target_final: (N, 1)
      meas_targets_list[d]: (N, k_d)
      out_targets_list[d]:  (N, k_d)
    """
    N, depth, n2 = _validate_basic_shapes(ds)
    field_0 = apply_rep(ds["field_bits"][:, 0, :], rep_field, beta=beta)  # n2
    comm_final = apply_rep(ds["comms_bits"][:, 0, :], rep_comm_target, beta=beta)  # (N,1)
    meas_list: List[np.ndarray] = []
    out_list: List[np.ndarray] = []
    for d in range(depth):
        _, k_d = level_sizes(n2, d)
        meas_list.append(apply_rep(_crop_meas_like(ds["meas_in_a_bits"][:, d, :], k_d), rep_meas_target, beta=beta))
        out_list.append(apply_rep(_crop_meas_like(ds["meas_out_a_bits"][:, d, :], k_d), rep_out_target, beta=beta))
    return field_0, comm_final, meas_list, out_list


def convert_internal_model_b(
    ds: CanonicalPyrDataset,
    *,
    rep_gun: TrainRep = "scaled",
    rep_comm_in: TrainRep = "hard_logit",
    rep_prev_meas: TrainRep = "hard_logit",
    rep_prev_out: TrainRep = "hard_logit",
    rep_shoot_target: TrainRep = "bits",
    rep_meas_b_target: TrainRep = "bits",
    rep_out_b_target: TrainRep = "bits",
    beta: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Internal Model B supervised view (mandatory lists).

    Returns:
      (gun_0, teacher_comm_0, prev_meas_list, prev_out_list, shoot_target)

    Shapes (cropped):
      gun_0: (N, n2)
      teacher_comm_0: (N, 1)
      prev_meas_list[d]: (N, k_d)
      prev_out_list[d]:  (N, k_d)
      shoot_target: (N, 1)
      meas_b_target: (N, k_d)
      out_b_target: (N, k_d)
    """
    N, depth, n2 = _validate_basic_shapes(ds)
    gun_0 = apply_rep(ds["gun_bits"][:, 0, :], rep_gun, beta=beta)  # n2
    teacher_comm_0 = apply_rep(ds["comms_bits"][:, 0, :], rep_comm_in, beta=beta)  # (N,1)
    prev_meas_list: List[np.ndarray] = []
    prev_out_list: List[np.ndarray] = []
    meas_b_list: List[np.ndarray] = []
    out_b_list: List[np.ndarray] = []
    for d in range(depth):
        _, k_d = level_sizes(n2, d)
        prev_meas_list.append(apply_rep(_crop_meas_like(ds["meas_in_a_bits"][:, d, :], k_d), rep_prev_meas, beta=beta))
        prev_out_list.append(apply_rep(_crop_meas_like(ds["meas_out_a_bits"][:, d, :], k_d), rep_prev_out, beta=beta))
        meas_b_list.append(apply_rep(_crop_meas_like(ds["meas_in_b_bits"][:, d, :], k_d), rep_meas_b_target, beta=beta))
        out_b_list.append(apply_rep(_crop_meas_like(ds["meas_out_b_bits"][:, d, :], k_d), rep_out_b_target, beta=beta))
    shoot_target = apply_rep(ds["shoot"][:, :], rep_shoot_target, beta=beta)
    return gun_0, teacher_comm_0, prev_meas_list, prev_out_list, meas_b_list, out_b_list, shoot_target


def convert_full_system(
    ds: CanonicalPyrDataset,
    *,
    rep_field: TrainRep = "scaled",
    rep_gun: TrainRep = "scaled",
    rep_teacher_comm_trace: TrainRep = "bits",
    rep_teacher_meas_a: TrainRep = "bits",
    rep_teacher_out_a: TrainRep = "bits",
    rep_shoot_target: TrainRep = "bits",
    beta: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Full-system training view (A→B), with teacher traces.

    Returns:
      (field_0, gun_0, teacher_comms_trace, teacher_meas_a_list, teacher_out_a_list, shoot_target)

    Shapes:
      field_0: (N, n2)
      gun_0:   (N, n2)
      teacher_comms_trace: (N, depth+1, 1)
      teacher_meas_a_list[d]: (N, k_d)
      teacher_out_a_list[d]:  (N, k_d)
      shoot_target: (N, 1)
    """
    N, depth, n2 = _validate_basic_shapes(ds)
    field_0 = apply_rep(ds["field_bits"][:, 0, :], rep_field, beta=beta)
    gun_0 = apply_rep(ds["gun_bits"][:, 0, :], rep_gun, beta=beta)
    teacher_comms_trace = apply_rep(ds["comms_bits"][:, : depth + 1, :], rep_teacher_comm_trace, beta=beta)
    teacher_meas_list: List[np.ndarray] = []
    teacher_out_list: List[np.ndarray] = []
    for d in range(depth):
        _, k_d = level_sizes(n2, d)
        teacher_meas_list.append(apply_rep(_crop_meas_like(ds["meas_in_a_bits"][:, d, :], k_d), rep_teacher_meas_a, beta=beta))
        teacher_out_list.append(apply_rep(_crop_meas_like(ds["meas_out_a_bits"][:, d, :], k_d), rep_teacher_out_a, beta=beta))
    shoot_target = apply_rep(ds["shoot"][:, :], rep_shoot_target, beta=beta)
    return field_0, gun_0, teacher_comms_trace, teacher_meas_list, teacher_out_list, shoot_target


def convert_all_traces(
    ds: CanonicalPyrDataset,
    *,
    # ---- per-trace representations ----
    rep_field: TrainRep = "bits",
    rep_gun: TrainRep = "bits",
    rep_comms: TrainRep = "bits",
    rep_meas_in_a: TrainRep = "bits",
    rep_meas_out_a: TrainRep = "bits",
    rep_meas_in_b: TrainRep = "bits",
    rep_meas_out_b: TrainRep = "bits",
    rep_shoot: TrainRep = "bits",
    # ---- hardness parameter (supports mixed-beta) ----
    beta: float | Sequence[float] = 10.0,
) -> dict[str, Any]:
    """
    Return a **complete, per-level** converted view of the canonical dataset.

    This is intended for:
      - model-B training diagnostics / ablations (oracle injection)
      - verifying comm shifts (comm[d] vs comm[d+1])
      - comparing any model output to its canonical targets at any depth

    The function is **pure** and **deterministic** and follows the same cropping rules
    as the other converters:
      - state at level d is cropped to L_d
      - measurements at transition d are cropped to k_d

    Mixed-beta behavior:
      - If beta is a sequence, the sample axis is deterministically split and
        round-robin interleaved (same ordering for every returned array).

    Returns a dict with:
      - "N", "depth", "n2"
      - "L" (list[int] of L_d for d=0..depth)
      - "k" (list[int] of k_d for d=0..depth-1)
      - "field"      : list[np.ndarray] length depth+1, each (N, L_d)
      - "gun"        : list[np.ndarray] length depth+1, each (N, L_d)
      - "comms"      : list[np.ndarray] length depth+1, each (N, 1)
      - "meas_in_a"  : list[np.ndarray] length depth,   each (N, k_d)
      - "meas_out_a" : list[np.ndarray] length depth,   each (N, k_d)
      - "meas_in_b"  : list[np.ndarray] length depth,   each (N, k_d)
      - "meas_out_b" : list[np.ndarray] length depth,   each (N, k_d)
      - "shoot"      : np.ndarray (N, 1)

    Notes:
      - For "bits" rep, values remain {0,1}.
      - For "scaled" rep, values are {-0.5, +0.5}.
      - For "hard_logit" rep, values are {-beta, +beta} (or a beta-mixture).
    """

    N, depth, n2 = _validate_basic_shapes(ds)

    # Geometry (including depth level for state)
    L_list: list[int] = []
    k_list: list[int] = []
    for d in range(depth + 1):
        L_d, k_d = level_sizes(n2, d)
        L_list.append(L_d)
        if d < depth:
            k_list.append(k_d)

    # ---- state traces (depth+1) ----
    field_list: list[np.ndarray] = []
    gun_list: list[np.ndarray] = []
    comm_list: list[np.ndarray] = []

    for d in range(depth + 1):
        L_d = L_list[d]
        field_d_bits = _crop_field_like(ds["field_bits"][:, d, :], L_d)
        gun_d_bits = _crop_field_like(ds["gun_bits"][:, d, :], L_d)
        comm_d_bits = ds["comms_bits"][:, d, :]  # (N,1)

        field_list.append(apply_rep(field_d_bits, rep_field, beta=beta))
        gun_list.append(apply_rep(gun_d_bits, rep_gun, beta=beta))
        comm_list.append(apply_rep(comm_d_bits, rep_comms, beta=beta))

    # ---- transition traces (depth) ----
    mi_a_list: list[np.ndarray] = []
    mo_a_list: list[np.ndarray] = []
    mi_b_list: list[np.ndarray] = []
    mo_b_list: list[np.ndarray] = []

    for d in range(depth):
        k_d = k_list[d]
        mi_a_bits = _crop_meas_like(ds["meas_in_a_bits"][:, d, :], k_d)
        mo_a_bits = _crop_meas_like(ds["meas_out_a_bits"][:, d, :], k_d)
        mi_b_bits = _crop_meas_like(ds["meas_in_b_bits"][:, d, :], k_d)
        mo_b_bits = _crop_meas_like(ds["meas_out_b_bits"][:, d, :], k_d)

        mi_a_list.append(apply_rep(mi_a_bits, rep_meas_in_a, beta=beta))
        mo_a_list.append(apply_rep(mo_a_bits, rep_meas_out_a, beta=beta))
        mi_b_list.append(apply_rep(mi_b_bits, rep_meas_in_b, beta=beta))
        mo_b_list.append(apply_rep(mo_b_bits, rep_meas_out_b, beta=beta))

    shoot = apply_rep(ds["shoot"][:, :], rep_shoot, beta=beta)

    return {
        "N": N,
        "depth": depth,
        "n2": n2,
        "L": L_list,
        "k": k_list,
        "field": field_list,
        "gun": gun_list,
        "comms": comm_list,
        "meas_in_a": mi_a_list,
        "meas_out_a": mo_a_list,
        "meas_in_b": mi_b_list,
        "meas_out_b": mo_b_list,
        "shoot": shoot,
    }

# # --------------- OLD ------------
#
# Remove after verifying new versions above are correct and used.
#
# def convert_layer_measure_a(
#     ds: CanonicalPyrDataset,
#     *,
#     rep_x: TrainRep = "scaled",
#     rep_y: TrainRep = "hard_logit",
#     beta: float = 10.0,
# ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
#     """
#     Per-level Measure A dataset.

#     Returns:
#       {d: (X, Y)} for d in [0 .. depth-1]

#     Shapes (cropped):
#       X: (N, L_d) from field_bits[:, d, :L_d]
#       Y: (N, k_d) from meas_in_a_bits[:, d, :k_d]
#     """
#     N, depth, n2 = _validate_basic_shapes(ds)
#     out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
#     for d in range(depth):
#         L_d, k_d = level_sizes(n2, d)
#         X = apply_rep(_crop_field_like(ds["field_bits"][:, d, :], L_d), rep_x, beta=beta)
#         Y = apply_rep(_crop_meas_like(ds["meas_in_a_bits"][:, d, :], k_d), rep_y, beta=beta)
#         out[d] = (X, Y)
#     return out


# def convert_layer_combine_a(
#     ds: CanonicalPyrDataset,
#     *,
#     rep_field: TrainRep = "scaled",
#     rep_outcome: TrainRep = "hard_logit",
#     rep_target: TrainRep = "hard_logit",
#     beta: float = 10.0,
# ) -> Dict[int, Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]]:
#     """
#     Per-level Combine A dataset.

#     Returns:
#       {d: ((field_d, out_a_d), field_d1)} for d in [0 .. depth-1]

#     Shapes (cropped):
#       field_d:  (N, L_d)
#       out_a_d:  (N, k_d)
#       field_d1: (N, L_{d+1})
#     """
#     N, depth, n2 = _validate_basic_shapes(ds)
#     out: Dict[int, Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = {}
#     for d in range(depth):
#         L_d, k_d = level_sizes(n2, d)
#         L_d1, _ = level_sizes(n2, d + 1)
#         field_d = apply_rep(_crop_field_like(ds["field_bits"][:, d, :], L_d), rep_field, beta=beta)
#         out_a_d = apply_rep(_crop_meas_like(ds["meas_out_a_bits"][:, d, :], k_d), rep_outcome, beta=beta)
#         field_d1 = apply_rep(_crop_field_like(ds["field_bits"][:, d + 1, :], L_d1), rep_target, beta=beta)
#         out[d] = ((field_d, out_a_d), field_d1)
#     return out


# def convert_layer_measure_b(
#     ds: CanonicalPyrDataset,
#     *,
#     rep_x: TrainRep = "scaled",
#     rep_y: TrainRep = "hard_logit",
#     beta: float = 10.0,
# ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
#     """
#     Per-level Measure B dataset.

#     Returns:
#       {d: (X, Y)} for d in [0 .. depth-1]

#     Shapes (cropped):
#       X: (N, L_d) from gun_bits[:, d, :L_d]
#       Y: (N, k_d) from meas_in_b_bits[:, d, :k_d]
#     """
#     N, depth, n2 = _validate_basic_shapes(ds)
#     out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
#     for d in range(depth):
#         L_d, k_d = level_sizes(n2, d)
#         X = apply_rep(_crop_field_like(ds["gun_bits"][:, d, :], L_d), rep_x, beta=beta)
#         Y = apply_rep(_crop_meas_like(ds["meas_in_b_bits"][:, d, :], k_d), rep_y, beta=beta)
#         out[d] = (X, Y)
#     return out


# def convert_layer_combine_b(
#     ds: CanonicalPyrDataset,
#     *,
#     rep_gun: TrainRep = "scaled",
#     rep_outcome_b: TrainRep = "hard_logit",
#     rep_comm_in: TrainRep = "hard_logit",
#     rep_gun_next: TrainRep = "hard_logit",
#     rep_comm_next: TrainRep = "hard_logit",
#     beta: float = 10.0,
# ) -> Dict[int, Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
#     """
#     Per-level Combine B dataset.

#     Returns:
#       {d: ((gun_d, out_b_d, comm_d), (gun_d1, comm_d1))} for d in [0 .. depth-1]

#     Shapes (cropped):
#       gun_d:   (N, L_d)
#       out_b_d: (N, k_d)
#       comm_d:  (N, 1)
#       gun_d1:  (N, L_{d+1})
#       comm_d1: (N, 1)
#     """
#     N, depth, n2 = _validate_basic_shapes(ds)
#     out: Dict[int, Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = {}
#     for d in range(depth):
#         L_d, k_d = level_sizes(n2, d)
#         L_d1, _ = level_sizes(n2, d + 1)
#         gun_d = apply_rep(_crop_field_like(ds["gun_bits"][:, d, :], L_d), rep_gun, beta=beta)
#         out_b_d = apply_rep(_crop_meas_like(ds["meas_out_b_bits"][:, d, :], k_d), rep_outcome_b, beta=beta)
#         comm_d = apply_rep(ds["comms_bits"][:, d, :], rep_comm_in, beta=beta)  # (N,1)
#         gun_d1 = apply_rep(_crop_field_like(ds["gun_bits"][:, d + 1, :], L_d1), rep_gun_next, beta=beta)
#         comm_d1 = apply_rep(ds["comms_bits"][:, d + 1, :], rep_comm_next, beta=beta)
#         out[d] = ((gun_d, out_b_d, comm_d), (gun_d1, comm_d1))
#     return out

