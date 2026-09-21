"""Utilities for converting canonical *linear* datasets into training views.

This module provides small conversion helpers that reshape and re-encode a
canonical linear dataset into the input/target tuples expected by individual
training stages (layer-wise training, internal model training, or full-system
training).

A "linear" dataset in this codebase has:
- Fixed depth=1, represented as depth+1 == 2 along axis 1 for player-held traces
  (e.g., ``field_bits`` and ``comms_bits``).
- No width-halving; spatial width is constant and represented by ``n2``.

Representations (``TrainRep``) control how bit-valued arrays are presented to
models:
- ``"bits"``: values remain in {0, 1} (float32).
- ``"scaled"``: values are shifted to {-0.5, +0.5}.
- ``"hard_logit"``: values are mapped to logits in {-beta, +beta}; the logical
  bit value is encoded by the sign of the logit.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Sequence, Tuple, TypedDict, Union

import numpy as np


TrainRep = Literal["bits", "scaled", "hard_logit"]


class CanonicalLinDataset(TypedDict):
    """Typed view of the expected canonical linear dataset dictionary.

    All arrays are NumPy ndarrays. Shapes use:
    - N: number of examples
    - n2: flattened board width (implementation-specific)
    - m: number of communication bits

    Attributes:
        field_bits: (N, 2, n2) player A's field trace; axis 1 is time (depth+1).
        gun_bits: (N, 2, n2) player B's gun trace; axis 1 is time (depth+1).
        comms_bits: (N, 2, m) communication trace; axis 1 is time (depth+1).
        meas_in_a_bits: (N, 1, n2) A's measurement input at depth=1.
        meas_out_a_bits: (N, 1, n2) A's measurement outcome at depth=1.
        meas_in_b_bits: (N, 1, n2) B's measurement input at depth=1.
        meas_out_b_bits: (N, 1, n2) B's measurement outcome at depth=1.
        shoot: (N, 1) shoot decision bit.
    """

    field_bits: np.ndarray        # (N, 2, n2)
    gun_bits: np.ndarray          # (N, 2, n2)
    comms_bits: np.ndarray        # (N, 2, m)
    meas_in_a_bits: np.ndarray    # (N, 1, n2)
    meas_out_a_bits: np.ndarray   # (N, 1, n2)
    meas_in_b_bits: np.ndarray    # (N, 1, n2)
    meas_out_b_bits: np.ndarray   # (N, 1, n2)
    shoot: np.ndarray             # (N, 1)


def _require_keys(ds: Dict[str, Any], keys: Sequence[str]) -> None:
    """Validate that required keys are present in a dataset mapping.

    Args:
        ds: Dataset mapping to validate.
        keys: Required key names.

    Raises:
        KeyError: If any key is missing.
    """
    missing = [k for k in keys if k not in ds]
    if missing:
        raise KeyError(f"Dataset missing keys: {missing}")


def _as_float32(x: np.ndarray) -> np.ndarray:
    """Return ``x`` as float32 without copying when possible."""
    if x.dtype != np.float32:
        return x.astype(np.float32, copy=False)
    return x


def apply_rep(
    x_bits: np.ndarray,
    rep: TrainRep,
    *,
    beta: Union[float, Sequence[float]],
) -> np.ndarray:
    """Convert a bit-valued array into a training representation.

    The input is assumed to be bit-like (0/1). The returned array is float32.

    Notes:
        ``beta`` is accepted as a scalar or a 1-element sequence for API
        compatibility with other converters. Linear converters currently require
        a scalar beta.

    Args:
        x_bits: Input array containing bit values.
        rep: Output representation.
        beta: Logit magnitude used when ``rep == "hard_logit"``.

    Returns:
        A float32 NumPy array with the same shape as ``x_bits``.

    Raises:
        ValueError: If ``rep`` is unknown or if ``beta`` is not scalar.
    """
    x_bits = _as_float32(x_bits)
    if isinstance(beta, (list, tuple, np.ndarray)):
        betas = [float(b) for b in beta]
        if len(betas) != 1:
            raise ValueError("Linear converters currently support scalar beta only.")
        b = betas[0]
    else:
        b = float(beta)

    if rep == "bits":
        return x_bits
    if rep == "scaled":
        return x_bits - np.float32(0.5)
    if rep == "hard_logit":
        # Map 0 -> -beta and 1 -> +beta. Sign encodes the logical bit value.
        return np.float32(b) * (np.float32(2.0) * x_bits - np.float32(1.0))
    raise ValueError(f"Unknown rep: {rep!r}")


def _validate_basic_shapes(ds: CanonicalLinDataset) -> tuple[int, int, int]:
    """Validate required keys and basic array shapes for linear datasets.

    Args:
        ds: Canonical linear dataset.

    Returns:
        Tuple of (N, n2, m), derived from the dataset arrays.

    Raises:
        KeyError: If required keys are missing.
        ValueError: If any array shape is inconsistent with the linear format.
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
        raise ValueError(f"field_bits must be (N,2,n2), got {field.shape}")
    N, depth_p1, n2 = field.shape
    if depth_p1 != 2:
        raise ValueError(f"Linear dataset depth+1 must be 2, got {depth_p1}.")

    if gun.shape != (N, 2, n2):
        raise ValueError(f"gun_bits shape mismatch: expected {(N, 2, n2)}, got {gun.shape}")
    if comms.ndim != 3 or comms.shape[0] != N or comms.shape[1] != 2:
        raise ValueError(f"comms_bits shape mismatch: expected (N,2,m), got {comms.shape}")
    m = int(comms.shape[2])

    if mi_a.shape != (N, 1, n2) or mo_a.shape != (N, 1, n2):
        raise ValueError("A measurement arrays must have shape (N,1,n2).")
    if mi_b.shape != (N, 1, n2) or mo_b.shape != (N, 1, n2):
        raise ValueError("B measurement arrays must have shape (N,1,n2).")
    if shoot.shape != (N, 1):
        raise ValueError(f"shoot shape mismatch: expected {(N,1)}, got {shoot.shape}")

    return int(N), int(n2), int(m)


def convert_layer_measure_a(
    ds: CanonicalLinDataset,
    *,
    rep_x: TrainRep = "scaled",
    rep_y: TrainRep = "hard_logit",
    beta: float = 10.0,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Create the training view for A's measurement layer.

    Uses the t=0 slice of the field trace as input and A's measurement input as
    the supervision target.

    Args:
        ds: Canonical linear dataset.
        rep_x: Representation for the field input.
        rep_y: Representation for the measurement target.
        beta: Logit magnitude used for ``"hard_logit"`` representations.

    Returns:
        Mapping from layer index to (X, Y), where X and Y are NumPy arrays.
    """
    _validate_basic_shapes(ds)
    X = apply_rep(ds["field_bits"][:, 0, :], rep_x, beta=beta)
    Y = apply_rep(ds["meas_in_a_bits"][:, 0, :], rep_y, beta=beta)
    return {0: (X, Y)}


def convert_layer_combine_a(
    ds: CanonicalLinDataset,
    *,
    rep_outcome: TrainRep = "hard_logit",
    rep_target: TrainRep = "hard_logit",
    beta: float = 10.0,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Create the training view for A's combine layer.

    Trains communication bits as a function of A's measurement outcome.

    Args:
        ds: Canonical linear dataset.
        rep_outcome: Representation for A's measurement outcome (input).
        rep_target: Representation for the communication target.
        beta: Logit magnitude used for ``"hard_logit"`` representations.

    Returns:
        Mapping from layer index to (outcome_a, comm_target).
    """
    _validate_basic_shapes(ds)
    out_a = apply_rep(ds["meas_out_a_bits"][:, 0, :], rep_outcome, beta=beta)
    comm = apply_rep(ds["comms_bits"][:, 0, :], rep_target, beta=beta)
    return {0: (out_a, comm)}


def convert_layer_measure_b(
    ds: CanonicalLinDataset,
    *,
    rep_x: TrainRep = "scaled",
    rep_y: TrainRep = "hard_logit",
    beta: float = 10.0,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Create the training view for B's measurement layer.

    Uses the t=0 slice of the gun trace as input and B's measurement input as
    the supervision target.

    Args:
        ds: Canonical linear dataset.
        rep_x: Representation for the gun input.
        rep_y: Representation for the measurement target.
        beta: Logit magnitude used for ``"hard_logit"`` representations.

    Returns:
        Mapping from layer index to (X, Y).
    """
    _validate_basic_shapes(ds)
    X = apply_rep(ds["gun_bits"][:, 0, :], rep_x, beta=beta)
    Y = apply_rep(ds["meas_in_b_bits"][:, 0, :], rep_y, beta=beta)
    return {0: (X, Y)}


def convert_layer_combine_b(
    ds: CanonicalLinDataset,
    *,
    rep_outcome_b: TrainRep = "hard_logit",
    rep_comm_in: TrainRep = "hard_logit",
    rep_shoot: TrainRep = "hard_logit",
    beta: float = 10.0,
) -> Dict[int, Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]]:
    """Create the training view for B's combine layer.

    The input is a tuple of (B measurement outcome, comms-in). The target is the
    shoot bit.

    Args:
        ds: Canonical linear dataset.
        rep_outcome_b: Representation for B's measurement outcome.
        rep_comm_in: Representation for the communication input to B.
        rep_shoot: Representation for the shoot target.
        beta: Logit magnitude used for ``"hard_logit"`` representations.

    Returns:
        Mapping from layer index to ((outcome_b, comm_in), shoot_target).
    """
    _validate_basic_shapes(ds)
    out_b = apply_rep(ds["meas_out_b_bits"][:, 0, :], rep_outcome_b, beta=beta)
    comm = apply_rep(ds["comms_bits"][:, 0, :], rep_comm_in, beta=beta)
    shoot = apply_rep(ds["shoot"][:, :], rep_shoot, beta=beta)
    return {0: ((out_b, comm), shoot)}


def convert_internal_model_a(
    ds: CanonicalLinDataset,
    *,
    rep_field: TrainRep = "scaled",
    rep_comm_target: TrainRep = "hard_logit",
    rep_meas_target: TrainRep = "hard_logit",
    rep_out_target: TrainRep = "hard_logit",
    beta: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Create inputs/targets for training A's internal model.

    Returns a single-step (depth=1) view, with list-wrapped measurement/outcome
    targets to match interfaces that expect a per-depth list.

    Args:
        ds: Canonical linear dataset.
        rep_field: Representation for the field input.
        rep_comm_target: Representation for the communication target.
        rep_meas_target: Representation for the measurement-input target.
        rep_out_target: Representation for the measurement-outcome target.
        beta: Logit magnitude used for ``"hard_logit"`` representations.

    Returns:
        Tuple of (field_0, comm_0, meas_list, out_list).
    """
    _validate_basic_shapes(ds)
    field_0 = apply_rep(ds["field_bits"][:, 0, :], rep_field, beta=beta)
    comm_0 = apply_rep(ds["comms_bits"][:, 0, :], rep_comm_target, beta=beta)
    meas_list = [apply_rep(ds["meas_in_a_bits"][:, 0, :], rep_meas_target, beta=beta)]
    out_list = [apply_rep(ds["meas_out_a_bits"][:, 0, :], rep_out_target, beta=beta)]
    return field_0, comm_0, meas_list, out_list


def convert_internal_model_b(
    ds: CanonicalLinDataset,
    *,
    rep_gun: TrainRep = "scaled",
    rep_comm_in: TrainRep = "hard_logit",
    rep_prev_meas: TrainRep = "hard_logit",
    rep_prev_out: TrainRep = "hard_logit",
    rep_shoot_target: TrainRep = "bits",
    rep_meas_b_target: TrainRep = "bits",
    rep_out_b_target: TrainRep = "bits",
    beta: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray]:
    """Create inputs/targets for training B's internal model.

    B conditions on its local state (gun), the received comm bits, and A's
    previous-step measurement traces. Targets include B's own measurement traces
    and the shoot decision.

    Args:
        ds: Canonical linear dataset.
        rep_gun: Representation for the gun input.
        rep_comm_in: Representation for the communication input to B.
        rep_prev_meas: Representation for A's previous measurement input.
        rep_prev_out: Representation for A's previous measurement outcome.
        rep_shoot_target: Representation for the shoot target.
        rep_meas_b_target: Representation for B's measurement-input target.
        rep_out_b_target: Representation for B's measurement-outcome target.
        beta: Logit magnitude used for ``"hard_logit"`` representations.

    Returns:
        Tuple of (gun_0, comm_0, prev_meas_list, prev_out_list,
        meas_b_list, out_b_list, shoot_target).
    """
    _validate_basic_shapes(ds)
    gun_0 = apply_rep(ds["gun_bits"][:, 0, :], rep_gun, beta=beta)
    comm_0 = apply_rep(ds["comms_bits"][:, 0, :], rep_comm_in, beta=beta)

    prev_meas_list = [apply_rep(ds["meas_in_a_bits"][:, 0, :], rep_prev_meas, beta=beta)]
    prev_out_list = [apply_rep(ds["meas_out_a_bits"][:, 0, :], rep_prev_out, beta=beta)]

    meas_b_list = [apply_rep(ds["meas_in_b_bits"][:, 0, :], rep_meas_b_target, beta=beta)]
    out_b_list = [apply_rep(ds["meas_out_b_bits"][:, 0, :], rep_out_b_target, beta=beta)]

    shoot_target = apply_rep(ds["shoot"][:, :], rep_shoot_target, beta=beta)
    return gun_0, comm_0, prev_meas_list, prev_out_list, meas_b_list, out_b_list, shoot_target


def convert_full_system(
    ds: CanonicalLinDataset,
    *,
    rep_field: TrainRep = "scaled",
    rep_gun: TrainRep = "scaled",
    rep_teacher_comm_trace: TrainRep = "bits",
    rep_teacher_meas_a: TrainRep = "bits",
    rep_teacher_out_a: TrainRep = "bits",
    rep_shoot_target: TrainRep = "bits",
    beta: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], np.ndarray]:
    """Create the training view for full-system training with teacher traces.

    This view provides:
    - Initial field and gun inputs at t=0
    - A full communication trace over axis 1 (depth+1 == 2)
    - Teacher-forced A measurement input/outcome for the single depth step
    - The shoot target

    Args:
        ds: Canonical linear dataset.
        rep_field: Representation for the initial field input.
        rep_gun: Representation for the initial gun input.
        rep_teacher_comm_trace: Representation for the teacher comm trace.
        rep_teacher_meas_a: Representation for teacher A measurement input.
        rep_teacher_out_a: Representation for teacher A measurement outcome.
        rep_shoot_target: Representation for the shoot target.
        beta: Logit magnitude used for ``"hard_logit"`` representations.

    Returns:
        Tuple of (field_0, gun_0, comm_trace, meas_a_list, out_a_list, shoot_target).
    """
    _validate_basic_shapes(ds)
    field_0 = apply_rep(ds["field_bits"][:, 0, :], rep_field, beta=beta)
    gun_0 = apply_rep(ds["gun_bits"][:, 0, :], rep_gun, beta=beta)
    comm_trace = apply_rep(ds["comms_bits"][:, :, :], rep_teacher_comm_trace, beta=beta)
    meas_a = [apply_rep(ds["meas_in_a_bits"][:, 0, :], rep_teacher_meas_a, beta=beta)]
    out_a = [apply_rep(ds["meas_out_a_bits"][:, 0, :], rep_teacher_out_a, beta=beta)]
    shoot_target = apply_rep(ds["shoot"][:, :], rep_shoot_target, beta=beta)
    return field_0, gun_0, comm_trace, meas_a, out_a, shoot_target