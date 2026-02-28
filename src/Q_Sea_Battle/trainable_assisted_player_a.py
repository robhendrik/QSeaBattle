
"""Trainable assisted Player A implementation.

Player A computes communication bits from the field using a trainable model.
It also stores intermediate "previous" tensors on its parent (TrainableAssistedPlayers)
so Player B can consume them.

See design document for the exact decide/log-prob/previous contract.

Author: Rob Hendriks
Package: Q_Sea_Battle
Version: 0.1
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import tensorflow as tf

from .lin_trainable_assisted_model_a import LinTrainableAssistedModelA
from .gameplay_adapters import GameplayModelAAdapter


def bernoulli_log_prob_from_logits(logits: tf.Tensor, actions01: tf.Tensor) -> tf.Tensor:
    """Compute log P(actions|logits) for independent Bernoulli bits.

    Args:
        logits: Tensor of shape (..., M)
        actions01: Tensor of shape (..., M), values 0/1

    Returns:
        Tensor of shape (...) with summed log-prob over last dimension.
    """
    actions01 = tf.cast(actions01, tf.float32)
    log_p1 = -tf.nn.softplus(-logits)  # log(sigmoid(logit))
    log_p0 = -tf.nn.softplus(logits)   # log(1 - sigmoid(logit))
    return tf.reduce_sum(actions01 * log_p1 + (1.0 - actions01) * log_p0, axis=-1)

try:
    from .players import PlayerA  # type: ignore
except Exception:  # pragma: no cover
    class PlayerA:
        """Fallback PlayerA base class."""

def _warn_if_not_binary_list(
        name: str,
        xs: Any,
        *,
        atol: float = 1e-6,
    ) -> None:
        """Warn if a list of tensors/arrays is not binary {0,1}.

        This is a gameplay-safety check only. It MUST NOT modify data.
        """
        if not isinstance(xs, (list, tuple)):
            return

        for i, x in enumerate(xs):
            if isinstance(x, tf.Tensor):
                v = x.numpy()
            elif isinstance(x, np.ndarray):
                v = x
            else:
                continue  # unknown type, ignore

            # Flatten for simplicity
            v = np.asarray(v).ravel()

            if v.size == 0:
                continue

            # Check if values are close to 0 or 1
            is_binary = np.all(
                (np.abs(v - 0.0) <= atol) | (np.abs(v - 1.0) <= atol)
            )

            if not is_binary:
                print(
                    f"[WARNING][Gameplay] {name}[{i}] contains non-binary values. "
                    f"This likely means logits or relaxed values leaked past the adapter. "
                    f"min={v.min():.3f}, max={v.max():.3f}"
                )

def _as_f32(x: Any) -> tf.Tensor:
    """Convert to tf.float32 tensor."""
    return tf.convert_to_tensor(x, dtype=tf.float32)

def _ensure_rank2(x: tf.Tensor, name: str) -> tf.Tensor:
    """
    Ensure tensor is rank-2 (B, D). If rank-1 (D,), add batch dim.
    """
    x = _as_f32(x)
    if x.shape.rank == 1:
        x = x[None, :]
    tf.debugging.assert_rank(x, 2, message=f"{name} must be rank-2 (B,D) (or rank-1 D)")
    return x

class TrainableAssistedPlayerA(PlayerA):
    """Player A wrapper around LinTrainableAssistedModelA.

    Public attributes:
        model_a: LinTrainableAssistedModelA
        parent: TrainableAssistedPlayers (set by TrainableAssistedPlayers.players())
        last_logprob_comm: float | None
        explore: bool (False=greedy, True=sample)
    """

    def __init__(self, game_layout: Any, model_a: LinTrainableAssistedModelA) -> None:
        self.game_layout = game_layout
        self.model_a = model_a
        self.parent: Any | None = None
        self.last_logprob_comm: float | None = None
        self.explore: bool = False

    def decide(self, field: np.ndarray, supp: Any | None = None, explore: bool | None = None) -> np.ndarray:
        """Decide communication bits based on the field.

        Backward compatibility:
        - If model_a is a GameplayModelAAdapter, it already returns comm_bits (and meas/out bits).
        - Otherwise, model_a returns comm_logits and we sample/threshold here (legacy behavior).

        Args:
            field: 1D array of ints, length n2, values in {0,1}.
            supp: Ignored.
            explore: Optional override of self.explore.

        Returns:
            1D NumPy array of shape (m,), dtype int32, values {0,1}.
        """
        del supp
        do_explore = self.explore if explore is None else bool(explore)

        n2 = int(getattr(self.game_layout, "field_size")) ** 2
        m = int(getattr(self.game_layout, "comms_size"))

        field = np.asarray(field)
        if field.shape != (n2,):
            raise ValueError(f"field must have shape ({n2},), got {field.shape}")
        if not np.all((field == 0) | (field == 1)):
            raise ValueError("field must contain only 0/1")

        field_batch = tf.convert_to_tensor(field[None, :], dtype=tf.float32)  # (1, n2)

        if isinstance(self.model_a, GameplayModelAAdapter):
            # Adapter path: already returns bits at boundary.
            comm_bits_tf, meas_list, out_list, comm_logits = self.model_a(field_batch, 
                                                             explore=do_explore,
                                                             return_comm_logits=True)  # comm_bits_tf expected shape (1, m)

            # Normalize comm bits to shape (1, m) and dtype int32 for consistency.
            comm_bits_tf = _ensure_rank2(comm_bits_tf, "comm_bits")  # (1, m) expected
            # If adapter returns float bits, cast to int32.
            comm_bits_tf = tf.cast(comm_bits_tf, tf.int32)

            logp = bernoulli_log_prob_from_logits(comm_logits, tf.cast(comm_bits_tf, tf.float32))
            self.last_logprob_comm = float(logp.numpy()[0]) 

        else:
            # Legacy path: model returns logits; we sample/threshold here.
            comm_logits, meas_list, out_list = self.model_a.compute_with_internal(field_batch)
            comm_logits = _ensure_rank2(comm_logits, "comm_logits")  # (1, m)

            # Validate width if statically known.
            if comm_logits.shape.rank == 2 and comm_logits.shape[1] is not None:
                if int(comm_logits.shape[1]) != m:
                    raise ValueError(f"comm_logits must have width m={m}, got {int(comm_logits.shape[1])}")

            comm_probs = tf.sigmoid(comm_logits)  # (1, m)

            if do_explore:
                # Sample Bernoulli bits per batch element.
                rnd = tf.random.uniform(shape=tf.shape(comm_probs), dtype=tf.float32)  # (1, m)
                comm_bits_tf = tf.cast(rnd < comm_probs, tf.int32)  # (1, m)
            else:
                comm_bits_tf = tf.cast(comm_probs >= 0.5, tf.int32)  # (1, m)

            # Log-prob under independent Bernoulli with logits.
            logp = bernoulli_log_prob_from_logits(comm_logits, tf.cast(comm_bits_tf, tf.float32))
            self.last_logprob_comm = float(logp.numpy()[0])

        # Gameplay safety check (non-breaking)
        _warn_if_not_binary_list("meas_list", meas_list)
        _warn_if_not_binary_list("out_list", out_list)

        # Store prev tensors on the parent
        if self.parent is not None:
            self.parent.previous = (meas_list, out_list)

        # Return 1D (m,) int32 as promised by docstring.
        comm_bits_np = comm_bits_tf.numpy().astype(np.int32)  # (1, m)
        if comm_bits_np.shape != (1, m):
            # In case some implementation returns (m,) or other equivalent
            comm_bits_np = comm_bits_np.reshape(1, -1)
        return comm_bits_np[0]

    def get_log_prob(self) -> float:
        """Return log-probability of last taken communication action."""
        if self.last_logprob_comm is None:
            raise RuntimeError("No log-prob available: decide() has not been called since reset().")
        return float(self.last_logprob_comm)

    def get_prev(self) -> Any | None:
        """Return parent.previous if available.

        Returns:
            (meas_list, out_list) or None if unavailable.
        """
        if self.parent is None or getattr(self.parent, "previous", None) is None:
            # Non-blocking; caller can handle None.
            return None
        return self.parent.previous

    def reset(self) -> None:
        """Reset internal state."""
        self.last_logprob_comm = None
