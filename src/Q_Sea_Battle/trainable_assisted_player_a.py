"""Trainable assisted Player A implementation.

This module implements the "Player A" side of the PR-assisted shared resource (SR)
interface for QSeaBattle.

Player A observes the local field and produces communication bits at the player/game
boundary. In addition to the boundary communication, Player A also produces
intermediate tensors ("previous") that Player B will consume on the next step. Those
intermediates are stored on the parent container (typically `TrainableAssistedPlayers`)
under `parent.previous`.

The exact contract for `decide()`, `get_log_prob()`, and the stored `previous` payload
is defined in the project's design documentation.

Notes:
    * Model interfaces use logits internally. Logical bit values represented as logits
      are determined by the logit sign.
    * Two code paths are supported for backward compatibility:
        - `GameplayModelAAdapter`: returns boundary bits directly (and optionally logits).
        - Legacy model: returns communication logits and this class performs sampling
          (stochastic) or thresholding (greedy).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import tensorflow as tf

from .lin_trainable_assisted_model_a import LinTrainableAssistedModelA
from .gameplay_adapters import GameplayModelAAdapter


def bernoulli_log_prob_from_logits(logits: tf.Tensor, actions01: tf.Tensor) -> tf.Tensor:
    """Compute log-probability under independent Bernoulli bits parameterized by logits.

    This helper interprets each element in the last dimension as an independent
    Bernoulli random variable whose probability is `sigmoid(logit)`.

    Args:
        logits: Logits tensor with shape (..., M).
        actions01: Tensor with shape (..., M) containing 0/1 bit values.

    Returns:
        Tensor with shape (...) containing the sum of per-bit log-probabilities over
        the last dimension.
    """
    actions01 = tf.cast(actions01, tf.float32)
    log_p1 = -tf.nn.softplus(-logits)  # log(sigmoid(logit))
    log_p0 = -tf.nn.softplus(logits)  # log(1 - sigmoid(logit))
    return tf.reduce_sum(actions01 * log_p1 + (1.0 - actions01) * log_p0, axis=-1)


try:
    from .players import PlayerA  # type: ignore
except Exception:  # pragma: no cover

    class PlayerA:
        """Fallback PlayerA base class.

        Used only when the runtime import of the real gameplay `PlayerA` fails (e.g.,
        during isolated documentation or tooling runs).
        """


def _warn_if_not_binary_list(
    name: str,
    xs: Any,
    *,
    atol: float = 1e-6,
) -> None:
    """Print a warning if a list/tuple contains non-binary arrays/tensors.

    This is a gameplay-safety diagnostic intended to catch cases where relaxed values
    (e.g., logits or probabilities) leak past an adapter that should output boundary
    bits. The function must not modify inputs.

    Args:
        name: Label used in the warning message (typically the variable name).
        xs: A list/tuple of `tf.Tensor`/`np.ndarray` objects to validate. Unknown
            element types are ignored.
        atol: Absolute tolerance when checking closeness to 0 or 1.
    """
    if not isinstance(xs, (list, tuple)):
        return

    for i, x in enumerate(xs):
        if isinstance(x, tf.Tensor):
            v = x.numpy()
        elif isinstance(x, np.ndarray):
            v = x
        else:
            continue  # Unknown type; ignore.

        v = np.asarray(v).ravel()
        if v.size == 0:
            continue

        is_binary = np.all((np.abs(v - 0.0) <= atol) | (np.abs(v - 1.0) <= atol))
        if not is_binary:
            print(
                f"[WARNING][Gameplay] {name}[{i}] contains non-binary values. "
                f"This likely means logits or relaxed values leaked past the adapter. "
                f"min={v.min():.3f}, max={v.max():.3f}"
            )


def _as_f32(x: Any) -> tf.Tensor:
    """Convert `x` to a `tf.float32` tensor."""
    return tf.convert_to_tensor(x, dtype=tf.float32)


def _ensure_rank2(x: tf.Tensor, name: str) -> tf.Tensor:
    """Ensure `x` is rank-2 with shape (B, D).

    If `x` is rank-1 (D,), a batch dimension is added to produce (1, D).

    Args:
        x: Input tensor (any numeric dtype).
        name: Human-readable tensor name for assertion messages.

    Returns:
        A `tf.float32` tensor with rank 2.
    """
    x = _as_f32(x)
    if x.shape.rank == 1:
        x = x[None, :]
    tf.debugging.assert_rank(x, 2, message=f"{name} must be rank-2 (B,D) (or rank-1 D)")
    return x


class TrainableAssistedPlayerA(PlayerA):
    """Player A wrapper around `LinTrainableAssistedModelA`.

    This class is responsible for:
      * Converting the environment field into a batch tensor.
      * Obtaining communication logits/bits from the underlying model.
      * Producing boundary communication bits (0/1) for gameplay.
      * Computing and exposing the log-probability of the selected communication.
      * Storing intermediate tensors on `parent.previous` for Player B.

    Attributes:
        model_a: Underlying trainable model or adapter.
        parent: Parent container expected to expose a writable `previous` attribute.
            Typically set by `TrainableAssistedPlayers.players()`.
        last_logprob_comm: Log-probability of the last communication action, or `None`
            until `decide()` is called.
        explore: If `True`, use stochastic sampling; if `False`, use greedy thresholding.
    """

    def __init__(self, game_layout: Any, model_a: LinTrainableAssistedModelA) -> None:
        self.game_layout = game_layout
        self.model_a = model_a
        self.parent: Any | None = None
        self.last_logprob_comm: float | None = None
        self.explore: bool = False

    def decide(self, field: np.ndarray, supp: Any | None = None, explore: bool | None = None) -> np.ndarray:
        """Decide communication bits based on the current field.

        Backward compatibility:
            * If `model_a` is a `GameplayModelAAdapter`, it already returns boundary
              communication bits. This path can also return `comm_logits` for log-prob
              computation.
            * Otherwise, `model_a` is assumed to return communication logits and this
              method performs sampling (stochastic) or thresholding (greedy).

        Args:
            field: Flat 1D NumPy array with shape (n2,) containing 0/1 values.
                `n2` is `field_size ** 2`.
            supp: Unused. Present for interface compatibility.
            explore: Optional override of `self.explore`.

        Returns:
            Flat 1D NumPy array with shape (m,) and dtype `int32` containing 0/1 bits,
            where `m` is `comms_size`.

        Raises:
            ValueError: If `field` has an unexpected shape or contains non-binary values.
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
            # Adapter path: the adapter is responsible for producing boundary bits.
            comm_bits_tf, meas_list, out_list, comm_logits = self.model_a(
                field_batch, explore=do_explore, return_comm_logits=True
            )

            # Normalize to (B, m) and int32 for downstream consistency.
            comm_bits_tf = _ensure_rank2(comm_bits_tf, "comm_bits")
            comm_bits_tf = tf.cast(comm_bits_tf, tf.int32)

            logp = bernoulli_log_prob_from_logits(comm_logits, tf.cast(comm_bits_tf, tf.float32))
            self.last_logprob_comm = float(logp.numpy()[0])

        else:
            # Legacy path: model returns logits; boundary conversion happens here.
            comm_logits, meas_list, out_list = self.model_a.compute_with_internal(field_batch)
            comm_logits = _ensure_rank2(comm_logits, "comm_logits")  # (1, m)

            # Validate width when statically available.
            if comm_logits.shape.rank == 2 and comm_logits.shape[1] is not None:
                if int(comm_logits.shape[1]) != m:
                    raise ValueError(f"comm_logits must have width m={m}, got {int(comm_logits.shape[1])}")

            comm_probs = tf.sigmoid(comm_logits)  # (1, m)

            if do_explore:
                # Stochastic: sample each bit independently.
                rnd = tf.random.uniform(shape=tf.shape(comm_probs), dtype=tf.float32)
                comm_bits_tf = tf.cast(rnd < comm_probs, tf.int32)
            else:
                # Greedy: threshold probabilities at 0.5.
                comm_bits_tf = tf.cast(comm_probs >= 0.5, tf.int32)

            logp = bernoulli_log_prob_from_logits(comm_logits, tf.cast(comm_bits_tf, tf.float32))
            self.last_logprob_comm = float(logp.numpy()[0])

        # Gameplay safety diagnostics: these should be boundary bits at this point.
        _warn_if_not_binary_list("meas_list", meas_list)
        _warn_if_not_binary_list("out_list", out_list)

        # Store intermediate tensors for Player B on the parent container.
        if self.parent is not None:
            self.parent.previous = (meas_list, out_list)

        # Return shape (m,) as expected by the gameplay interface.
        comm_bits_np = comm_bits_tf.numpy().astype(np.int32)  # typically (1, m)
        if comm_bits_np.shape != (1, m):
            # Defensive: tolerate equivalent shapes like (m,) from alternate adapters.
            comm_bits_np = comm_bits_np.reshape(1, -1)
        return comm_bits_np[0]

    def get_log_prob(self) -> float:
        """Return the log-probability of the last communication decision.

        Returns:
            Log-probability of the boundary communication bits returned by the most
            recent call to `decide()`.

        Raises:
            RuntimeError: If `decide()` has not been called since the last `reset()`.
        """
        if self.last_logprob_comm is None:
            raise RuntimeError("No log-prob available: decide() has not been called since reset().")
        return float(self.last_logprob_comm)

    def get_prev(self) -> Any | None:
        """Return the stored intermediate tensors for Player B, if available.

        Returns:
            The value of `parent.previous`, typically `(meas_list, out_list)`, or
            `None` if no parent is set or if no previous tensors have been stored.
        """
        if self.parent is None or getattr(self.parent, "previous", None) is None:
            return None
        return self.parent.previous

    def reset(self) -> None:
        """Reset per-episode/per-rollout state."""
        self.last_logprob_comm = None