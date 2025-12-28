
"""Trainable assisted Player A implementation.

Player A computes communication bits from the field using a trainable model.
It also stores intermediate "previous" tensors on its parent (TrainableAssistedPlayers)
so Player B can consume them.

See design document for the exact decide/log-prob/previous contract. fileciteturn3file2
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import tensorflow as tf

from .lin_trainable_assisted_model_a import LinTrainableAssistedModelA

try:
    from .logit_utils import bernoulli_log_prob_from_logits  # type: ignore
except Exception:  # pragma: no cover
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

        Args:
            field: 1D array of ints, length n2, values in {0,1}.
            supp: Ignored.
            explore: Optional override of self.explore.

        Returns:
            1D NumPy array of shape (M,), dtype int, values {0,1}.
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

        field_batch = tf.convert_to_tensor(field[None, :], dtype=tf.float32)  # (1,n2)

        comm_logits, meas_list, out_list = self.model_a.compute_with_internal(field_batch)

        # Store prev tensors on the parent
        if self.parent is not None:
            self.parent.previous = (meas_list, out_list)

        comm_probs = tf.sigmoid(comm_logits)[0]  # (m,)

        if do_explore:
            # Sample Bernoulli bits.
            rnd = tf.random.uniform(shape=(m,), dtype=tf.float32)
            comm_bits = tf.cast(rnd < comm_probs, tf.int32)
        else:
            comm_bits = tf.cast(comm_probs >= 0.5, tf.int32)

        # Log-prob under independent Bernoulli with logits.
        logp = bernoulli_log_prob_from_logits(comm_logits[0:1, :], tf.cast(comm_bits[None, :], tf.float32))
        self.last_logprob_comm = float(logp.numpy()[0])

        return comm_bits.numpy().astype(np.int32)

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
