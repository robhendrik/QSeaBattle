
"""Trainable assisted Player B implementation.

Player B consumes the "previous" tensors stored by Player A on the parent,
and combines these with its own gun measurement + received comm bits to decide shoot.

See design document for the exact decide/log-prob/previous contract. 

Author: Rob Hendriks
Package: Q_Sea_Battle
Version: 0.1
"""
from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf

from .lin_trainable_assisted_model_b import LinTrainableAssistedModelB

try:
    from .logit_utils import bernoulli_log_prob_from_logits  # type: ignore
except Exception:  # pragma: no cover
    def bernoulli_log_prob_from_logits(logits: tf.Tensor, actions01: tf.Tensor) -> tf.Tensor:
        actions01 = tf.cast(actions01, tf.float32)
        log_p1 = -tf.nn.softplus(-logits)
        log_p0 = -tf.nn.softplus(logits)
        return tf.reduce_sum(actions01 * log_p1 + (1.0 - actions01) * log_p0, axis=-1)

try:
    from .players import PlayerB  # type: ignore
except Exception:  # pragma: no cover
    class PlayerB:
        """Fallback PlayerB base class."""


class TrainableAssistedPlayerB(PlayerB):
    """Player B wrapper around LinTrainableAssistedModelB.

    Public attributes:
        model_b: LinTrainableAssistedModelB
        parent: TrainableAssistedPlayers (set by TrainableAssistedPlayers.players())
        last_logprob_shoot: float | None
        explore: bool (False=greedy, True=sample)
    """

    def __init__(self, game_layout: Any, model_b: LinTrainableAssistedModelB) -> None:
        self.game_layout = game_layout
        self.model_b = model_b
        self.parent: Any | None = None
        self.last_logprob_shoot: float | None = None
        self.explore: bool = False

    def decide(
        self,
        gun: np.ndarray,
        comm: np.ndarray,
        supp: Any | None = None,
        explore: bool | None = None,
    ) -> int:
        """Decide whether to shoot (0/1) based on gun + comm + parent's previous tensors.

        Args:
            gun: 1D array of ints length n2, values in {0,1}.
            comm: 1D array of ints length m, values in {0,1} (or float in [0,1] for DRU).
            supp: Ignored.
            explore: Optional override of self.explore.

        Returns:
            int 0 or 1
        """
        del supp
        do_explore = self.explore if explore is None else bool(explore)

        n2 = int(getattr(self.game_layout, "field_size")) ** 2
        m = int(getattr(self.game_layout, "comms_size"))

        gun = np.asarray(gun)
        if gun.shape != (n2,):
            raise ValueError(f"gun must have shape ({n2},), got {gun.shape}")
        if not np.all((gun == 0) | (gun == 1)):
            raise ValueError("gun must contain only 0/1")

        comm = np.asarray(comm)
        if comm.shape != (m,):
            raise ValueError(f"comm must have shape ({m},), got {comm.shape}")

        gun_batch = tf.convert_to_tensor(gun[None, :], dtype=tf.float32)  # (1,n2)
        comm_batch = tf.convert_to_tensor(comm[None, :], dtype=tf.float32)  # (1,m)

        if self.parent is None or getattr(self.parent, "previous", None) is None:
            raise RuntimeError("parent.previous is None: PlayerA must decide() before PlayerB.")

        prev_meas_list, prev_out_list = self.parent.previous
        if not (isinstance(prev_meas_list, list) and isinstance(prev_out_list, list)):
            raise TypeError("parent.previous must be (list, list).")
        if len(prev_meas_list) < 1 or len(prev_out_list) < 1:
            raise ValueError("parent.previous lists must have length >= 1.")

        prev_meas_list, prev_out_list = self.parent.previous

        # Normalize to lists (linear case: single tensor → list of length 1)
        if not isinstance(prev_meas_list, (list, tuple)):
            prev_meas_list = [prev_meas_list]
        if not isinstance(prev_out_list, (list, tuple)):
            prev_out_list = [prev_out_list]

        prev_meas_batch = []
        prev_out_batch = []

        for pm, po in zip(prev_meas_list, prev_out_list):
            if isinstance(pm, np.ndarray):
                pm = tf.convert_to_tensor(pm, dtype=tf.float32)
            if isinstance(po, np.ndarray):
                po = tf.convert_to_tensor(po, dtype=tf.float32)

            if pm.shape.rank == 1:
                pm = pm[None, :]
            if po.shape.rank == 1:
                po = po[None, :]

            prev_meas_batch.append(pm)
            prev_out_batch.append(po)

        shoot_logit = self.model_b(
            [gun_batch, comm_batch, prev_meas_batch, prev_out_batch]
        )  # (1,1)

        shoot_prob = tf.sigmoid(shoot_logit)[0, 0]

        if do_explore:
            rnd = tf.random.uniform(shape=(), dtype=tf.float32)
            shoot = int((rnd < shoot_prob).numpy())
        else:
            shoot = int((shoot_prob >= 0.5).numpy())

        logp = bernoulli_log_prob_from_logits(shoot_logit, tf.constant([[float(shoot)]], dtype=tf.float32))
        self.last_logprob_shoot = float(logp.numpy()[0])

        return shoot

    def get_log_prob(self) -> float:
        """Return log-probability of last taken shoot action."""
        if self.last_logprob_shoot is None:
            raise RuntimeError("No log-prob available: decide() has not been called since reset().")
        return float(self.last_logprob_shoot)

    def reset(self) -> None:
        """Reset internal state."""
        self.last_logprob_shoot = None
