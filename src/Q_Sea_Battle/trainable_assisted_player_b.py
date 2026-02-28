
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
from .gameplay_adapters import GameplayModelBAdapter

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

        Backward compatibility:
        - If model_b is a GameplayModelBAdapter, it returns shoot_bit directly (0/1).
        - Otherwise, model_b returns shoot_logit and we sample/threshold here (legacy behavior).

        Args:
            gun: 1D array of ints length n2, values in {0,1}.
            comm: 1D array of ints length m, values in {0,1} (or float in [0,1] for DRU legacy).
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


        gun_batch = tf.convert_to_tensor(gun[None, :], dtype=tf.float32)   # (1, n2)
        comm_batch = tf.convert_to_tensor(comm[None, :], dtype=tf.float32) # (1, m)

        if self.parent is None or getattr(self.parent, "previous", None) is None:
            raise RuntimeError("parent.previous is None: PlayerA must decide() before PlayerB.")

        prev_meas_list, prev_out_list = self.parent.previous

        # Gameplay safety check (non-breaking)
        _warn_if_not_binary_list("prev_meas_list", prev_meas_list)
        _warn_if_not_binary_list("prev_out_list", prev_out_list)

        # Normalize to lists (linear case: single tensor → list of length 1)
        if not isinstance(prev_meas_list, (list, tuple)):
            prev_meas_list = [prev_meas_list]
        if not isinstance(prev_out_list, (list, tuple)):
            prev_out_list = [prev_out_list]

        if len(prev_meas_list) < 1 or len(prev_out_list) < 1:
            raise ValueError("parent.previous lists must have length >= 1.")

        prev_meas_batch: list[tf.Tensor] = []
        prev_out_batch: list[tf.Tensor] = []

        for pm, po in zip(prev_meas_list, prev_out_list):
            if isinstance(pm, np.ndarray):
                pm = tf.convert_to_tensor(pm, dtype=tf.float32)
            if isinstance(po, np.ndarray):
                po = tf.convert_to_tensor(po, dtype=tf.float32)

            if getattr(pm, "shape", None) is None or getattr(po, "shape", None) is None:
                raise TypeError("prev_meas_list/prev_out_list elements must be tensors or numpy arrays.")

            if pm.shape.rank == 1:
                pm = pm[None, :]
            if po.shape.rank == 1:
                po = po[None, :]

            prev_meas_batch.append(pm)
            prev_out_batch.append(po)

        # ---- Model call (adapter vs legacy) ----
        if isinstance(self.model_b, GameplayModelBAdapter):
            # Adapter path: returns shoot_bit (0/1) as tensor.
            shoot_bit, shoot_logit = self.model_b([gun_batch, comm_batch, prev_meas_batch, prev_out_batch],
                                                   return_shoot_logit=True,
                                                   explore=do_explore)
            shoot_bit = _ensure_rank2(shoot_bit, "shoot_bit")  # (1,1) expected

            # Normalize to python int 0/1
            shoot = int(tf.cast(shoot_bit[0, 0], tf.int32).numpy())
            if shoot not in (0, 1):
                raise ValueError(f"shoot_bit must be 0/1, got {shoot}")

            logp = bernoulli_log_prob_from_logits(shoot_logit, tf.cast(shoot_bit, tf.float32))
            self.last_logprob_shoot = float(logp.numpy()[0]) 

        else:
            # Legacy path: model returns shoot_logit (1,1)
            shoot_logit = self.model_b([gun_batch, comm_batch, prev_meas_batch, prev_out_batch])
            shoot_logit = _ensure_rank2(shoot_logit, "shoot_logit")  # (1,1)

            shoot_prob = tf.sigmoid(shoot_logit)[0, 0]

            if do_explore:
                rnd = tf.random.uniform(shape=(), dtype=tf.float32)
                shoot = int((rnd < shoot_prob).numpy())
            else:
                shoot = int((shoot_prob >= 0.5).numpy())

            logp = bernoulli_log_prob_from_logits(
                shoot_logit,
                tf.constant([[float(shoot)]], dtype=tf.float32),
            )
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
