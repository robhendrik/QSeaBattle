"""Trainable PR-assisted Player B implementation.

This module provides a :class:`TrainableAssistedPlayerB` wrapper that turns a
trainable TensorFlow model into a gameplay-facing "Player B" policy.

Protocol overview:
- Player A runs first and stores intermediate tensors on the shared parent under
  ``parent.previous``.
- Player B then consumes:
  - its local gun measurement bits,
  - received comm bits,
  - the tensors from ``parent.previous``,
  and decides whether to shoot (bit 0/1).

The exact "decide / log-prob / previous" contract is defined in the project
design document. This module follows the current implementation contract:
- If the underlying model is a :class:`~.gameplay_adapters.GameplayModelBAdapter`,
  it produces a discrete shoot bit directly (and optionally the corresponding
  logit for training).
- Otherwise (legacy path), the model returns a shoot logit and this wrapper
  performs sampling (stochastic) or thresholding (greedy) to obtain the bit.

Notes:
- "Bit" values are expected to be in {0, 1}. Logits represent logical bits by
  their sign (positive => 1, negative => 0).
- A lightweight runtime warning is emitted if tensors that should be binary
  appear to contain relaxed / non-binary values.
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
        """Compute Bernoulli log-probabilities from logits and 0/1 actions.

        This fallback is used in test/standalone contexts when the project-level
        utility is unavailable.

        Args:
            logits: Tensor of Bernoulli logits.
            actions01: Tensor of actions in {0, 1}, broadcastable to `logits`.

        Returns:
            Tensor of per-example log-probabilities, reduced over the last axis.
        """
        actions01 = tf.cast(actions01, tf.float32)
        log_p1 = -tf.nn.softplus(-logits)
        log_p0 = -tf.nn.softplus(logits)
        return tf.reduce_sum(actions01 * log_p1 + (1.0 - actions01) * log_p0, axis=-1)


try:
    from .players import PlayerB  # type: ignore
except Exception:  # pragma: no cover

    class PlayerB:
        """Fallback PlayerB base class.

        This placeholder keeps the module importable in isolation; the full
        gameplay base class is provided by the package.
        """


def _warn_if_not_binary_list(
    name: str,
    xs: Any,
    *,
    atol: float = 1e-6,
) -> None:
    """Print a warning if a list of tensors/arrays contains non-binary values.

    This is a gameplay-safety check only. It MUST NOT modify data.

    Args:
        name: Label to include in the warning message.
        xs: Expected to be a list/tuple of NumPy arrays or TF tensors.
        atol: Absolute tolerance for considering a value to be 0 or 1.
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

        # Accept values close to 0 or 1 (float transport/rounding can occur).
        is_binary = np.all((np.abs(v - 0.0) <= atol) | (np.abs(v - 1.0) <= atol))

        if not is_binary:
            print(
                f"[WARNING][Gameplay] {name}[{i}] contains non-binary values. "
                f"This likely means logits or relaxed values leaked past the adapter. "
                f"min={v.min():.3f}, max={v.max():.3f}"
            )


def _as_f32(x: Any) -> tf.Tensor:
    """Convert an input to a `tf.float32` tensor."""
    return tf.convert_to_tensor(x, dtype=tf.float32)


def _ensure_rank2(x: tf.Tensor, name: str) -> tf.Tensor:
    """Ensure `x` is rank-2 in the form (B, D).

    If `x` is rank-1 (D,), a batch dimension is added to produce (1, D).

    Args:
        x: Input tensor.
        name: Name used in assertion messages.

    Returns:
        A rank-2 float32 tensor.
    """
    x = _as_f32(x)
    if x.shape.rank == 1:
        x = x[None, :]
    tf.debugging.assert_rank(x, 2, message=f"{name} must be rank-2 (B,D) (or rank-1 D)")
    return x


class TrainableAssistedPlayerB(PlayerB):
    """Gameplay-facing Player B driven by a trainable assisted model.

    This wrapper:
    - validates and batches gun/comm inputs,
    - pulls Player A's saved tensors from ``parent.previous``,
    - calls the underlying model (adapter or legacy),
    - returns a discrete shoot decision (0/1),
    - stores the log-probability of the taken action for training.

    Attributes:
        game_layout: Gameplay layout object providing `field_size` and `comms_size`.
        model_b: Underlying model for Player B.
        parent: Container that provides the shared `previous` tensors (set externally).
        last_logprob_shoot: Log-probability of the last returned shoot action.
        explore: If True, sample stochastically; if False, act greedily.
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
        """Decide whether to shoot (0/1) from gun bits, comm bits, and `parent.previous`.

        Backward compatibility:
        - If `model_b` is a :class:`~.gameplay_adapters.GameplayModelBAdapter`, it
          returns `shoot_bit` directly (0/1). This path may also return a
          `shoot_logit` used to compute a training log-prob.
        - Otherwise, `model_b` returns a `shoot_logit` and this wrapper performs
          sampling/thresholding to produce the bit (legacy behavior).

        Args:
            gun: NumPy array of shape `(n2,)` with values in {0, 1}, where
                `n2 = field_size ** 2`.
            comm: NumPy array of shape `(m,)` with values typically in {0, 1}.
                (Some legacy pipelines may pass relaxed floats.)
            supp: Unused; accepted for compatibility with other player APIs.
            explore: Optional override for `self.explore`.

        Returns:
            The shoot decision as an `int` in {0, 1}.

        Raises:
            ValueError: If `gun` or `comm` shapes are inconsistent with the layout,
                or if `gun` contains values outside {0, 1}.
            RuntimeError: If `parent.previous` is missing (Player A must act first).
            TypeError: If `parent.previous` elements are not tensors/arrays.
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

        gun_batch = tf.convert_to_tensor(gun[None, :], dtype=tf.float32)  # (1, n2)
        comm_batch = tf.convert_to_tensor(comm[None, :], dtype=tf.float32)  # (1, m)

        if self.parent is None or getattr(self.parent, "previous", None) is None:
            raise RuntimeError("parent.previous is None: PlayerA must decide() before PlayerB.")

        prev_meas_list, prev_out_list = self.parent.previous

        # Non-breaking runtime check: these tensors are expected to represent bits.
        _warn_if_not_binary_list("prev_meas_list", prev_meas_list)
        _warn_if_not_binary_list("prev_out_list", prev_out_list)

        # Normalize to lists (linear case: single tensor -> list of length 1).
        if not isinstance(prev_meas_list, (list, tuple)):
            prev_meas_list = [prev_meas_list]
        if not isinstance(prev_out_list, (list, tuple)):
            prev_out_list = [prev_out_list]

        if len(prev_meas_list) < 1 or len(prev_out_list) < 1:
            raise ValueError("parent.previous lists must have length >= 1.")

        prev_meas_batch: list[tf.Tensor] = []
        prev_out_batch: list[tf.Tensor] = []

        # Ensure each previous tensor has an explicit batch dimension.
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
            # Adapter path: the adapter owns the sampling/greedy policy and returns a bit.
            shoot_bit, shoot_logit = self.model_b(
                [gun_batch, comm_batch, prev_meas_batch, prev_out_batch],
                return_shoot_logit=True,
                explore=do_explore,
            )
            shoot_bit = _ensure_rank2(shoot_bit, "shoot_bit")  # (1, 1) expected

            shoot = int(tf.cast(shoot_bit[0, 0], tf.int32).numpy())
            if shoot not in (0, 1):
                raise ValueError(f"shoot_bit must be 0/1, got {shoot}")

            logp = bernoulli_log_prob_from_logits(shoot_logit, tf.cast(shoot_bit, tf.float32))
            self.last_logprob_shoot = float(logp.numpy()[0])

        else:
            # Legacy path: model emits logits and this wrapper performs sampling/thresholding.
            shoot_logit = self.model_b([gun_batch, comm_batch, prev_meas_batch, prev_out_batch])
            shoot_logit = _ensure_rank2(shoot_logit, "shoot_logit")  # (1, 1)

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
        """Return the log-probability of the most recent shoot decision.

        Returns:
            Log-probability as a Python float.

        Raises:
            RuntimeError: If `decide()` has not been called since the last `reset()`.
        """
        if self.last_logprob_shoot is None:
            raise RuntimeError("No log-prob available: decide() has not been called since reset().")
        return float(self.last_logprob_shoot)

    def reset(self) -> None:
        """Reset per-episode/per-turn cached state."""
        self.last_logprob_shoot = None