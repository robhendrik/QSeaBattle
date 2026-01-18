"""Neural network-based Player B implementation.

Author: Rob Hendriks
Package: Q_Sea_Battle
Version: 0.1
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import tensorflow as tf

from .game_layout import GameLayout
from .players_base import PlayerB
from .logit_utilities import logit_to_prob, logit_to_logprob


def _gun_one_hot_to_index(gun: np.ndarray) -> np.ndarray:
    """Convert a one-hot gun vector to a (normalised) scalar index.

    The public interface still uses a flattened one-hot vector of length ``n2``.
    Internally we compress this to a single scalar in [0, 1] representing

        idx_norm = idx / max(1, (n2 - 1))

    where ``idx`` is the integer index of the 1-bit.
    """
    gun = np.asarray(gun, dtype=np.float32)
    gun = gun.reshape(gun.shape[0], -1)  # (batch, n2)
    n2 = gun.shape[1]

    # Fallback: if the vector is not strictly one-hot (e.g. all zeros),
    # we take the argmax which is stable and well-defined.
    idx = np.argmax(gun, axis=1).astype(np.float32)
    denom = max(1, n2 - 1)
    idx_norm = idx / float(denom)
    idx_norm = idx_norm.reshape(-1, 1)
    return idx_norm


class NeuralNetPlayerB(PlayerB):
    """Player B driven by a Keras shoot model.

    The model receives a compact representation consisting of the normalised
    gun index (scalar) concatenated with the communication bits from Player A.
    It produces a single logit for the shoot decision. Depending on
    :attr:`explore`, the decision is either a deterministic threshold at 0.5
    or sampled from the underlying Bernoulli distribution. The log-probability
    of the chosen action is stored in :attr:`last_logprob`.
    """

    def __init__(
        self,
        game_layout: GameLayout,
        model_b: tf.keras.Model,
        explore: bool = False,
    ) -> None:
        """Initialise a :class:`NeuralNetPlayerB` instance.

        Args:
            game_layout: Shared :class:`GameLayout` describing the environment.
            model_b: Keras model mapping vectors of shape ``(1 + m,)``
                (normalised gun index + comm bits) to a single shoot logit.
            explore: If ``True``, sample shoot actions; if ``False``, act
                greedily by thresholding probabilities.
        """
        super().__init__(game_layout=game_layout)
        self.model_b: tf.keras.Model = model_b
        self.explore: bool = explore
        self.last_logprob: Optional[float] = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def decide(
        self,
        gun: np.ndarray,
        comm: np.ndarray,
        supp: Any | None = None,
    ) -> int:
        """Decide whether to shoot based on gun and communication.

        Args:
            gun: Flattened one-hot gun vector of length ``n2``.
            comm: Communication vector from Player A of length ``m``.
            supp: Optional supporting information (unused).

        Returns:
            ``1`` to shoot or ``0`` to not shoot.
        """
        gun = np.asarray(gun, dtype=np.float32).reshape(1, -1)
        comm = np.asarray(comm, dtype=np.float32).reshape(1, -1)

        gun_idx_norm = _gun_one_hot_to_index(gun)  # shape (1, 1)
        x = np.concatenate([gun_idx_norm, comm], axis=1)

        logits = self.model_b(x, training=False).numpy().reshape(-1)[0]
        prob = float(self.logit_to_probs(logits))

        if self.explore:
            rnd = np.random.rand()
            action = 1.0 if rnd < prob else 0.0
        else:
            action = 1.0 if prob >= 0.5 else 0.0

        log_prob = float(self.logit_to_log_probs(logits, action))
        self.last_logprob = log_prob

        return int(action)

    # ------------------------------------------------------------------
    # Helper functions for probabilities and log-probs
    # ------------------------------------------------------------------
    @staticmethod
    def logit_to_probs(logits: np.ndarray | float) -> np.ndarray | float:
        """Backward-compatible wrapper around :func:`logit_to_prob`."""
        return logit_to_prob(logits)

    @staticmethod
    def logit_to_log_probs(
        logits: np.ndarray | float,
        actions: np.ndarray | float,
    ) -> np.ndarray | float:
        """Backward-compatible wrapper around :func:`logit_to_logprob`."""
        return logit_to_logprob(logits, actions)

    # ------------------------------------------------------------------
    # Log-probability interface
    # ------------------------------------------------------------------
    def get_log_prob(self) -> float:
        """Return the log-probability of the last action.

        Returns:
            Log-probability as a scalar float.

        Raises:
            RuntimeError: If no decision has been taken since the last reset.
        """
        if self.last_logprob is None:
            raise RuntimeError("No log-prob stored; call decide() first or reset.")
        return float(self.last_logprob)

    def reset(self) -> None:
        """Reset internal state (e.g. stored log-probability)."""
        self.last_logprob = None
