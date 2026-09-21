"""Neural network-based implementation of Player B.

This module defines :class:`NeuralNetPlayerB`, a :class:`~.players_base.PlayerB`
implementation whose shoot decision is produced by a Keras model. The public
environment interface represents the gun location as a flattened one-hot vector;
this module compresses that representation to a single normalized scalar index
before passing it to the model.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import tensorflow as tf

from .game_layout import GameLayout
from .players_base import PlayerB
from .logit_utilities import logit_to_prob, logit_to_logprob


def _gun_one_hot_to_index(gun: np.ndarray) -> np.ndarray:
    """Convert a one-hot gun vector to a normalized scalar index.

    The external interface provides the gun location as a flattened one-hot
    vector of length ``n2``. This helper compresses that vector into a scalar
    in the range ``[0, 1]``:

        ``idx_norm = idx / max(1, n2 - 1)``

    where ``idx`` is the argmax index of the vector. Using argmax provides a
    stable fallback if the input is not strictly one-hot (e.g., all zeros).

    Args:
        gun: Array containing a batch of gun vectors. The input is reshaped to
            ``(batch, n2)`` and converted to ``float32``.

    Returns:
        A ``float32`` NumPy array of shape ``(batch, 1)`` containing the
        normalized indices.
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
    """Player B controlled by a Keras model that outputs a shoot logit.

    The model input is a compact feature vector formed by concatenating:

    - the normalized gun index (shape ``(1,)``), and
    - the communication bits received from Player A (shape ``(m,)``).

    The model output is a single logit representing the Bernoulli parameter for
    the shoot action. The chosen action is either:

    - sampled from the Bernoulli distribution if :attr:`explore` is ``True``, or
    - selected greedily by thresholding the probability at 0.5 if :attr:`explore`
      is ``False``.

    The log-probability of the selected action is stored and can be retrieved
    via :meth:`get_log_prob`, which is typically used for policy-gradient style
    training.

    Attributes:
        model_b: Keras model mapping input features to a single logit.
        explore: Whether to sample actions (stochastic) or act greedily.
        last_logprob: Log-probability of the most recent action, if available.
    """

    def __init__(
        self,
        game_layout: GameLayout,
        model_b: tf.keras.Model,
        explore: bool = False,
    ) -> None:
        """Initialize a :class:`NeuralNetPlayerB` instance.

        Args:
            game_layout: Shared :class:`~.game_layout.GameLayout` describing the
                environment.
            model_b: Keras model mapping vectors of shape ``(1 + m,)`` (normalized
                gun index + communication bits) to a single shoot logit.
            explore: If ``True``, sample shoot actions; if ``False``, choose
                actions greedily by thresholding the probability.
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
        """Decide whether Player B shoots.

        This method converts the public gun representation (flattened one-hot) to
        a normalized scalar index, concatenates it with the communication vector,
        and forwards the resulting feature vector through :attr:`model_b`.

        Args:
            gun: Flattened one-hot gun vector of length ``n2``.
            comm: Communication vector from Player A of length ``m``.
            supp: Optional supporting information (unused).

        Returns:
            ``1`` if shooting is selected, otherwise ``0``.
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
        """Convert logit(s) to Bernoulli probability/probabilities.

        This is a backward-compatible wrapper around :func:`~.logit_utilities.logit_to_prob`.

        Args:
            logits: Scalar logit or NumPy array of logits.

        Returns:
            Probability value(s) with the same structure as ``logits``.
        """
        return logit_to_prob(logits)

    @staticmethod
    def logit_to_log_probs(
        logits: np.ndarray | float,
        actions: np.ndarray | float,
    ) -> np.ndarray | float:
        """Compute log-probability of Bernoulli action(s) under given logit(s).

        This is a backward-compatible wrapper around
        :func:`~.logit_utilities.logit_to_logprob`.

        Args:
            logits: Scalar logit or NumPy array of logits.
            actions: Action(s) encoded as 0/1 (or float equivalents).

        Returns:
            Log-probability value(s) with a structure compatible with inputs.
        """
        return logit_to_logprob(logits, actions)

    # ------------------------------------------------------------------
    # Log-probability interface
    # ------------------------------------------------------------------
    def get_log_prob(self) -> float:
        """Return the log-probability of the most recent action.

        Returns:
            Log-probability as a scalar float.

        Raises:
            RuntimeError: If no decision has been taken since the last reset.
        """
        if self.last_logprob is None:
            raise RuntimeError("No log-prob stored; call decide() first or reset.")
        return float(self.last_logprob)

    def reset(self) -> None:
        """Reset internal episode state.

        Currently this clears any stored log-probability from the previous
        decision.
        """
        self.last_logprob = None