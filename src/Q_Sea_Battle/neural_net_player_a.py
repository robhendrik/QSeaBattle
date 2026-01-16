"""Neural network-based Player A implementation.

Author: Rob Hendriks
Version: 0.3
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import tensorflow as tf

from .game_layout import GameLayout
from .players_base import PlayerA
from .logit_utilities import logit_to_prob, logit_to_logprob


def _scale_field(field: np.ndarray) -> np.ndarray:
    """Scale a binary {0,1} field to be centred around zero.

    We use the simple affine transform

        x_scaled = x - 0.5

    so that 0 -> -0.5 and 1 -> +0.5. This makes the global average linearly
    visible to the network and improves learning of majority-style signals.
    """
    field = np.asarray(field, dtype=np.float32)
    return field - 0.5


class NeuralNetPlayerA(PlayerA):
    """Player A driven by a Keras communication model.

    The model receives the flattened *scaled* field as input and produces
    logits for each communication bit. Depending on :attr:`explore`, decisions
    are either deterministic (thresholded at 0.5) or sampled from the
    Bernoulli distribution induced by the predicted probabilities.

    The log-probability of the taken action is stored in :attr:`last_logprob`
    and can be retrieved via :meth:`get_log_prob` for RL-style training.
    """

    def __init__(
        self,
        game_layout: GameLayout,
        model_a: tf.keras.Model,
        explore: bool = False,
    ) -> None:
        """Initialise a :class:`NeuralNetPlayerA` instance.

        Args:
            game_layout: Shared :class:`GameLayout` describing the environment.
            model_a: Keras model mapping scaled field vectors of shape
                ``(n2,)`` to communication logits of shape ``(m,)``.
            explore: If ``True``, sample communication bits; if ``False``,
                act greedily by thresholding probabilities.
        """
        super().__init__(game_layout=game_layout)
        self.model_a: tf.keras.Model = model_a
        self.explore: bool = explore
        self.last_logprob: Optional[float] = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def decide(self, field: np.ndarray, supp: Any | None = None) -> np.ndarray:
        """Decide on the communication vector for the given field.

        Args:
            field: Flattened field array of shape ``(n2,)`` with values in
                ``{0, 1}``.
            supp: Optional supporting information (unused).

        Returns:
            NumPy array of shape ``(m,)`` with integer bits in ``{0, 1}``.
        """
        # Basic validation and scaling.
        field = np.asarray(field, dtype=np.float32).reshape(1, -1)
        field_scaled = _scale_field(field)

        # Forward pass through the model (logits).
        logits = self.model_a(field_scaled, training=False).numpy()[0]
        probs = self.logit_to_probs(logits)

        if self.explore:
            # Sample each bit from Bernoulli(prob).
            rnd = np.random.rand(*probs.shape)
            actions = (rnd < probs).astype(np.float32)
        else:
            # Greedy threshold at 0.5.
            actions = (probs >= 0.5).astype(np.float32)

        # Compute and store the log-probability of the chosen action.
        log_probs_bits = self.logit_to_log_probs(logits, actions)
        # Sum over bits to obtain a scalar log-probability.
        self.last_logprob = float(np.sum(log_probs_bits))

        return actions.astype(int)

    # ------------------------------------------------------------------
    # Helper functions for probabilities and log-probs
    # ------------------------------------------------------------------
    @staticmethod
    def logit_to_probs(logits: np.ndarray | float) -> np.ndarray | float:
        """Backward-compatible wrapper around :func:`logit_to_prob`.

        This keeps the old name but delegates to the shared utility
        implementation in :mod:`Q_Sea_Battle.logit_utils`.
        """
        return logit_to_prob(logits)

    @staticmethod
    def logit_to_log_probs(
        logits: np.ndarray | float,
        actions: np.ndarray | float,
    ) -> np.ndarray | float:
        """Backward-compatible wrapper around :func:`logit_to_logprob`.

        Args:
            logits: Scalar or array of logits.
            actions: Scalar or array of actions in {0, 1}.

        Returns:
            Log-probabilities with the same shape as the broadcast of
            ``logits`` and ``actions``.
        """
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
