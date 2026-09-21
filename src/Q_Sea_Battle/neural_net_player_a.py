"""Neural network-based implementation of Player A.

This module provides :class:`NeuralNetPlayerA`, a :class:`~.players_base.PlayerA`
implementation that uses a Keras model to map a binary game field to a vector of
communication bits.

The model operates on a scaled version of the binary field (values shifted from
``{0, 1}`` to ``{-0.5, +0.5}``) and outputs per-bit logits. Actions can be chosen
greedily (deterministic thresholding) or stochastically (Bernoulli sampling),
and the log-probability of the most recent action is stored for RL-style
training.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import tensorflow as tf

from .game_layout import GameLayout
from .players_base import PlayerA
from .logit_utilities import logit_to_prob, logit_to_logprob


def _scale_field(field: np.ndarray) -> np.ndarray:
    """Center a binary field around zero for neural network input.

    Applies an affine transform::

        x_scaled = x - 0.5

    mapping ``0 -> -0.5`` and ``1 -> +0.5``. This provides a zero-centered input,
    which can improve optimization and make global average signals linearly
    accessible to the model.

    Args:
        field: Array-like input expected to contain values in ``{0, 1}``.

    Returns:
        A ``np.float32`` NumPy array with the same shape as ``field``.
    """
    field = np.asarray(field, dtype=np.float32)
    return field - 0.5


class NeuralNetPlayerA(PlayerA):
    """Player A driven by a Keras communication model.

    The model receives the flattened, scaled field and outputs one logit per
    communication bit. The sign and magnitude of each logit correspond to the
    implied Bernoulli probability for that bit.

    If :attr:`explore` is enabled, bits are sampled independently from the
    Bernoulli distribution induced by the model probabilities. Otherwise, bits
    are chosen deterministically by thresholding probabilities at ``0.5``.

    The (summed) log-probability of the most recent action is stored in
    :attr:`last_logprob` and can be retrieved with :meth:`get_log_prob`.
    """

    def __init__(
        self,
        game_layout: GameLayout,
        model_a: tf.keras.Model,
        explore: bool = False,
    ) -> None:
        """Initialize a :class:`NeuralNetPlayerA`.

        Args:
            game_layout: Game layout describing field size and communication
                dimensions.
            model_a: Keras model mapping a batch of scaled field vectors
                (shape ``(batch, n2)``) to per-bit logits (shape ``(batch, m)``).
            explore: If ``True``, sample communication bits; if ``False``, act
                greedily by thresholding probabilities.
        """
        super().__init__(game_layout=game_layout)
        self.model_a: tf.keras.Model = model_a
        self.explore: bool = explore
        self.last_logprob: Optional[float] = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def decide(self, field: np.ndarray, supp: Any | None = None) -> np.ndarray:
        """Compute a communication vector for the given field.

        Args:
            field: Flattened field array of shape ``(n2,)`` with values in
                ``{0, 1}``.
            supp: Optional supporting information (currently unused).

        Returns:
            NumPy array of shape ``(m,)`` with integer bits in ``{0, 1}``.
        """
        # Convert to a batch of size 1 to match typical Keras model inputs.
        field = np.asarray(field, dtype=np.float32).reshape(1, -1)
        field_scaled = _scale_field(field)

        # Forward pass through the model. The model is expected to output logits.
        logits = self.model_a(field_scaled, training=False).numpy()[0]
        probs = self.logit_to_probs(logits)

        if self.explore:
            # Independent Bernoulli sampling per bit.
            rnd = np.random.rand(*probs.shape)
            actions = (rnd < probs).astype(np.float32)
        else:
            # Greedy: choose the most likely bit under a 0.5 probability threshold.
            actions = (probs >= 0.5).astype(np.float32)

        # Per-bit log-probabilities under the Bernoulli distribution parameterized
        # by the logits, conditioned on the sampled/greedy actions.
        log_probs_bits = self.logit_to_log_probs(logits, actions)
        # Aggregate over bits into a scalar log-probability for the full message.
        self.last_logprob = float(np.sum(log_probs_bits))

        return actions.astype(int)

    # ------------------------------------------------------------------
    # Helper functions for probabilities and log-probs
    # ------------------------------------------------------------------
    @staticmethod
    def logit_to_probs(logits: np.ndarray | float) -> np.ndarray | float:
        """Convert logits to probabilities.

        This is a compatibility wrapper that delegates to
        :func:`~.logit_utilities.logit_to_prob`.

        Args:
            logits: Scalar or array of logits.

        Returns:
            Scalar or array of probabilities with the same shape as ``logits``.
        """
        return logit_to_prob(logits)

    @staticmethod
    def logit_to_log_probs(
        logits: np.ndarray | float,
        actions: np.ndarray | float,
    ) -> np.ndarray | float:
        """Compute per-bit log-probabilities for given actions under logits.

        This is a compatibility wrapper that delegates to
        :func:`~.logit_utilities.logit_to_logprob`.

        Args:
            logits: Scalar or array of logits.
            actions: Scalar or array of actions in ``{0, 1}``.

        Returns:
            Log-probabilities with the same shape as the broadcast of ``logits``
            and ``actions``.
        """
        return logit_to_logprob(logits, actions)

    # ------------------------------------------------------------------
    # Log-probability interface
    # ------------------------------------------------------------------
    def get_log_prob(self) -> float:
        """Return the log-probability of the most recent decided action.

        Returns:
            The summed log-probability over all communication bits.

        Raises:
            RuntimeError: If :meth:`decide` has not been called since the last
                :meth:`reset`.
        """
        if self.last_logprob is None:
            raise RuntimeError("No log-prob stored; call decide() first or reset.")
        return float(self.last_logprob)

    def reset(self) -> None:
        """Reset internal state.

        Clears any stored log-probability from the previous decision.
        """
        self.last_logprob = None