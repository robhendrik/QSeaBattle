"""Linear combine layer A (outcomes -> communication logits).

This module defines :class:`LinCombineLayerA`, a small learnable network that
maps measurement outcomes to communication logits.

The layer is intentionally minimal: an MLP (Dense + ReLU stack) followed by a
linear projection to the communication channel dimension.

Design agreements (per project spec):
- Input: ``outcomes`` with shape ``(B, n2)`` or ``(n2,)``.
- Output: ``comm_logits`` with shape ``(B, m)`` or ``(m,)``.
- Outputs are logits (i.e., not squashed). Downstream code applies
  sigmoid/DRU/etc. as appropriate.
"""

from __future__ import annotations

from collections.abc import Sequence

import tensorflow as tf


def _normalize_hidden_units(hidden_units: int | Sequence[int]) -> tuple[int, ...]:
    """Normalize hidden-layer configuration to a tuple.

    Args:
        hidden_units: Either an ``int`` (single hidden layer width) or a sequence
            of ``int`` values (one width per hidden layer).

    Returns:
        A tuple of ``int`` values, one per hidden layer.
    """
    if isinstance(hidden_units, int):
        return (int(hidden_units),)
    return tuple(int(u) for u in hidden_units)


class LinCombineLayerA(tf.keras.layers.Layer):
    """Learnable mapping from measurement outcomes to communication logits.

    This layer implements a configurable MLP (Dense+ReLU stack) and a final Dense
    layer producing ``comms_size`` logits.

    Attributes:
        comms_size: Number of communication channels (``m``).
        hidden_units: Tuple of hidden layer widths.
    """

    def __init__(
        self,
        comms_size: int,
        hidden_units: int | Sequence[int] = 64,
        name: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize the layer.

        Args:
            comms_size: Number of communication channels (``m``).
            hidden_units: Either an ``int`` (single hidden layer width) or a
                sequence of ``int`` values specifying the widths of a stack of
                Dense-ReLU layers.
            name: Optional layer name. Defaults to ``"LinCombineLayerA"``.
            **kwargs: Forwarded to ``tf.keras.layers.Layer``.
        """
        super().__init__(name=name or "LinCombineLayerA", **kwargs)
        self.comms_size = int(comms_size)
        self.hidden_units = _normalize_hidden_units(hidden_units)

        self._mlp: list[tf.keras.layers.Layer] = []
        for u in self.hidden_units:
            self._mlp.append(tf.keras.layers.Dense(int(u), activation="relu"))
        self._out = tf.keras.layers.Dense(self.comms_size, activation=None)

    def call(self, outcomes: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Compute communication logits from measurement outcomes.

        Args:
            outcomes: Measurement outcomes tensor with shape ``(B, n2)`` or
                ``(n2,)``.
            training: Whether the call is in training mode.

        Returns:
            Communication logits with shape ``(B, m)`` if the input was batched,
            otherwise shape ``(m,)``.
        """
        x = tf.convert_to_tensor(outcomes)

        # Support unbatched input ``(n2,)`` by promoting to ``(1, n2)`` and then
        # squeezing back to preserve the caller-visible shape contract.
        squeeze = False
        if x.shape.rank == 1:
            x = tf.expand_dims(x, axis=0)
            squeeze = True

        for layer in self._mlp:
            x = layer(x, training=training)
        logits = self._out(x, training=training)

        if squeeze:
            logits = tf.squeeze(logits, axis=0)
        return logits