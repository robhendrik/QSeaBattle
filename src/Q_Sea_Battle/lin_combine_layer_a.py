"""Linear combine layer A (outcomes -> communication logits).

This module defines :class:`LinCombineLayerA`, a small learnable network that maps
measurement outcomes to communication logits.

Design agreements (per project spec):
- Input: outcomes of shape (B, n2) or (n2,)
- Output: comm_logits of shape (B, m) or (m,)
- Output are *logits* (not squashed); downstream code applies sigmoid/DRU/etc.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable

import tensorflow as tf


def _normalize_hidden_units(hidden_units: int | Sequence[int]) -> tuple[int, ...]:
    """Accept an int or a sequence of ints and normalize to a tuple."""
    if isinstance(hidden_units, int):
        return (int(hidden_units),)
    # Support tuples/lists/other sequences
    return tuple(int(u) for u in hidden_units)


class LinCombineLayerA(tf.keras.layers.Layer):
    """Learnable mapping outcomes -> comm logits.

    Parameters
    ----------
    comms_size:
        Number of communication channels (m).
    hidden_units:
        Either an int (single hidden layer width) or a sequence of ints
        specifying a stack of Dense-ReLU layers.
    """

    def __init__(
        self,
        comms_size: int,
        hidden_units: int | Sequence[int] = 64,
        name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name or "LinCombineLayerA", **kwargs)
        self.comms_size = int(comms_size)
        self.hidden_units = _normalize_hidden_units(hidden_units)

        self._mlp: list[tf.keras.layers.Layer] = []
        for u in self.hidden_units:
            self._mlp.append(tf.keras.layers.Dense(int(u), activation="relu"))
        self._out = tf.keras.layers.Dense(self.comms_size, activation=None)

    def call(self, outcomes: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = tf.convert_to_tensor(outcomes)
        # Support (n2,) by promoting to (1,n2) then squeezing back.
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
