"""Linear combine layer B ((outcomes, comm) -> shoot logit).

This module defines :class:`LinCombineLayerB`, a small learnable network that maps
measurement outcomes and the communicated message to a single shoot logit.

Design agreements (per project spec):
- Inputs:
    outcomes: shape (B, n2) or (n2,)
    comm:     shape (B, m)  or (m,)
- Output:
    shoot_logit: shape (B, 1) or (1,)
- Output is a *logit* (not squashed).
"""

from __future__ import annotations

from collections.abc import Sequence

import tensorflow as tf


def _normalize_hidden_units(hidden_units: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(hidden_units, int):
        return (int(hidden_units),)
    return tuple(int(u) for u in hidden_units)


class LinCombineLayerB(tf.keras.layers.Layer):
    """Learnable mapping (outcomes, comm) -> shoot logit.

    Parameters
    ----------
    comms_size:
        Number of communication channels (m). Used for shape checks only.
    hidden_units:
        Either an int (single hidden layer width) or a sequence of ints.
    """

    def __init__(
        self,
        comms_size: int,
        hidden_units: int | Sequence[int] = 64,
        name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name or "LinCombineLayerB", **kwargs)
        self.comms_size = int(comms_size)
        self.hidden_units = _normalize_hidden_units(hidden_units)

        self._mlp: list[tf.keras.layers.Layer] = []
        for u in self.hidden_units:
            self._mlp.append(tf.keras.layers.Dense(int(u), activation="relu"))
        self._out = tf.keras.layers.Dense(1, activation=None)

    def call(self, outcomes: tf.Tensor, comm: tf.Tensor, training: bool = False) -> tf.Tensor:
        o = tf.convert_to_tensor(outcomes)
        c = tf.convert_to_tensor(comm)

        squeeze = False
        if o.shape.rank == 1:
            o = tf.expand_dims(o, axis=0)
            squeeze = True
        if c.shape.rank == 1:
            c = tf.expand_dims(c, axis=0)

        x = tf.concat([o, c], axis=-1)

        for layer in self._mlp:
            x = layer(x, training=training)
        logit = self._out(x, training=training)

        if squeeze:
            logit = tf.squeeze(logit, axis=0)
        return logit
