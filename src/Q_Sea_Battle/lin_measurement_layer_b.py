"""Linear (learnable) measurement layer B.

Provides :class:`LinMeasurementLayerB`, a learnable mapping from a *gun*
vector (flattened, length n2) to per-cell measurement probabilities.

Contract:
- Input: Tensor of shape (B, n2) or (n2,)
- Output: Tensor with same shape as input
- Range: probabilities in [0, 1] (sigmoid)

This is a low-level primitive: it is intentionally a `tf.keras.layers.Layer`
so it can be composed into larger trainable models.
"""

from __future__ import annotations

from typing import Optional, Sequence

import tensorflow as tf


class LinMeasurementLayerB(tf.keras.layers.Layer):
    """Learnable mapping: gun -> per-cell measurement probabilities in [0, 1]."""

    def __init__(
        self,
        n2: int,
        hidden_units: Sequence[int] = (64,),
        name: Optional[str] = "LinMeasurementLayerB",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        if n2 <= 0:
            raise ValueError("n2 must be positive.")
        self.n2 = int(n2)
        self.hidden_units = tuple(int(u) for u in hidden_units)

        self._mlp: list[tf.keras.layers.Layer] = []
        self._built_mlp = False

    def build(self, input_shape) -> None:
        """Create weights based on input shape."""
        if self._built_mlp:
            return

        for i, u in enumerate(self.hidden_units):
            self._mlp.append(
                tf.keras.layers.Dense(
                    u,
                    activation="relu",
                    name=f"{self.name}_dense_{i}",
                )
            )

        self._mlp.append(
            tf.keras.layers.Dense(
                self.n2,
                activation="sigmoid",
                name=f"{self.name}_dense_out",
            )
        )

        self._built_mlp = True
        super().build(input_shape)

    def call(self, guns, training: bool = False):
        """Forward pass.

        Args:
            guns: Tensor of shape (B, n2) or (n2,).
            training: Standard Keras training flag.

        Returns:
            Probabilities in [0, 1] with the same shape as `guns`.
        """
        x = tf.convert_to_tensor(guns)
        if not x.dtype.is_floating:
            x = tf.cast(x, tf.float32)

        squeeze = False
        if x.shape.rank == 1:
            x = tf.expand_dims(x, axis=0)
            squeeze = True

        if x.shape.rank != 2:
            raise ValueError("Input must have rank 1 or 2.")
        if x.shape[-1] is not None and int(x.shape[-1]) != self.n2:
            raise ValueError(f"Expected last dim {self.n2}, got {int(x.shape[-1])}.")

        y = x
        for layer in self._mlp:
            y = layer(y, training=training)

        if squeeze:
            y = tf.squeeze(y, axis=0)
        return y
