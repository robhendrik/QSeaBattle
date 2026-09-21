"""Trainable `LinMeasurementLayerB` (gun -> measurement logits).

This module defines a small feed-forward measurement model that maps *gun* logits
to *measurement* logits with the same width.

The layer is intended for internal models that represent logical bit values as
logits, where the sign of each logit indicates the corresponding bit value.

Tensor conventions:
- Input: `gun_batch` logits, rank-2 tensor with shape `(B, n2)`
- Output: measurement logits, rank-2 tensor with shape `(B, n2)`
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import tensorflow as tf


def _ensure_rank2(x: tf.Tensor, name: str) -> None:
    """Validate that a tensor is rank-2.

    Args:
        x: Tensor to validate.
        name: Human-readable tensor name used in error messages.

    Raises:
        ValueError: If `x` has a statically known rank that is not 2.
    """
    if x.shape.rank is not None and x.shape.rank != 2:
        raise ValueError(f"{name} must be rank-2 (B, D). Got rank={x.shape.rank}, shape={x.shape}.")


class LinMeasurementLayerB(tf.keras.layers.Layer):
    """Trainable layer mapping gun logits to measurement logits.

    The mapping is implemented as a two-layer MLP:
    `Dense(hidden_units, relu)` -> `Dense(n2, linear)`

    The output width (`n2`) is inferred from the input shape at build time and
    matches the last dimension of the input.

    Attributes:
        hidden_units: Number of units in the hidden dense layer.
    """

    def __init__(
        self,
        hidden_units: int = 64,
        name: Optional[str] = None,
        dtype: Optional[tf.dtypes.DType] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the layer.

        Args:
            hidden_units: Width of the hidden layer. Must be >= 1.
            name: Optional Keras layer name.
            dtype: Optional Keras dtype for layer variables and computation.
            **kwargs: Additional keyword arguments forwarded to `Layer`.

        Raises:
            ValueError: If `hidden_units` is less than 1.
        """
        super().__init__(name=name, dtype=dtype, trainable=True, **kwargs)
        if hidden_units < 1:
            raise ValueError("hidden_units must be >= 1.")
        self.hidden_units = int(hidden_units)

        # Created in `build()` once the input width is known.
        self._dense_hidden: Optional[tf.keras.layers.Dense] = None
        self._dense_out: Optional[tf.keras.layers.Dense] = None

    def build(self, input_shape: Any) -> None:
        """Create sub-layers once the input shape is known.

        Args:
            input_shape: Keras-provided shape for `gun_batch`. The final dimension
                must be statically known to determine the output width.

        Raises:
            ValueError: If the final dimension of `input_shape` is unknown.
        """
        x_shape = tf.TensorShape(input_shape)
        d = x_shape[-1]
        if d is None:
            raise ValueError(f"gun_batch last dimension must be known. Got shape={x_shape}.")
        n2 = int(d)

        self._dense_hidden = tf.keras.layers.Dense(
            self.hidden_units,
            activation="relu",
            name="dense_hidden",
            dtype=self.dtype,
        )
        self._dense_out = tf.keras.layers.Dense(
            n2,
            activation=None,
            name="dense_out",
            dtype=self.dtype,
        )
        super().build(input_shape)

    def call(self, gun_batch: tf.Tensor, training: bool = False, **kwargs: Any) -> tf.Tensor:
        """Run the forward pass.

        Args:
            gun_batch: Gun logits tensor with shape `(B, n2)`.
            training: Whether the call is in training mode.
            **kwargs: Unused. Present for Keras API compatibility.

        Returns:
            Measurement logits tensor with shape `(B, n2)`.

        Raises:
            ValueError: If `gun_batch` has a statically known rank not equal to 2.
            RuntimeError: If the layer has not been built.
        """
        x = tf.convert_to_tensor(gun_batch, dtype=self.dtype or tf.float32)
        _ensure_rank2(x, "gun_batch")

        if self._dense_hidden is None or self._dense_out is None:
            raise RuntimeError("LinMeasurementLayerB is not built correctly.")

        h = self._dense_hidden(x, training=training)
        return self._dense_out(h, training=training)

    def get_config(self) -> Dict[str, Any]:
        """Return the serialized layer configuration for Keras."""
        cfg = super().get_config()
        cfg.update({"hidden_units": self.hidden_units})
        return cfg