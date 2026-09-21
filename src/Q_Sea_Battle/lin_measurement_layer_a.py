"""Trainable LinMeasurementLayerA (field logits -> measurement logits).

This module defines a small feed-forward (MLP) measurement layer used to map a
batch of "field" logits to a batch of "measurement" logits with the same width.

Tensor semantics:
- Logits represent logical bit values by sign (positive vs. negative).
- Input and output use the same final dimension (commonly ``n2``), and both are
  rank-2 tensors shaped ``(B, n2)``, where ``B`` is the batch size.

The layer is trainable and implemented as Dense(ReLU) -> Dense(linear).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import tensorflow as tf


def _ensure_rank2(x: tf.Tensor, name: str) -> None:
    """Validate that a tensor is rank-2.

    Args:
        x: Tensor to validate.
        name: Human-readable tensor name for error messages.

    Raises:
        ValueError: If ``x`` has a statically-known rank and it is not 2.
    """
    if x.shape.rank is not None and x.shape.rank != 2:
        raise ValueError(f"{name} must be rank-2 (B, D). Got rank={x.shape.rank}, shape={x.shape}.")


class LinMeasurementLayerA(tf.keras.layers.Layer):
    """Map field logits to measurement logits using a small MLP.

    This layer expects a rank-2 tensor of field logits with shape ``(B, n2)``
    and returns measurement logits with the same shape.

    Architecture:
        Dense(hidden_units, relu) -> Dense(n2, linear)

    Attributes:
        hidden_units: Width of the hidden Dense layer.
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
            hidden_units: Number of units in the hidden Dense layer. Must be >= 1.
            name: Optional layer name.
            dtype: Optional layer dtype.
            **kwargs: Additional keyword arguments forwarded to `tf.keras.layers.Layer`.

        Raises:
            ValueError: If ``hidden_units < 1``.
        """
        super().__init__(name=name, dtype=dtype, trainable=True, **kwargs)
        if hidden_units < 1:
            raise ValueError("hidden_units must be >= 1.")
        self.hidden_units = int(hidden_units)

        # Created in `build()` once the input width (n2) is known.
        self._dense_hidden: Optional[tf.keras.layers.Dense] = None
        self._dense_out: Optional[tf.keras.layers.Dense] = None

    def build(self, input_shape: Any) -> None:
        """Create sublayers once the input feature dimension is known.

        Args:
            input_shape: Keras-provided input shape. The final dimension must be
                statically known, as it determines the output width.
        """
        x_shape = tf.TensorShape(input_shape)
        d = x_shape[-1]
        if d is None:
            raise ValueError(f"field_batch last dimension must be known. Got shape={x_shape}.")
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

    def call(self, field_batch: tf.Tensor, training: bool = False, **kwargs: Any) -> tf.Tensor:
        """Forward pass.

        Args:
            field_batch: Field logits tensor with shape ``(B, n2)``.
            training: Whether the call is in training mode.
            **kwargs: Unused extra keyword arguments (kept for Keras compatibility).

        Returns:
            Measurement logits tensor with shape ``(B, n2)``.

        Raises:
            ValueError: If ``field_batch`` is not rank-2 (when rank is statically known).
            RuntimeError: If the layer has not been built correctly.
        """
        x = tf.convert_to_tensor(field_batch, dtype=self.dtype or tf.float32)
        _ensure_rank2(x, "field_batch")

        if self._dense_hidden is None or self._dense_out is None:
            raise RuntimeError("LinMeasurementLayerA is not built correctly.")

        h = self._dense_hidden(x, training=training)
        return self._dense_out(h, training=training)

    def get_config(self) -> Dict[str, Any]:
        """Return the serializable config for Keras."""
        cfg = super().get_config()
        cfg.update({"hidden_units": self.hidden_units})
        return cfg