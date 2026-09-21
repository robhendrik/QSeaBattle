"""Trainable PyrMeasurementLayerA (field -> measurement logits).

This module defines :class:`PyrMeasurementLayerA`, a small trainable Keras layer used in
the Pyramid (Pyr) assisted architecture.

Keras 3 build note:
Sublayers are created in :meth:`build` based on the input width ``L``. No state is
created in :meth:`call`.

Internal-training variant notes:
- Inputs are expected in the *scaled* domain (typically values in ``{-0.5, +0.5}``).
- Outputs are **logits** (no sigmoid). Downstream code may apply sigmoid/DRU/etc.

Contract-aligned layer interface:
- ``call(field_batch) -> meas_logits``
- ``field_batch``: ``tf.Tensor`` with shape ``(B, L)`` (cropped; ``L`` is the active
  width ``L_d``)
- ``meas_logits``: ``tf.Tensor`` with shape ``(B, L/2)`` (cropped; equals ``k_d``)

Implementation (MLP):
- Dense(hidden_units, activation="relu")
- Dense(L/2, activation=None)  # logits

No rule-based / teacher mapping is implemented here.

"""

from __future__ import annotations

from typing import Any, Dict, Optional

import tensorflow as tf


def _ensure_rank2(x: tf.Tensor, name: str) -> None:
    """Validate that a tensor is rank-2 when the rank is statically known.

    Args:
        x: Tensor to validate.
        name: Human-readable tensor name used in the error message.

    Raises:
        ValueError: If the rank is statically known and is not 2.
    """
    if x.shape.rank is not None and x.shape.rank != 2:
        raise ValueError(f"{name} must be rank-2 (B, D). Got rank={x.shape.rank}, shape={x.shape}.")


def _require_known_last_dim(shape: tf.TensorShape, name: str) -> int:
    """Require that a TensorShape has a statically known last dimension.

    The layer's output width depends on the input width, so the last dimension must be
    known at build time.

    Args:
        shape: Input tensor shape.
        name: Human-readable tensor name used in the error message.

    Returns:
        The last dimension as a Python int.

    Raises:
        ValueError: If the last dimension is not statically known.
    """
    d = shape[-1]
    if d is None:
        raise ValueError(f"{name} last dimension must be statically known. Got shape={shape}.")
    return int(d)


class PyrMeasurementLayerA(tf.keras.layers.Layer):
    """Map a cropped field state to measurement logits.

    This layer implements a simple 2-layer MLP:
    ``(B, L) -> (B, hidden_units) -> (B, L/2)``.

    The output is logits (no sigmoid). Logical bit values represented as logits are
    determined by the logit sign.

    Attributes:
        hidden_units: Number of hidden units in the intermediate Dense layer.
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
            hidden_units: Width of the hidden Dense layer. Must be >= 1.
            name: Optional Keras layer name.
            dtype: Optional dtype for the layer and its sublayers.
            **kwargs: Forwarded to ``tf.keras.layers.Layer``.
        """
        super().__init__(name=name, dtype=dtype, trainable=True, **kwargs)
        if hidden_units < 1:
            raise ValueError("hidden_units must be >= 1.")
        self.hidden_units = int(hidden_units)

        # Created in build() because the output dimension depends on the input width L.
        self._dense_hidden: Optional[tf.keras.layers.Dense] = None
        self._dense_out: Optional[tf.keras.layers.Dense] = None
        self._built_for_L: Optional[int] = None

    def build(self, input_shape: Any) -> None:
        """Create sublayers based on the input width.

        Args:
            input_shape: Shape of ``field_batch`` with last dimension ``L``.

        Raises:
            ValueError: If ``L`` is unknown or not even (required so that ``L/2`` is an
                integer).
        """
        x_shape = tf.TensorShape(input_shape)
        L = _require_known_last_dim(x_shape, "field_batch")
        if L % 2 != 0:
            raise ValueError(f"field_batch last dimension L must be even so that L/2 is integer. Got L={L}.")
        out_dim = L // 2

        self._dense_hidden = tf.keras.layers.Dense(
            self.hidden_units,
            activation="relu",
            name="dense_hidden",
            dtype=self.dtype,
        )
        # IMPORTANT: output head produces logits (no sigmoid).
        self._dense_out = tf.keras.layers.Dense(
            out_dim,
            activation=None,
            name="dense_out",
            dtype=self.dtype,
        )
        self._built_for_L = L
        super().build(input_shape)

    def call(self, field_batch: tf.Tensor, training: bool = False, **kwargs: Any) -> tf.Tensor:
        """Run the forward pass.

        Args:
            field_batch: Input tensor with shape ``(B, L)``.
            training: Standard Keras training flag.
            **kwargs: Unused; accepted for Keras compatibility.

        Returns:
            Measurement logits with shape ``(B, L/2)``.

        Raises:
            RuntimeError: If the layer was not built correctly (missing sublayers).
            ValueError: If the input rank is statically known and not rank-2.
        """
        x = tf.convert_to_tensor(field_batch, dtype=self.dtype or tf.float32)
        _ensure_rank2(x, "field_batch")

        # Also enforce the even-width constraint at runtime (covers dynamic shapes).
        tf.debugging.assert_equal(
            tf.shape(x)[-1] % 2,
            0,
            message="field_batch last dimension L must be even so that L/2 is integer.",
        )

        if self._dense_hidden is None or self._dense_out is None:
            raise RuntimeError("PyrMeasurementLayerA is not built correctly (missing sublayers).")

        h = self._dense_hidden(x, training=training)
        meas_logits = self._dense_out(h, training=training)
        return meas_logits

    def get_config(self) -> Dict[str, Any]:
        """Return the serializable layer configuration."""
        cfg = super().get_config()
        cfg.update({"hidden_units": self.hidden_units})
        return cfg