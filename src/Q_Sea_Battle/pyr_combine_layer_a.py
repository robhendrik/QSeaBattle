# Author: Rob Hendriks

"""Trainable PyrCombineLayerA (field + SR outcome -> next-field logits).

This module defines :class:`PyrCombineLayerA`, a small trainable Keras layer that
combines a per-player *field* representation with a shared resource (SR) outcome
vector and produces **logits** for the next field.

Keras multi-input build note:
Keras may call `build()` with only the first input shape even when `call()`
accepts multiple inputs. This layer allocates weights solely from the field
width `L`; the SR outcome width is expected to be `L/2` and is validated at
runtime.

Training-domain conventions (internal-training variant):
- `field_batch` is expected to be in the scaled/logit-like domain used by the
  training pipeline (often values such as {-0.5, +0.5}).
- `sr_outcome_batch` is expected to be logits (often values such as {-beta, +beta}).
- The output is **logits** (no sigmoid).

Shape contract (cropped / active widths):
- `field_batch`: shape (B, L)
- `sr_outcome_batch`: shape (B, L/2)
- return `next_field_logits`: shape (B, L/2)

Architecture:
- Concatenate: concat([field_batch, sr_outcome_batch]) -> (B, 3L/2)
- Dense(hidden_units, relu)
- Dense(L/2, linear) -> logits

Author: Rob Hendriks
Package: Q_Sea_Battle
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import tensorflow as tf


def _ensure_rank2(x: tf.Tensor, name: str) -> None:
    """Validate that `x` is a rank-2 tensor when rank is statically known.

    Args:
        x: Input tensor.
        name: Human-readable tensor name for error messages.

    Raises:
        ValueError: If the static rank is known and is not 2.
    """
    if x.shape.rank is not None and x.shape.rank != 2:
        raise ValueError(f"{name} must be rank-2 (B, D). Got rank={x.shape.rank}, shape={x.shape}.")


def _require_known_last_dim(shape: tf.TensorShape, name: str) -> int:
    """Return the statically known last dimension size.

    This layer relies on a statically known field width `L` to create Dense
    weights in `build()`.

    Args:
        shape: Tensor shape to inspect.
        name: Human-readable tensor name for error messages.

    Returns:
        The last dimension size as a Python int.

    Raises:
        ValueError: If the last dimension is not statically known.
    """
    d = shape[-1]
    if d is None:
        raise ValueError(f"{name} last dimension must be statically known. Got shape={shape}.")
    return int(d)


class PyrCombineLayerA(tf.keras.layers.Layer):
    """Combine field values and SR outcome logits into next-field logits.

    Inputs are expected to be rank-2 tensors with a shared batch dimension:
    - `field_batch`: shape (B, L)
    - `sr_outcome_batch`: shape (B, L/2)

    The output is a rank-2 tensor of logits with shape (B, L/2).
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
            dtype: Optional Keras dtype for layer variables and computations.
            **kwargs: Forwarded to `tf.keras.layers.Layer`.
        """
        super().__init__(name=name, dtype=dtype, trainable=True, **kwargs)
        if hidden_units < 1:
            raise ValueError("hidden_units must be >= 1.")
        self.hidden_units = int(hidden_units)

        # Created in build()
        self._dense_hidden: Optional[tf.keras.layers.Dense] = None
        self._dense_out: Optional[tf.keras.layers.Dense] = None
        self._built_for_L: Optional[int] = None

    def build(self, input_shape: Any) -> None:
        """Create sublayers for the given field width `L`.

        Notes:
            Keras may pass only the first input's shape for multi-input layers.
            This implementation uses only the field width `L` to size the output
            head as `L/2`; the SR outcome width is checked in `call()`.

        Args:
            input_shape: Shape for `field_batch`, or a multi-input shape
                structure where the first element corresponds to `field_batch`.

        Raises:
            ValueError: If `L` is not statically known or is not even.
        """
        # Keras may pass only the first input shape for multi-input layers.
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2 and not isinstance(input_shape[0], int):
            if isinstance(input_shape[0], (list, tuple, tf.TensorShape)) and isinstance(
                input_shape[1], (list, tuple, tf.TensorShape)
            ):
                field_shape = tf.TensorShape(input_shape[0])
            else:
                field_shape = tf.TensorShape(input_shape)
        else:
            field_shape = tf.TensorShape(input_shape)

        L = _require_known_last_dim(field_shape, "field_batch")
        if L % 2 != 0:
            raise ValueError(f"field_batch last dimension L must be even so that L/2 is integer. Got L={L}.")
        out_dim = L // 2

        self._dense_hidden = tf.keras.layers.Dense(
            self.hidden_units,
            activation="relu",
            name="dense_hidden",
            dtype=self.dtype,
        )
        # IMPORTANT: output head returns logits (no sigmoid).
        self._dense_out = tf.keras.layers.Dense(
            out_dim,
            activation=None,
            name="dense_out",
            dtype=self.dtype,
        )
        self._built_for_L = L
        super().build(input_shape)

    def call(
        self,
        field_batch: tf.Tensor,
        sr_outcome_batch: tf.Tensor,
        training: bool = False,
        **kwargs: Any,
    ) -> tf.Tensor:
        """Run the forward pass.

        Args:
            field_batch: Field tensor of shape (B, L). Typically in the scaled
                values used by training.
            sr_outcome_batch: SR outcome logits of shape (B, L/2).
            training: Standard Keras `training` flag passed to sublayers.
            **kwargs: Unused; present for Keras compatibility.

        Returns:
            Next-field logits tensor with shape (B, L/2).

        Raises:
            RuntimeError: If sublayers were not created in `build()`.
        """
        field_batch = tf.convert_to_tensor(field_batch, dtype=self.dtype or tf.float32)
        sr_outcome_batch = tf.convert_to_tensor(sr_outcome_batch, dtype=self.dtype or tf.float32)

        _ensure_rank2(field_batch, "field_batch")
        _ensure_rank2(sr_outcome_batch, "sr_outcome_batch")

        tf.debugging.assert_equal(
            tf.shape(sr_outcome_batch)[-1],
            tf.shape(field_batch)[-1] // 2,
            message="sr_outcome_batch last dimension must equal L/2.",
        )

        if self._dense_hidden is None or self._dense_out is None:
            raise RuntimeError("PyrCombineLayerA is not built correctly (missing sublayers).")

        x = tf.concat([field_batch, sr_outcome_batch], axis=-1)
        h = self._dense_hidden(x, training=training)
        next_field_logits = self._dense_out(h, training=training)
        return next_field_logits

    def get_config(self) -> Dict[str, Any]:
        """Return the serialized configuration for Keras cloning/serialization."""
        cfg = super().get_config()
        cfg.update({"hidden_units": self.hidden_units})
        return cfg