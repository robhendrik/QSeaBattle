"""Trainable PyrMeasurementLayerA (field -> measurement logits).

This module defines :class:`PyrMeasurementLayerA`, a small trainable Keras layer used in
the Pyramid (Pyr) assisted architecture.

Keras 3 build note:
All sublayers are created in `build()` based on the input dimension L.
No state is created in `call()`.

IMPORTANT (internal-training variant):
- Inputs are expected in the **scaled** domain (typically values in {-0.5, +0.5}).
- Outputs are **logits** (no sigmoid). Downstream code may apply sigmoid/DRU/etc.

Contract-aligned layer interface:
- call(field_batch) -> meas_logits
- field_batch: tf.Tensor, shape (B, L)  (cropped; L is the active width L_d)
- meas_logits: tf.Tensor, shape (B, L/2)  (cropped; equals k_d)

Implementation (MLP):
- Dense(hidden_units, activation="relu")
- Dense(L/2, activation=None)  # logits

No rule-based / teacher mapping is implemented here.

Author: Rob Hendriks
Package: Q_Sea_Battle
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import tensorflow as tf


def _ensure_rank2(x: tf.Tensor, name: str) -> None:
    if x.shape.rank is not None and x.shape.rank != 2:
        raise ValueError(f"{name} must be rank-2 (B, D). Got rank={x.shape.rank}, shape={x.shape}.")


def _require_known_last_dim(shape: tf.TensorShape, name: str) -> int:
    d = shape[-1]
    if d is None:
        raise ValueError(f"{name} last dimension must be statically known. Got shape={shape}.")
    return int(d)


class PyrMeasurementLayerA(tf.keras.layers.Layer):
    """Trainable layer mapping field state -> measurement logits."""

    def __init__(
        self,
        hidden_units: int = 64,
        name: Optional[str] = None,
        dtype: Optional[tf.dtypes.DType] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, dtype=dtype, trainable=True, **kwargs)
        if hidden_units < 1:
            raise ValueError("hidden_units must be >= 1.")
        self.hidden_units = int(hidden_units)

        # Created in build().
        self._dense_hidden: Optional[tf.keras.layers.Dense] = None
        self._dense_out: Optional[tf.keras.layers.Dense] = None
        self._built_for_L: Optional[int] = None

    def build(self, input_shape: Any) -> None:
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
        # IMPORTANT: logits output head (no sigmoid)
        self._dense_out = tf.keras.layers.Dense(
            out_dim,
            activation=None,
            name="dense_out",
            dtype=self.dtype,
        )
        self._built_for_L = L
        super().build(input_shape)

    def call(self, field_batch: tf.Tensor, training: bool = False, **kwargs: Any) -> tf.Tensor:
        x = tf.convert_to_tensor(field_batch, dtype=self.dtype or tf.float32)
        _ensure_rank2(x, "field_batch")

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
        cfg = super().get_config()
        cfg.update({"hidden_units": self.hidden_units})
        return cfg
