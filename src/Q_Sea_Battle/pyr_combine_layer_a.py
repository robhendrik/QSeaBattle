"""Trainable PyrCombineLayerA (field + SR outcome -> next field logits).

This module defines :class:`PyrCombineLayerA`, a small trainable Keras layer that
combines the current field representation with a shared-resource (SR) outcome
vector and produces logits for the next field.

Keras 3 multi-input build note:
Keras may call `build()` with only the *first* input shape even if `call()` accepts
multiple positional inputs. For this layer, that is sufficient because the SR
outcome dimension is determined by L/2. Therefore we create *all* state in
`build()` using the field shape only, and validate SR shape at runtime.

Contract (must match existing public API exactly):
- call(field_batch, sr_outcome_batch, training=False) -> next_field
- field_batch: tf.Tensor, shape (B, L)
- sr_outcome_batch: tf.Tensor, shape (B, L/2)
- next_field: tf.Tensor, shape (B, L/2)  # probabilities in [0, 1]

MLP:
- concat([field_batch, sr_outcome_batch]) -> (B, 3L/2)
- Dense(hidden_units, relu)
- Dense(L/2, sigmoid) -> probabilities

No rule-based / teacher mapping (XOR/parity/etc.) is implemented here.

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


class PyrCombineLayerA(tf.keras.layers.Layer):
    """Trainable layer mapping (field, sr_outcome) -> next_field probabilities."""

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

        # These are created in build() (Keras 3: do not create new state in call()).
        self._dense_hidden: Optional[tf.keras.layers.Dense] = None
        self._dense_out: Optional[tf.keras.layers.Dense] = None
        self._built_for_L: Optional[int] = None

    def build(self, input_shape: Any) -> None:
        # Keras may pass either:
        # - field_shape                (B, L)  (common for multi-input layers)
        # - (field_shape, sr_shape)     ((B, L), (B, L/2))  (sometimes)
        #
        # We only need L to build, because sr_dim = L/2.
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2 and not isinstance(input_shape[0], int):
            # Could be (B, L) as flat ints OR (field_shape, sr_shape) as nested shapes.
            # Distinguish nested shapes by checking whether the first element is shape-like.
            if isinstance(input_shape[0], (list, tuple, tf.TensorShape)) and isinstance(input_shape[1], (list, tuple, tf.TensorShape)):
                field_shape = tf.TensorShape(input_shape[0])
            else:
                field_shape = tf.TensorShape(input_shape)  # treat as (B, L)
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
        self._dense_out = tf.keras.layers.Dense(
            out_dim,
            activation="sigmoid",
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
        field_batch = tf.convert_to_tensor(field_batch, dtype=self.dtype or tf.float32)
        sr_outcome_batch = tf.convert_to_tensor(sr_outcome_batch, dtype=self.dtype or tf.float32)

        _ensure_rank2(field_batch, "field_batch")
        _ensure_rank2(sr_outcome_batch, "sr_outcome_batch")

        # Runtime shape consistency check.
        tf.debugging.assert_equal(
            tf.shape(sr_outcome_batch)[-1],
            tf.shape(field_batch)[-1] // 2,
            message="sr_outcome_batch last dimension must equal L/2 where L is field_batch last dimension.",
        )

        if self._dense_hidden is None or self._dense_out is None:
            raise RuntimeError("PyrCombineLayerA is not built correctly (missing sublayers).")

        x = tf.concat([field_batch, sr_outcome_batch], axis=-1)
        x = self._dense_hidden(x, training=training)
        next_field = self._dense_out(x, training=training)
        return next_field

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({"hidden_units": self.hidden_units})
        return cfg
