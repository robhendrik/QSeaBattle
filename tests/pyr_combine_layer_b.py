"""Trainable PyrCombineLayerB (gun + SR outcome + comm -> next gun probabilities, next comm probabilities).

This module defines :class:`PyrCombineLayerB`, a trainable Keras layer that
combines the current gun representation, shared-resource (SR) outcome, and
communication bit, and produces probabilities for the next pyramid level.

Keras 3 build note:
All sublayers are created in `build()` using the gun dimension L. No state is
created in `call()`.

Contract (must match existing public API exactly):
- call(gun_batch, sr_outcome_batch, comm_batch, training=False)
  -> (next_gun, next_comm)
- gun_batch: tf.Tensor, shape (B, L)
- sr_outcome_batch: tf.Tensor, shape (B, L/2)
- comm_batch: tf.Tensor, shape (B, 1)
- next_gun: tf.Tensor, shape (B, L/2)   # probabilities in [0, 1]
- next_comm: tf.Tensor, shape (B, 1)    # probabilities in [0, 1]
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import tensorflow as tf


def _ensure_rank2(x: tf.Tensor, name: str) -> None:
    if x.shape.rank is not None and x.shape.rank != 2:
        raise ValueError(f"{name} must be rank-2 (B, D). Got rank={x.shape.rank}, shape={x.shape}.")


def _require_known_last_dim(shape: tf.TensorShape, name: str) -> int:
    d = shape[-1]
    if d is None:
        raise ValueError(f"{name} last dimension must be statically known. Got shape={shape}.")
    return int(d)


class PyrCombineLayerB(tf.keras.layers.Layer):
    """Trainable layer mapping (gun, sr_outcome, comm) -> (next_gun, next_comm)."""

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

        # Created in build()
        self._dense_hidden: Optional[tf.keras.layers.Dense] = None
        self._dense_gun: Optional[tf.keras.layers.Dense] = None
        self._dense_comm: Optional[tf.keras.layers.Dense] = None
        self._built_for_L: Optional[int] = None

    def build(self, input_shape: Any) -> None:
        # Keras may pass only gun shape for multi-input layers
        gun_shape = tf.TensorShape(input_shape)
        L = _require_known_last_dim(gun_shape, "gun_batch")
        if L % 2 != 0:
            raise ValueError(f"gun_batch last dimension L must be even. Got L={L}.")

        gun_out_dim = L // 2

        self._dense_hidden = tf.keras.layers.Dense(
            self.hidden_units,
            activation="relu",
            name="dense_hidden",
            dtype=self.dtype,
        )
        self._dense_gun = tf.keras.layers.Dense(
            gun_out_dim,
            activation="sigmoid",
            name="dense_gun",
            dtype=self.dtype,
        )
        self._dense_comm = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            name="dense_comm",
            dtype=self.dtype,
        )
        self._built_for_L = L
        super().build(input_shape)

    def call(
        self,
        gun_batch: tf.Tensor,
        sr_outcome_batch: tf.Tensor,
        comm_batch: tf.Tensor,
        training: bool = False,
        **kwargs: Any,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        gun_batch = tf.convert_to_tensor(gun_batch, dtype=self.dtype or tf.float32)
        sr_outcome_batch = tf.convert_to_tensor(sr_outcome_batch, dtype=self.dtype or tf.float32)
        comm_batch = tf.convert_to_tensor(comm_batch, dtype=self.dtype or tf.float32)

        _ensure_rank2(gun_batch, "gun_batch")
        _ensure_rank2(sr_outcome_batch, "sr_outcome_batch")
        _ensure_rank2(comm_batch, "comm_batch")

        # Runtime consistency checks
        tf.debugging.assert_equal(
            tf.shape(sr_outcome_batch)[-1],
            tf.shape(gun_batch)[-1] // 2,
            message="sr_outcome_batch last dimension must equal L/2.",
        )
        tf.debugging.assert_equal(
            tf.shape(comm_batch)[-1],
            1,
            message="comm_batch last dimension must be 1.",
        )

        if self._dense_hidden is None or self._dense_gun is None or self._dense_comm is None:
            raise RuntimeError("PyrCombineLayerB is not built correctly (missing sublayers).")

        x = tf.concat([gun_batch, sr_outcome_batch, comm_batch], axis=-1)
        x = self._dense_hidden(x, training=training)
        next_gun = self._dense_gun(x, training=training)
        next_comm = self._dense_comm(x, training=training)
        return next_gun, next_comm

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({"hidden_units": self.hidden_units})
        return cfg
