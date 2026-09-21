"""Trainable PyrCombineLayerB (gun + SR outcome + comm -> next gun logits, next comm logit).

This module defines :class:`PyrCombineLayerB`, a trainable Keras layer that
combines the current "gun" representation, the shared resource (SR) outcome,
and a communication bit, and produces **logits** for the next pyramid level.

Keras 3 build note:
All sublayers are created in :meth:`build` based on the gun width ``L``.
No state is created in :meth:`call`.

Internal-training variant:
Inputs are expected in the scaled/logit domains used during training:
  * ``gun_batch``: scaled values (often near ``{-0.5, +0.5}``)
  * ``sr_outcome_batch``: SR outcome logits (often near ``{-beta, +beta}``)
  * ``comm_batch``: communication-bit logit with shape ``(B, 1)``
Outputs are logits (no sigmoid). Logical bit values represented as logits are
determined by logit sign.

Layer interface:
  call(gun_batch, sr_outcome_batch, comm_batch, training=False)
    -> (next_gun_logits, next_comm_logit)

Expected shapes (rank-2, batch-major):
  * ``gun_batch``: ``(B, L)``          (current level width ``L``)
  * ``sr_outcome_batch``: ``(B, L/2)`` (must match ``L//2``)
  * ``comm_batch``: ``(B, 1)``
  * ``next_gun_logits``: ``(B, L/2)``
  * ``next_comm_logit``: ``(B, 1)``

MLP structure:
  * concat([gun_batch, sr_outcome_batch, comm_batch]) -> (B, 3L/2 + 1)
  * Dense(hidden_units, relu)
  * Dense(L/2, linear) for next gun logits
  * Dense(1, linear)   for next comm logit
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import tensorflow as tf


def _ensure_rank2(x: tf.Tensor, name: str) -> None:
    """Validate that ``x`` is rank-2 (batch, features) when statically known."""
    if x.shape.rank is not None and x.shape.rank != 2:
        raise ValueError(f"{name} must be rank-2 (B, D). Got rank={x.shape.rank}, shape={x.shape}.")


def _require_known_last_dim(shape: tf.TensorShape, name: str) -> int:
    """Require that the last dimension of a TensorShape is statically known."""
    d = shape[-1]
    if d is None:
        raise ValueError(f"{name} last dimension must be statically known. Got shape={shape}.")
    return int(d)


class PyrCombineLayerB(tf.keras.layers.Layer):
    """Combine gun state, SR outcome, and comm into next-step logits.

    This layer is used at a pyramid level transition. It consumes:
      * the current gun representation,
      * the shared resource (SR) outcome for the level,
      * the current communication-bit logit,

    and produces logits for:
      * the next-level gun representation (width ``L//2``),
      * the next communication bit.

    Notes:
      * This layer operates in logit space; it does not apply sigmoid.
      * ``gun_batch`` must have an even last dimension ``L`` so that ``L//2``
        is well-defined for the next level.
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
            hidden_units: Number of hidden units in the intermediate Dense layer.
            name: Layer name.
            dtype: Layer dtype. Inputs are converted to this dtype (or float32 if
                not provided).
            **kwargs: Passed to the base Keras Layer.
        """
        super().__init__(name=name, dtype=dtype, trainable=True, **kwargs)
        if hidden_units < 1:
            raise ValueError("hidden_units must be >= 1.")
        self.hidden_units = int(hidden_units)

        # Created in build().
        self._dense_hidden: Optional[tf.keras.layers.Dense] = None
        self._dense_gun: Optional[tf.keras.layers.Dense] = None
        self._dense_comm: Optional[tf.keras.layers.Dense] = None
        self._built_for_L: Optional[int] = None

    def build(self, input_shape: Any) -> None:
        """Create sublayers using the statically-known gun width ``L``.

        Keras may pass only the first input's shape for multi-input layers,
        so this method derives dimensions from the gun input shape.
        """
        # Keras may pass only gun shape for multi-input layers.
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
        # IMPORTANT: heads emit logits (no sigmoid).
        self._dense_gun = tf.keras.layers.Dense(
            gun_out_dim,
            activation=None,
            name="dense_gun",
            dtype=self.dtype,
        )

        # NOTE: The comm head uses an explicit initializer (small stddev) to
        # reduce early-step saturation when the comm logit is used downstream.
        # The call() method also adds a residual connection from comm_batch.
        self._dense_comm = tf.keras.layers.Dense(
            1,
            activation=None,
            name="dense_comm",
            dtype=self.dtype,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer="zeros",
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
        """Run the forward pass.

        Args:
            gun_batch: Tensor of shape ``(B, L)`` representing the current gun
                state (typically scaled values during training).
            sr_outcome_batch: Tensor of shape ``(B, L//2)`` representing SR
                outcome logits aligned to the current level.
            comm_batch: Tensor of shape ``(B, 1)`` containing the current comm
                bit as a logit.
            training: Whether to run in training mode (passed to Dense layers).
            **kwargs: Unused; accepted for Keras compatibility.

        Returns:
            A tuple ``(next_gun_logits, next_comm_logit)`` where:
              * ``next_gun_logits`` has shape ``(B, L//2)``
              * ``next_comm_logit`` has shape ``(B, 1)``
        """
        gun_batch = tf.convert_to_tensor(gun_batch, dtype=self.dtype or tf.float32)
        sr_outcome_batch = tf.convert_to_tensor(sr_outcome_batch, dtype=self.dtype or tf.float32)
        comm_batch = tf.convert_to_tensor(comm_batch, dtype=self.dtype or tf.float32)

        _ensure_rank2(gun_batch, "gun_batch")
        _ensure_rank2(sr_outcome_batch, "sr_outcome_batch")
        _ensure_rank2(comm_batch, "comm_batch")

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
        h = self._dense_hidden(x, training=training)
        next_gun_logits = self._dense_gun(h, training=training)

        # Residual update for comm: predict a delta (logit-space) and add it to
        # the current comm logit.
        next_comm_logit = self._dense_comm(h, training=training) + comm_batch

        return next_gun_logits, next_comm_logit

    def get_config(self) -> Dict[str, Any]:
        """Return the serialized configuration for Keras."""
        cfg = super().get_config()
        cfg.update({"hidden_units": self.hidden_units})
        return cfg