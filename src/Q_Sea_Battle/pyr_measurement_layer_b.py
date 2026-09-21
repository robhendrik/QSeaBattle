"""Trainable PyrMeasurementLayerB (gun -> measurement logits).

This module defines :class:`PyrMeasurementLayerB`, a small trainable Keras layer used
in the Pyramid (Pyr) assisted architecture.

Keras 3 build note:
Sublayers are created in :meth:`build` based on the (statically known) input width
``L``. No state is created in :meth:`call`.

Internal training variant:
* Inputs are expected in the **scaled** domain (typically values in ``{-0.5, +0.5}``).
* Outputs are **logits** (no sigmoid). Downstream code may apply sigmoid/DRU/etc.

Layer interface (contract-aligned):
* ``call(gun_batch) -> meas_logits``
* ``gun_batch``: rank-2 tensor of shape ``(B, L)`` (cropped; ``L`` is the active width
  ``L_d``)
* ``meas_logits``: rank-2 tensor of shape ``(B, L/2)`` (cropped; equals ``k_d``)

Implementation:
A simple MLP head:
* Dense(hidden_units, activation="relu")
* Dense(L/2, activation=None)  # logits

No rule-based / teacher mapping is implemented here.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import tensorflow as tf


def _ensure_rank2(x: tf.Tensor, name: str) -> None:
    """Validate that ``x`` is rank-2 when its rank is statically known.

    This is a lightweight shape check that runs only when TensorFlow can infer the
    rank at trace/build time.

    Args:
        x: Tensor to validate.
        name: Human-readable tensor name used in error messages.

    Raises:
        ValueError: If the static rank is known and is not 2.
    """
    if x.shape.rank is not None and x.shape.rank != 2:
        raise ValueError(f"{name} must be rank-2 (B, D). Got rank={x.shape.rank}, shape={x.shape}.")


def _require_known_last_dim(shape: tf.TensorShape, name: str) -> int:
    """Require a statically known last dimension and return it as ``int``.

    The layer constructs its output head based on ``L`` (the input width), so the
    last dimension must be known during :meth:`build`.

    Args:
        shape: TensorShape to inspect.
        name: Human-readable tensor name used in error messages.

    Returns:
        The last dimension as a Python ``int``.

    Raises:
        ValueError: If the last dimension is not statically known.
    """
    d = shape[-1]
    if d is None:
        raise ValueError(f"{name} last dimension must be statically known. Got shape={shape}.")
    return int(d)


class PyrMeasurementLayerB(tf.keras.layers.Layer):
    """Map a gun-state vector to measurement logits.

    The mapping is learned (trainable) and implemented as a small MLP. The output is
    logits (no sigmoid), where downstream components interpret bit values from the
    logit sign or via an applied squashing/noise function.

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
            hidden_units: Width of the hidden ReLU layer. Must be >= 1.
            name: Optional Keras layer name.
            dtype: Optional dtype for layer weights and computations.
            **kwargs: Forwarded to the Keras base Layer.
        """
        super().__init__(name=name, dtype=dtype, trainable=True, **kwargs)
        if hidden_units < 1:
            raise ValueError("hidden_units must be >= 1.")
        self.hidden_units = int(hidden_units)

        # Created in build() once the input width L is known.
        self._dense_hidden: Optional[tf.keras.layers.Dense] = None
        self._dense_out: Optional[tf.keras.layers.Dense] = None
        self._built_for_L: Optional[int] = None

    def build(self, input_shape: Any) -> None:
        """Create sublayers based on the input width ``L``.

        Args:
            input_shape: Shape of ``gun_batch``. The last dimension must be statically
                known and even (so that ``L/2`` is an integer).

        Raises:
            ValueError: If ``L`` is unknown or not even.
        """
        x_shape = tf.TensorShape(input_shape)
        L = _require_known_last_dim(x_shape, "gun_batch")
        if L % 2 != 0:
            raise ValueError(f"gun_batch last dimension L must be even so that L/2 is integer. Got L={L}.")
        out_dim = L // 2

        self._dense_hidden = tf.keras.layers.Dense(
            self.hidden_units,
            activation="relu",
            name="dense_hidden",
            dtype=self.dtype,
        )
        # Output head produces logits (no sigmoid).
        self._dense_out = tf.keras.layers.Dense(
            out_dim,
            activation=None,
            name="dense_out",
            dtype=self.dtype,
        )
        self._built_for_L = L
        super().build(input_shape)

    def call(self, gun_batch: tf.Tensor, training: bool = False, **kwargs: Any) -> tf.Tensor:
        """Run a forward pass.

        Args:
            gun_batch: Gun-state tensor of shape ``(B, L)``.
            training: Keras training flag forwarded to sublayers.
            **kwargs: Unused. Present for Keras compatibility.

        Returns:
            Measurement logits tensor of shape ``(B, L/2)``.

        Raises:
            RuntimeError: If sublayers are missing (layer not built correctly).
            ValueError: If ``gun_batch`` is not rank-2 when its rank is statically known.
        """
        x = tf.convert_to_tensor(gun_batch, dtype=self.dtype or tf.float32)

        _ensure_rank2(x, "gun_batch")

        # Dynamic check to catch mismatched widths when tracing with unknown shapes.
        tf.debugging.assert_equal(
            tf.shape(x)[-1] % 2,
            0,
            message="gun_batch last dimension L must be even so that L/2 is integer.",
        )

        if self._dense_hidden is None or self._dense_out is None:
            raise RuntimeError("PyrMeasurementLayerB is not built correctly (missing sublayers).")

        h = self._dense_hidden(x, training=training)
        meas_logits = self._dense_out(h, training=training)
        return meas_logits

    def get_config(self) -> Dict[str, Any]:
        """Return the Keras serialization config."""
        cfg = super().get_config()
        cfg.update({"hidden_units": self.hidden_units})
        return cfg