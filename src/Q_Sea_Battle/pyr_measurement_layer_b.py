"""Pyramid measurement layer for Player B.

This module defines :class:`~Q_Sea_Battle.PyrMeasurementLayerB`, a Keras layer
used in the Pyramid (Pyr) assisted architecture.

The Step 1 implementation encodes the teacher measurement rule described in the
QSeaBattle specification: pairwise "¬even AND odd".

Author: Rob Hendriks
Package: Q_Sea_Battle
Version: 0.1
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf


class PyrMeasurementLayerB(tf.keras.layers.Layer):
    """Compute Player B's per-level measurement vector.

    Purpose:
        Given a binary gun state ``G^ℓ`` of length ``L`` (even), produce the
        measurement vector ``M_B^ℓ`` of length ``L/2`` defined per pair:

            M_B^ℓ[i] = (NOT G^ℓ[2*i]) AND G^ℓ[2*i + 1].

        This matches the "¬even AND odd" rule in the Pyr dataset spec.

        This layer is intentionally *non-trainable* for Step 1: it encodes the
        reference/teacher rule so dataset generation and early integration
        tests can be validated.

    Args:
        name: Optional Keras layer name.

    Call Args:
        gun_batch: Tensor of shape ``(B, L)`` with values in ``{0, 1}``.
            ``L`` must be even.

    Returns:
        Tensor of shape ``(B, L/2)`` with values in ``{0, 1}`` (dtype float32).
    """

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name, trainable=False)

    def call(self, gun_batch: tf.Tensor) -> tf.Tensor:
        """Compute pairwise "¬even AND odd" measurements.

        Args:
            gun_batch: Tensor of shape ``(B, L)`` with values in ``{0, 1}``.

        Returns:
            Tensor of shape ``(B, L/2)`` in ``{0, 1}``.
        """
        g = tf.convert_to_tensor(gun_batch, dtype=tf.float32)
        L = tf.shape(g)[-1]
        tf.debugging.assert_equal(L % 2, 0, message="Active length L must be even.")
        pairs = tf.reshape(g, tf.concat([tf.shape(g)[:-1], [L // 2, 2]], axis=0))
        even = pairs[..., 0]
        odd = pairs[..., 1]
        meas = (1.0 - even) * odd
        # Ensure {0,1} (in case inputs weren't strictly binary floats)
        meas = tf.clip_by_value(meas, 0.0, 1.0)
        return meas
