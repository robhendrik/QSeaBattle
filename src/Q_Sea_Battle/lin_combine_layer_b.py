"""Trainable LinCombineLayerB (SR outcome logits + comm logits -> shoot logit).

This module defines a small Keras layer used in the QSeaBattle linear setup to
combine:
- SR outcome logits (one logit per SR bit), and
- communication logits (one logit per comm channel)

into a single shoot logit. Optionally, the layer can also return an intermediate
"flip" logit intended to represent a parity-like decision that modulates the
communication signal.

Tensor semantics
----------------
All inputs/outputs are logits. Logical bit values represented as logits are
determined by the sign of the logit.

Key idea
--------
The layer explicitly includes a multiplicative interaction term between the
communication signal and the learned flip logit. This enables XOR-like
composition that is not linearly separable using only linear terms.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, Optional

import tensorflow as tf


def _ensure_rank2(x: tf.Tensor, name: str) -> None:
    """Validate that a tensor is rank-2.

    Args:
        x: Tensor expected to have shape (B, D).
        name: Name used in error messages.

    Raises:
        ValueError: If `x` has a known rank and it is not 2.
    """
    if x.shape.rank is not None and x.shape.rank != 2:
        raise ValueError(f"{name} must be rank-2 (B, D). Got rank={x.shape.rank}, shape={x.shape}.")


def _normalize_hidden_units(hidden_units: int | Sequence[int]) -> tuple[int, ...]:
    """Normalize `hidden_units` to a tuple of ints.

    Args:
        hidden_units: Either a single layer width (int) or a sequence of layer
            widths.

    Returns:
        Tuple of hidden layer widths.
    """
    if isinstance(hidden_units, int):
        return (int(hidden_units),)
    return tuple(int(u) for u in hidden_units)


class LinCombineLayerB(tf.keras.layers.Layer):
    """Map (SR outcome logits, comm logits) to a shoot logit, with an optional flip logit.

    The layer implements a two-stage decomposition:

    1) A small MLP maps `outcome_batch` to a scalar `flip_logit` per batch element.
       This is intended to represent a parity-like decision derived from the SR
       outcome bits.
    2) The final head combines the first communication channel with `flip_logit`
       using both linear terms and a multiplicative interaction term
       (`comm * flip`) to produce `shoot_logit`.

    Args:
        comms_size: Number of communication channels `m`. The current computation
            uses only the first channel, but the argument is kept for interface
            compatibility.
        hidden_units: Hidden-layer widths for the parity/flip subnetwork. May be
            an int (single hidden layer) or a sequence of ints (stack of Dense
            layers with ReLU activations).
        name: Optional Keras layer name.
        dtype: Optional Keras dtype for layer variables and computations.
        **kwargs: Forwarded to `tf.keras.layers.Layer`.
    """

    def __init__(
        self,
        comms_size: int,
        hidden_units: int | Sequence[int] = (64, 64),
        name: Optional[str] = None,
        dtype: Optional[tf.dtypes.DType] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, dtype=dtype, trainable=True, **kwargs)
        if comms_size < 1:
            raise ValueError("comms_size must be >= 1.")

        self.comms_size = int(comms_size)
        self.hidden_units = _normalize_hidden_units(hidden_units)

        # Parity/flip subnetwork: outcome_batch -> flip_logit.
        self._mlp: list[tf.keras.layers.Layer] = []
        self._dense_flip: Optional[tf.keras.layers.Dense] = None

        # Final shoot head on engineered interaction features.
        self._dense_shoot: Optional[tf.keras.layers.Dense] = None

    def build(self, input_shape: Any) -> None:
        """Create sub-layers.

        Note:
            The Dense layers infer input dimensionality at first call, so no
            specific `input_shape` contract is required here beyond rank-2 usage
            in `call()`.
        """
        self._mlp = [
            tf.keras.layers.Dense(
                int(u),
                activation="relu",
                name=f"dense_hidden_{i}",
                dtype=self.dtype,
            )
            for i, u in enumerate(self.hidden_units)
        ]

        self._dense_flip = tf.keras.layers.Dense(
            1,
            activation=None,
            name="dense_flip",
            dtype=self.dtype,
        )

        # Shoot head sees a compact engineered feature vector:
        #   [comm, flip, comm * flip]
        #
        # Including the interaction term provides a straightforward route to
        # learn XOR-like composition while retaining linear correction terms.
        self._dense_shoot = tf.keras.layers.Dense(
            1,
            activation=None,
            name="dense_shoot",
            dtype=self.dtype,
        )

        super().build(input_shape)

    def call(
        self,
        outcome_batch: tf.Tensor,
        comm_batch: tf.Tensor,
        training: bool = False,
        return_flip: bool = False,
        **kwargs: Any,
    ) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor]:
        """Run the layer.

        Args:
            outcome_batch: SR outcome logits with shape (B, n2).
            comm_batch: Communication logits with shape (B, m).
            training: Whether to run in training mode (forwarded to sub-layers).
            return_flip: If True, return a tuple `(shoot_logit, flip_logit)`.
            **kwargs: Unused; accepted for Keras call compatibility.

        Returns:
            If `return_flip` is False: `shoot_logit` with shape (B, 1).
            If `return_flip` is True: Tuple `(shoot_logit, flip_logit)`, each of
            shape (B, 1).

        Raises:
            ValueError: If inputs are not rank-2 when rank is known.
            RuntimeError: If the layer has not been built correctly.
        """
        outcome_batch = tf.convert_to_tensor(outcome_batch, dtype=self.dtype or tf.float32)
        comm_batch = tf.convert_to_tensor(comm_batch, dtype=self.dtype or tf.float32)

        _ensure_rank2(outcome_batch, "outcome_batch")
        _ensure_rank2(comm_batch, "comm_batch")

        if self._dense_flip is None or self._dense_shoot is None:
            raise RuntimeError("LinCombineLayerB is not built correctly.")

        # ------------------------------------------------------------
        # 1) Parity/flip subnetwork
        # ------------------------------------------------------------
        x = outcome_batch
        for layer in self._mlp:
            x = layer(x, training=training)
        flip_logit = self._dense_flip(x, training=training)  # (B, 1)

        # ------------------------------------------------------------
        # 2) Shoot composition
        # ------------------------------------------------------------
        # Use only the first comm channel as the scalar comm signal, consistent
        # with the existing implementation and `comms_size`-agnostic callers.
        comm_scalar = comm_batch[:, :1]  # (B, 1)

        # Provide linear terms plus the multiplicative interaction. The
        # interaction term makes XOR-like composition learnable by a linear head
        # over these engineered features.
        shoot_features = tf.concat(
            [comm_scalar, flip_logit, comm_scalar * flip_logit],
            axis=-1,
        )  # (B, 3)

        shoot_logit = self._dense_shoot(shoot_features, training=training)  # (B, 1)

        if return_flip:
            return shoot_logit, flip_logit
        return shoot_logit

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for Keras serialization."""
        cfg = super().get_config()
        cfg.update(
            {
                "comms_size": self.comms_size,
                "hidden_units": self.hidden_units,
            }
        )
        return cfg