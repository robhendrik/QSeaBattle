"""Pyramid Trainable Assisted Model A.

This module defines :class:`~Q_Sea_Battle.pyr_trainable_assisted_model_a.PyrTrainableAssistedModelA`,
the Player-A model for the pyramid (Pyr) assisted architecture.

Update (per-level layers)
-------------------------
This version supports **per-level trained layers** (Option 2A: train each level separately)
by storing measurement/combine layers as lists:

- ``measure_layers[level]`` consumes length ``L`` and outputs ``L/2``.
- ``combine_layers[level]`` consumes ``(state, sr_outcome)`` and outputs ``L/2``.

If no custom layers are supplied, the model instantiates the deterministic Step-1
pyramid primitives at each level, preserving Step-2 behavior and tests.

Backward compatibility
----------------------
For legacy code that expects ``model.measure_layer`` / ``model.combine_layer``, we
keep these attributes as aliases to the first level's layers. The forward pass uses
the per-level lists.

Notes
-----
This module intentionally uses a *hard* (non-trainable) conversion from the final
bit to a logit, because Step-2 focuses on interface and contract testing rather
than learning.

For gameplay, the SharedRandomnessLayer must be used in ``mode="sample"`` as per
the v2 spec.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

import tensorflow as tf

from .pyr_measurement_layer_a import PyrMeasurementLayerA
from .pyr_combine_layer_a import PyrCombineLayerA
from .shared_randomness_layer import SharedRandomnessLayer


def _infer_n2_and_m(game_layout: Any) -> tuple[int, int]:
    """Infer (n2, m) from a GameLayout-like object."""
    if hasattr(game_layout, "n2"):
        n2 = int(getattr(game_layout, "n2"))
    else:
        field_size = int(getattr(game_layout, "field_size"))
        n2 = field_size * field_size

    m = int(getattr(game_layout, "comms_size", getattr(game_layout, "M", 1)))
    return n2, m


def _validate_power_of_two(n: int) -> int:
    """Return log2(n) if n is a power of two, else raise ValueError."""
    if n <= 0:
        raise ValueError("n2 must be positive.")
    k = int(round(math.log2(n)))
    if 2**k != n:
        raise ValueError(f"n2 must be a power of 2; got n2={n}.")
    return k


class PyrTrainableAssistedModelA(tf.keras.Model):
    """Player-A pyramid assisted model (Step 2, per-level layers)."""

    def __init__(
        self,
        game_layout: Any,
        p_high: float = 0.9,
        sr_mode: str = "sample",
        measure_layers: Optional[Sequence[tf.keras.layers.Layer]] = None,
        combine_layers: Optional[Sequence[tf.keras.layers.Layer]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Create a pyramid Model A.

        Parameters
        ----------
        game_layout:
            GameLayout-like object with ``field_size`` (or ``n2``) and ``comms_size``.
        p_high:
            Correlation parameter forwarded to each SharedRandomnessLayer.
        sr_mode:
            "sample" for gameplay / dataset generation (required), or "expected" for analysis.
        measure_layers:
            Optional per-level measurement layers. If provided, length must equal K.
        combine_layers:
            Optional per-level combine layers. If provided, length must equal K.
        name:
            Optional Keras model name.
        """
        super().__init__(name=name)
        self.n2, self.M = _infer_n2_and_m(game_layout)
        if self.M != 1:
            raise ValueError(f"Pyr architecture requires comms_size==1; got {self.M}.")
        self.depth = _validate_power_of_two(self.n2)

        # --- Per-level measurement/combine layers ---
        if measure_layers is None:
            self.measure_layers: List[tf.keras.layers.Layer] = [PyrMeasurementLayerA() for _ in range(self.depth)]
        else:
            if len(measure_layers) != self.depth:
                raise ValueError(f"measure_layers must have length K={self.depth}; got {len(measure_layers)}.")
            self.measure_layers = list(measure_layers)

        if combine_layers is None:
            self.combine_layers: List[tf.keras.layers.Layer] = [PyrCombineLayerA() for _ in range(self.depth)]
        else:
            if len(combine_layers) != self.depth:
                raise ValueError(f"combine_layers must have length K={self.depth}; got {len(combine_layers)}.")
            self.combine_layers = list(combine_layers)

        # Backward-compat aliases (legacy code may reference these).
        self.measure_layer = self.measure_layers[0]
        self.combine_layer = self.combine_layers[0]

        # --- Shared randomness (one resource per level) ---
        self.sr_layers: List[SharedRandomnessLayer] = []
        active = self.n2
        for level in range(self.depth):
            length = active // 2
            self.sr_layers.append(
                SharedRandomnessLayer(length=length, p_high=p_high, mode=sr_mode, resource_index=level)
            )
            active //= 2

    def call(self, field_batch: tf.Tensor) -> tf.Tensor:
        """Compute comm logits from the field."""
        comm_logits, _, _ = self.compute_with_internal(field_batch)
        return comm_logits

    def compute_with_internal(
        self, field_batch: tf.Tensor
    ) -> Tuple[tf.Tensor, List[tf.Tensor], List[tf.Tensor]]:
        """Forward pass returning comm logits plus per-level tensors."""
        x = tf.convert_to_tensor(field_batch)
        if x.shape.rank != 2:
            raise ValueError(f"field_batch must be rank-2 (B,n2); got shape {x.shape}.")

        measurements: List[tf.Tensor] = []
        outcomes: List[tf.Tensor] = []

        state = tf.cast(x, tf.float32)

        for level, sr in enumerate(self.sr_layers):
            meas_layer = self.measure_layers[level]
            comb_layer = self.combine_layers[level]

            meas = meas_layer(state)  # (B, L/2)

            # First measurement for the SR layer: previous_* are ignored, but we pass zeros.
            zeros = tf.zeros_like(meas)
            first_flag = tf.ones((tf.shape(meas)[0], 1), dtype=tf.float32)
            out = sr(
                {
                    "current_measurement": meas,
                    "previous_measurement": zeros,
                    "previous_outcome": zeros,
                    "first_measurement": first_flag,
                }
            )
            next_state = comb_layer(state, out)

            measurements.append(tf.cast(meas, tf.float32))
            outcomes.append(tf.cast(out, tf.float32))
            state = tf.cast(next_state, tf.float32)

        # state is (B,1). Convert hard bit to a logit for downstream Bernoulli usage.
        bit = tf.clip_by_value(state, 0.0, 1.0)
        comm_logits = (bit * 2.0 - 1.0) * 10.0  # {-10,+10}
        return comm_logits, measurements, outcomes
