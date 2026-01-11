"""SharedRandomnessLayer (deprecated; Keras layer).

This module is kept for backward compatibility.

- New module: ``pr_assisted_layer.py``
- New class: :class:`~Q_Sea_Battle.pr_assisted_layer.PRAssistedLayer`

Any use of :class:`SharedRandomnessLayer` will emit a ``DeprecationWarning`` and
delegate all behaviour to :class:`PRAssistedLayer`.

Author: Rob Hendriks
Version: 0.1
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import tensorflow as tf

from .pr_assisted_layer import PRAssistedLayer, _PRInputs


@dataclass(frozen=True)
class _SRInputs:
    """Validated inputs for :class:`SharedRandomnessLayer` (deprecated alias).

    Attributes:
        current_measurement: Tensor of shape ``(..., length)``.
        previous_measurement: Tensor of shape ``(..., length)``.
        previous_outcome: Tensor of shape ``(..., length)``.
        first_measurement: Tensor broadcastable to ``(..., length)``.
    """

    current_measurement: tf.Tensor
    previous_measurement: tf.Tensor
    previous_outcome: tf.Tensor
    first_measurement: tf.Tensor


class SharedRandomnessLayer(PRAssistedLayer):
    """Deprecated alias for :class:`~Q_Sea_Battle.pr_assisted_layer.PRAssistedLayer`.

    Args:
        length: Number of bits in each measurement/outcome vector.
        p_high: Correlation parameter in ``[0, 1]``.
        mode: Either ``"expected"`` or ``"sample"``.
        resource_index: Optional identifier for this shared resource.
        seed: Optional integer seed used for deterministic sampling in
            ``mode='sample'``.
        name: Optional layer name.
        **kwargs: Forwarded to ``tf.keras.layers.Layer``.

    Notes:
        The underlying implementation is :class:`PRAssistedLayer`. This class only
        exists to preserve old import paths.
    """

    def __init__(
        self,
        length: int,
        p_high: float,
        mode: str = "expected",
        resource_index: Optional[int] = None,
        seed: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        warnings.warn(
            "SharedRandomnessLayer is deprecated; use PRAssistedLayer from "
            "Q_Sea_Battle.pr_assisted_layer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            length=length,
            p_high=p_high,
            mode=mode,
            resource_index=resource_index,
            seed=seed,
            name=name,
            **kwargs,
        )

    # Keep a compatible return type name for internal callers/tests that may inspect it.
    def _validate_inputs(self, inputs: Dict[str, tf.Tensor]) -> _SRInputs:  # type: ignore[override]
        v = super()._validate_inputs(inputs)
        # v is _PRInputs; re-wrap as _SRInputs for backward compatibility.
        assert isinstance(v, _PRInputs)
        return _SRInputs(
            current_measurement=v.current_measurement,
            previous_measurement=v.previous_measurement,
            previous_outcome=v.previous_outcome,
            first_measurement=v.first_measurement,
        )
