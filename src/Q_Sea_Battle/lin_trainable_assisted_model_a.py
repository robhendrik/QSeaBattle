"""Trainable linear PR-assisted model A.

This module provides a backward-compatible wrapper around
:class:`~.lin_internal_model_a.LinInternalModelA`.

Historically, player code constructed a "trainable assisted" model by passing
``field_size`` and ``comms_size`` directly. The internal implementation,
however, expects a layout-like object with these attributes. This wrapper keeps
the legacy constructor interface while delegating all behavior to
``LinInternalModelA``.

The shared resource (SR) mode argument is also normalized for compatibility:
``sample``/``stochastic`` map to ``"stochastic"`` and ``expected``/``replay``
map to ``"replay"``.
"""

from __future__ import annotations

from typing import Any, Optional

from .lin_internal_model_a import LinInternalModelA


def _map_sr_mode(sr_mode: str) -> str:
    """Normalize legacy SR mode aliases to the internal SR mode strings.

    Args:
        sr_mode: User-provided SR mode string. Common legacy aliases include
            ``"sample"``/``"stochastic"`` and ``"expected"``/``"replay"``.

    Returns:
        The normalized SR mode string used by the internal model: ``"stochastic"``
        or ``"replay"``.

    Raises:
        ValueError: If ``sr_mode`` is not a supported alias.
    """
    mode = str(sr_mode).lower()
    if mode in {"sample", "stochastic"}:
        return "stochastic"
    if mode in {"expected", "replay"}:
        return "replay"
    raise ValueError(f"Unsupported sr_mode={sr_mode!r}. Use replay/expected or stochastic/sample.")


class LinTrainableAssistedModelA(LinInternalModelA):
    """Backward-compatible wrapper for the trainable PR-assisted linear model.

    This class exists to preserve the historical constructor signature used by
    player code. It adapts ``field_size`` and ``comms_size`` into a minimal
    layout object (with matching attributes) expected by
    :class:`~.lin_internal_model_a.LinInternalModelA`.

    Attributes:
        field_size: Game field size passed to the constructor.
        comms_size: Communication channel size passed to the constructor.

    New code should use "stochastic" or "replay" directly.
    """

    def __init__(
        self,
        field_size: int,
        comms_size: int,
        *,
        sr_mode: str = "sample",
        p_rule: float = 1.0,
        beta: float = 10.0,
        alpha: float = 5.0,
        seed: Optional[int] = None,
        hidden_units_measure: int = 64,
        hidden_units_combine: int = 64,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the model using the legacy constructor interface.

        Args:
            field_size: Size of the game field.
            comms_size: Size of the communication channel.
            sr_mode: SR mode. Legacy aliases are accepted and normalized via
                :func:`_map_sr_mode`.
            p_rule: Probability of applying the rule-based/shared-resource path,
                as used by the internal model.
            beta: Internal model hyperparameter forwarded to the base class.
            alpha: Internal model hyperparameter forwarded to the base class.
            seed: Optional random seed forwarded to the base class.
            hidden_units_measure: Hidden units for the internal "measure" MLP.
            hidden_units_combine: Hidden units for the internal "combine" MLP.
            name: Optional model name.
            **kwargs: Accepted for backward compatibility. Currently unused.

        Notes:
            ``kwargs`` is intentionally ignored to preserve older call sites that
            passed additional arguments.
        """
        self.field_size = int(field_size)
        self.comms_size = int(comms_size)

        class _Layout:
            """Minimal layout object expected by LinInternalModelA."""

            pass

        layout = _Layout()
        layout.field_size = self.field_size
        layout.comms_size = self.comms_size

        super().__init__(
            layout,
            sr_mode=_map_sr_mode(sr_mode),
            p_rule=p_rule,
            beta=beta,
            alpha=alpha,
            seed=seed,
            hidden_units_measure=hidden_units_measure,
            hidden_units_combine=hidden_units_combine,
            name=name,
        )