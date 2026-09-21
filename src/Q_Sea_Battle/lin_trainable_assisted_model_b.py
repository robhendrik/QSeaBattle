"""Backward-compatible wrapper for the trainable linear PR-assisted Model B.

This module preserves the historical constructor interface used by external
player code while delegating the actual implementation to
:class:`~.lin_internal_model_b.LinInternalModelB`.

The wrapper adapts legacy shared resource (SR) mode names to the current SR
modes used by the internal model:
- ``"stochastic"``: SR is sampled (legacy alias: ``"sample"``).
- ``"replay"``: SR uses an expected/replayed value (legacy alias: ``"expected"``).

Only documentation and compatibility glue live here; the learning logic and
logit-based interfaces are implemented in ``LinInternalModelB``.
"""

from __future__ import annotations

from typing import Any, Optional

from .lin_internal_model_b import LinInternalModelB


def _map_sr_mode(sr_mode: str) -> str:
    """Normalize legacy SR mode names to the internal model's accepted values.

    Args:
        sr_mode: User-facing SR mode string. Common legacy values include
            ``"sample"`` and ``"expected"``.

    Returns:
        Normalized mode string accepted by :class:`LinInternalModelB`:
        ``"stochastic"`` or ``"replay"``.

    Raises:
        ValueError: If ``sr_mode`` is not a supported alias/value.
    """
    mode = str(sr_mode).lower()
    if mode in {"sample", "stochastic"}:
        return "stochastic"
    if mode in {"expected", "replay"}:
        return "replay"
    raise ValueError(f"Unsupported sr_mode={sr_mode!r}. Use replay/expected or stochastic/sample.")


class LinTrainableAssistedModelB(LinInternalModelB):
    """Trainable PR-assisted Model B with a legacy-compatible constructor.

    This class subclasses :class:`LinInternalModelB` and provides:
    - Constructor parameters historically used by player code (``field_size``,
      ``comms_size``, and legacy ``sr_mode`` aliases).
    - A minimal ``layout`` object with the attributes required by the internal
      model.

    All learning behavior and internal logit semantics are inherited from
    :class:`LinInternalModelB`.
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
        """Initialize the model using legacy layout arguments.

        Args:
            field_size: Game field size used to construct the internal layout.
            comms_size: Communication channel size used to construct the internal
                layout.
            sr_mode: Shared resource mode. Legacy aliases are accepted:
                    ``"sample"`` -> ``"stochastic"``, ``"expected"`` -> ``"replay"``.
                    New code should use "stochastic" or "replay" directly.
            p_rule: Probability of applying the rule-based component (see
                internal model for details).
            beta: Internal hyperparameter forwarded to the internal model.
            alpha: Internal hyperparameter forwarded to the internal model.
            seed: Optional RNG seed forwarded to the internal model.
            hidden_units_measure: Width of the measurement subnetwork forwarded
                to the internal model.
            hidden_units_combine: Width of the combine subnetwork forwarded to
                the internal model.
            name: Optional model name forwarded to the internal model.
            **kwargs: Accepted for backward compatibility with older call sites.
                Currently unused by this wrapper.

        Raises:
            ValueError: If ``sr_mode`` is not a supported value/alias.
        """
        self.field_size = int(field_size)
        self.comms_size = int(comms_size)

        # The internal model expects a layout-like object with `field_size` and
        # `comms_size` attributes. Use an ad-hoc object to preserve the legacy
        # constructor signature without importing additional layout types.
        class _Layout:
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