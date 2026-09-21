"""Trainable PR-assisted player wiring (A/B) for QSeaBattle.

This module provides a `Players`-style wrapper that wires together two player
wrappers which act as a coordinated pair:

- `TrainableAssistedPlayerA` produces communication bits (as logits) and stores
  intermediate tensors from the internal model in `previous`.
- `TrainableAssistedPlayerB` consumes the stored tensors in `previous` and
  decides the shoot action (optionally sampling when `explore=True`).

The contract for `previous` is::

    previous == (measurements_per_layer, outcomes_per_layer)

Both entries are Python lists of tensors. Each tensor is expected to have shape
`(B, n2)` where `B` is the batch dimension. The precise meaning of
"measurements" and "outcomes" is defined by the underlying trainable models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

try:
    # These base classes exist in the project.
    from .players import Players  # type: ignore
except Exception:  # pragma: no cover
    class Players:  # minimal fallback for unit tests
        """Fallback `Players` base class.

        This is only used if the project base class cannot be imported (e.g., in
        isolated unit tests). It intentionally exposes only the attributes used
        by this module.
        """

        has_log_probs: bool = False

from .trainable_assisted_player_a import TrainableAssistedPlayerA
from .trainable_assisted_player_b import TrainableAssistedPlayerB

from .lin_trainable_assisted_model_a import LinTrainableAssistedModelA
from .lin_trainable_assisted_model_b import LinTrainableAssistedModelB


class TrainableAssistedPlayers(Players):
    """Factory/wrapper that provides a coordinated (A, B) PR-assisted player pair.

    The instance owns the two trainable models and exposes `players()` to obtain
    the two player wrappers that integrate with the tournament/game interfaces.

    Attributes:
        game_layout: Game-layout-like object. Must provide `field_size` and
            `comms_size` attributes.
        model_a: Internal trainable model used by player A.
        model_b: Internal trainable model used by player B.
        explore: Shared exploration flag. When True, players may sample actions
            and record log-probabilities; when False, they act greedily.
        previous: Storage written by player A and consumed by player B. See the
            module docstring for the expected structure.
    """

    has_log_probs: bool = True

    def __init__(
        self,
        game_layout: Any,
        p_rule: float = 0.9,
        num_iterations: Optional[int] = None,
        hidden_dim: int = 32,
        L_meas: Optional[int] = None,
        model_a: Optional[LinTrainableAssistedModelA] = None,
        model_b: Optional[LinTrainableAssistedModelB] = None,
    ) -> None:
        """Initialize the paired players and (optionally) their models.

        Args:
            game_layout: Game-layout-like object providing `field_size` and
                `comms_size`.
            p_rule: Unused by the current linear models. Kept for compatibility
                with other/older configurations.
            num_iterations: Unused by the current linear models. Kept for
                compatibility with other/older configurations.
            hidden_dim: Unused by the current linear models. Kept for
                compatibility with other/older configurations.
            L_meas: Unused by the current linear models. Kept for compatibility
                with other/older configurations.
            model_a: Optional pre-constructed model for player A.
            model_b: Optional pre-constructed model for player B.
        """
        self.game_layout = game_layout
        self.explore: bool = False
        self._playerA: Optional[TrainableAssistedPlayerA] = None
        self._playerB: Optional[TrainableAssistedPlayerB] = None
        self.has_prev: bool = True
        # Typically: (measurements_per_layer, outcomes_per_layer).
        self.previous: Any | None = None

        # Build default models if needed.
        #
        # Note: p_rule/num_iterations/hidden_dim/L_meas are included for forward
        # compatibility with future architectures. The linear models currently
        # depend only on `field_size`, `comms_size`, and SR settings.
        if model_a is None:
            self.model_a = LinTrainableAssistedModelA(
                field_size=int(getattr(game_layout, "field_size")),
                comms_size=int(getattr(game_layout, "comms_size")),
                # SR = shared resource. The implementation supports different SR
                # modes; this module uses the stochastic mode.
                sr_mode="sample",
                seed=123,
            )
        else:
            self.model_a = model_a

        if model_b is None:
            self.model_b = LinTrainableAssistedModelB(
                field_size=int(getattr(game_layout, "field_size")),
                comms_size=int(getattr(game_layout, "comms_size")),
                sr_mode="sample",
                seed=123,
            )
        else:
            self.model_b = model_b

    def check_model_correspondence(self) -> bool:
        """Check that model A and model B appear dimensionally compatible.

        This checks `field_size` and `comms_size` when those attributes are
        available on both models.

        Returns:
            True if basic dimensions match (or cannot be checked), otherwise
            False.
        """
        try:
            return (
                int(getattr(self.model_a, "field_size")) == int(getattr(self.model_b, "field_size"))
                and int(getattr(self.model_a, "comms_size")) == int(getattr(self.model_b, "comms_size"))
            )
        except Exception:
            # If the models don't expose these attributes, assume compatibility.
            return True

    def players(self) -> Tuple[TrainableAssistedPlayerA, TrainableAssistedPlayerB]:
        """Return the (player A, player B) wrappers.

        The wrappers are created lazily and cached so that state (e.g.,
        `previous`, log-probabilities) persists across calls until `reset()`.

        Returns:
            A tuple `(player_a, player_b)`.
        """
        if self._playerA is None:
            self._playerA = TrainableAssistedPlayerA(self.game_layout, model_a=self.model_a)
            self._playerA.explore = self.explore
            self._playerA.parent = self

        if self._playerB is None:
            self._playerB = TrainableAssistedPlayerB(self.game_layout, model_b=self.model_b)
            self._playerB.explore = self.explore
            self._playerB.parent = self

        return (self._playerA, self._playerB)

    def reset(self) -> None:
        """Reset per-game state.

        This clears cached `previous` tensors and forwards the reset to any
        already-instantiated player wrappers.
        """
        if self._playerA is not None:
            self._playerA.reset()
        if self._playerB is not None:
            self._playerB.reset()
        self.previous = None

    def set_explore(self, flag: bool) -> None:
        """Enable or disable exploration for both players.

        Args:
            flag: If True, players may sample actions and store log-probabilities
                (when supported by the underlying player/model). If False, they
                act greedily.
        """
        self.explore = bool(flag)
        if self._playerA is not None:
            self._playerA.explore = self.explore
        if self._playerB is not None:
            self._playerB.explore = self.explore