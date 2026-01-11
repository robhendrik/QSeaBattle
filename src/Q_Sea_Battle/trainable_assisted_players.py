
"""Trainable assisted player wiring (A/B) for QSeaBattle.

This module defines a Players-style wrapper that wires together:
- TrainableAssistedPlayerA: produces communication bits and stores "previous" tensors
- TrainableAssistedPlayerB: consumes the stored tensors and decides a shoot action

The contract for `previous` is:
    previous == (measurements_per_layer, outcomes_per_layer)
where both entries are Python lists of tensors, each tensor shaped (B, n2).

Author: QSeabattle (generated)
Version: 0.1
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

try:
    # These base classes exist in the project.
    from .players import Players  # type: ignore
except Exception:  # pragma: no cover
    class Players:  # minimal fallback for unit tests
        """Fallback Players base class (used only if project base class is unavailable)."""
        has_log_probs: bool = False

from .trainable_assisted_player_a import TrainableAssistedPlayerA
from .trainable_assisted_player_b import TrainableAssistedPlayerB

from .lin_trainable_assisted_model_a import LinTrainableAssistedModelA
from .lin_trainable_assisted_model_b import LinTrainableAssistedModelB


class TrainableAssistedPlayers(Players):
    """Wires TrainableAssistedPlayerA and TrainableAssistedPlayerB.

    This class owns the two Keras models and provides two Player-like wrappers that
    match the Tournament/Game interfaces.

    Public attributes:
        game_layout: GameLayout-like object with attributes `field_size` and `comms_size`.
        model_a: LinTrainableAssistedModelA
        model_b: LinTrainableAssistedModelB
        explore: Shared exploration flag (False = greedy, True = sampling).
        previous: Stores (meas_list, out_list) set by player A, consumed by player B.
    """

    has_log_probs: bool = True

    def __init__(
        self,
        game_layout: Any,
        p_high: float = 0.9,
        num_iterations: Optional[int] = None,
        hidden_dim: int = 32,
        L_meas: Optional[int] = None,
        model_a: Optional[LinTrainableAssistedModelA] = None,
        model_b: Optional[LinTrainableAssistedModelB] = None,
    ) -> None:
        self.game_layout = game_layout
        self.explore: bool = False
        self._playerA: Optional[TrainableAssistedPlayerA] = None
        self._playerB: Optional[TrainableAssistedPlayerB] = None
        self.has_prev: bool = True
        self.previous: Any | None = None  # typically (measurements_per_layer, outcomes_per_layer)

        # Build default models if needed.
        # Note: p_high/num_iterations/hidden_dim/L_meas are included for forward compatibility
        # with future architectures; the Lin models currently use field_size/comms_size + shared-resource (SR) settings.
        if model_a is None:
            self.model_a = LinTrainableAssistedModelA(
                field_size=int(getattr(game_layout, "field_size")),
                comms_size=int(getattr(game_layout, "comms_size")),
                # sr_mode: SR = shared resource (not \"shared randomness\").

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
        """Check that model A and B are compatible.

        Returns:
            True if basic dimensions match, otherwise False.
        """
        try:
            return (
                int(getattr(self.model_a, "field_size")) == int(getattr(self.model_b, "field_size"))
                and int(getattr(self.model_a, "comms_size")) == int(getattr(self.model_b, "comms_size"))
            )
        except Exception:
            # If the models don't expose these attrs, we assume they are compatible.
            return True

    def players(self) -> Tuple[TrainableAssistedPlayerA, TrainableAssistedPlayerB]:
        """Return (PlayerA, PlayerB) wrappers.

        Creates the wrappers lazily and keeps references so state (previous/logprobs)
        persists across calls until reset().

        Returns:
            Tuple (player_a, player_b).
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
        """Reset internal state between games."""
        if self._playerA is not None:
            self._playerA.reset()
        if self._playerB is not None:
            self._playerB.reset()
        self.previous = None

    def set_explore(self, flag: bool) -> None:
        """Set exploration flag for both players.

        Args:
            flag: If True, players sample actions and store log-probs.
        """
        self.explore = bool(flag)
        if self._playerA is not None:
            self._playerA.explore = self.explore
        if self._playerB is not None:
            self._playerB.explore = self.explore
