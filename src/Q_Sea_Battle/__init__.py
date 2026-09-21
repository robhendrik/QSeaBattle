# Author: Rob Hendriks

"""
QSeaBattle package initialization.

This module defines a layered public API:

- Layer 0 (core): Lightweight gameplay and baseline components that should remain
  importable without optional ML dependencies.
- Layer 1+ (optional): Heavier and/or experimental components (e.g., TensorFlow-
  based models, imitation utilities, dataset tooling) that are exposed via lazy
  imports to keep package import-time stable.

Lazy imports are implemented via ``__getattr__`` (PEP 562). Optional modules are
imported only when their exported symbols are accessed, so failures in optional
dependencies surface at access time rather than at package import time.

Author: Rob Hendriks
Version: 0.2 (layered API)
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

# -----------------------------------------------------------------------------
# Layer 0: Core API (eager imports; should be lightweight and always stable)
# -----------------------------------------------------------------------------
from .game_layout import GameLayout
from .game_env import GameEnv
from .players_base import Players
from .player_base_a import PlayerA
from .player_base_b import PlayerB
from .game import Game
from .tournament import Tournament
from .tournament_log import TournamentLog
from .gameplay_adapters import GameplayModelAAdapter, GameplayModelBAdapter
from .simple_players import SimplePlayers
from .majority_players import MajorityPlayers

from .pr_assisted import PRAssisted
#from .pr_assisted_layer import PRAssistedLayer
from .pr_assisted_players import PRAssistedPlayers
from .pr_assisted_player_a import PRAssistedPlayerA
from .pr_assisted_player_b import PRAssistedPlayerB

from .reference_performance_utilities import (
    binary_entropy,
    binary_entropy_reverse,
    expected_win_rate_simple,
    expected_win_rate_majority,
    expected_win_rate_assisted,
    limit_from_mutual_information,
)

from .logit_utilities import logit_to_prob, logit_to_logprob
from .dru_utilities import dru_train, dru_execute


# -----------------------------------------------------------------------------
# Layer 1+: Optional/ML/Imitation API (lazy-loaded)
# -----------------------------------------------------------------------------
# Mapping: exported_name -> (module_path, attribute_name)
#
# Notes:
# - Keys are the public names exposed from this package namespace.
# - Values identify where the implementation lives and which attribute to load.
# - Imports are deferred to keep base gameplay usable without ML dependencies.
_LAZY: Dict[str, Tuple[str, str]] = {
    # Neural net players (TF)
    "NeuralNetPlayers": (".neural_net_players", "NeuralNetPlayers"),
    "NeuralNetPlayerA": (".neural_net_player_a", "NeuralNetPlayerA"),
    "NeuralNetPlayerB": (".neural_net_player_b", "NeuralNetPlayerB"),

    # Trainable assisted (TF)
    "TrainableAssistedPlayers": (".trainable_assisted_players", "TrainableAssistedPlayers"),
    "TrainableAssistedPlayerA": (".trainable_assisted_player_a", "TrainableAssistedPlayerA"),
    "TrainableAssistedPlayerB": (".trainable_assisted_player_b", "TrainableAssistedPlayerB"),

    # LIN modules
        "LinMeasurementLayerA": (".lin_measurement_layer_a", "LinMeasurementLayerA"),
        "LinMeasurementLayerB": (".lin_measurement_layer_b", "LinMeasurementLayerB"),
        "LinCombineLayerA": (".lin_combine_layer_a", "LinCombineLayerA"),
        "LinCombineLayerB": (".lin_combine_layer_b", "LinCombineLayerB"),
        "LinInternalModelA": (".lin_internal_model_a", "LinInternalModelA"),
        "LinInternalModelB": (".lin_internal_model_b", "LinInternalModelB"),
        "LinTrainableAssistedModelA": (".lin_trainable_assisted_model_a", "LinTrainableAssistedModelA"),
        "LinTrainableAssistedModelB": (".lin_trainable_assisted_model_b", "LinTrainableAssistedModelB"),
    

    # PR/PYR modules
        "PRAssistedReplay": (".pr_assisted_replay", "PRAssistedReplay"),
        "PyrMeasurementLayerA": (".pyr_measurement_layer_a", "PyrMeasurementLayerA"),
        "PyrMeasurementLayerB": (".pyr_measurement_layer_b", "PyrMeasurementLayerB"),
        "PyrCombineLayerA": (".pyr_combine_layer_a", "PyrCombineLayerA"),
        "PyrCombineLayerB": (".pyr_combine_layer_b", "PyrCombineLayerB"),
        "PyrInternalModelA": (".pyr_internal_model_a", "PyrInternalModelA"),
        "PyrInternalModelB": (".pyr_internal_model_b", "PyrInternalModelB"),

    # Neural-net imitation utilities (TF)
    "make_segments": (".neural_net_imitation_utilities", "make_segments"),
    "compute_majority_comm": (".neural_net_imitation_utilities", "compute_majority_comm"),
    "generate_majority_dataset_model_a": (".neural_net_imitation_utilities", "generate_majority_dataset_model_a"),
    "generate_majority_dataset_model_b": (".neural_net_imitation_utilities", "generate_majority_dataset_model_b"),

    # LIN dataset utilities
        "generate_lin_dataset": (".lin_dataset_generation_utilities", "generate_lin_dataset"),
        "convert_lin_layer_measure_a": (".lin_dataset_conversion_utilities", "convert_layer_measure_a"),
        "convert_lin_layer_combine_a": (".lin_dataset_conversion_utilities", "convert_layer_combine_a"),
        "convert_lin_layer_measure_b": (".lin_dataset_conversion_utilities", "convert_layer_measure_b"),
        "convert_lin_layer_combine_b": (".lin_dataset_conversion_utilities", "convert_layer_combine_b"),
        "convert_lin_internal_model_a": (".lin_dataset_conversion_utilities", "convert_internal_model_a"),
        "convert_lin_internal_model_b": (".lin_dataset_conversion_utilities", "convert_internal_model_b"),
        "convert_lin_full_system": (".lin_dataset_conversion_utilities", "convert_full_system"),

    # PYR dataset utilities
        "generate_pyr_dataset": (".pyr_dataset_generation_utilities", "generate_pyr_dataset"),
        "save_npz": (".pyr_dataset_generation_utilities", "save_npz"),
        "convert_layer_measure_a": (".pyr_dataset_conversion_utilities", "convert_layer_measure_a"),
        "convert_layer_combine_a": (".pyr_dataset_conversion_utilities", "convert_layer_combine_a"),
        "convert_layer_measure_b": (".pyr_dataset_conversion_utilities", "convert_layer_measure_b"),
        "convert_layer_combine_b": (".pyr_dataset_conversion_utilities", "convert_layer_combine_b"),
        "convert_internal_model_a": (".pyr_dataset_conversion_utilities", "convert_internal_model_a"),
        "convert_internal_model_b": (".pyr_dataset_conversion_utilities", "convert_internal_model_b"),
        "convert_full_system": (".pyr_dataset_conversion_utilities", "convert_full_system"),
}


def __getattr__(name: str) -> Any:
    """Resolve lazily exported attributes.

    This keeps ``import Q_Sea_Battle`` robust even when optional ML modules are
    not installed or are temporarily broken. Import errors (or other exceptions
    raised during import) occur only when the corresponding symbol is accessed.

    Args:
        name: Attribute name requested from the package namespace.

    Returns:
        The resolved attribute from the lazily imported module.

    Raises:
        AttributeError: If ``name`` is not part of the eagerly imported public
            API and is not registered as a lazy export.
    """
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod_path, attr = spec
    mod = import_module(mod_path, package=__name__)
    value = getattr(mod, attr)
    globals()[name] = value  # Cache for subsequent attribute access.
    return value


def __dir__() -> list[str]:
    """Return a directory listing that includes lazy exports."""
    # Expose a friendly dir() including lazy exports.
    return sorted(set(list(globals().keys()) + list(_LAZY.keys())))


# -----------------------------------------------------------------------------
# Public exports
# -----------------------------------------------------------------------------
__all__ = [
    # Core
    "GameLayout",
    "GameEnv",
    "Players",
    "PlayerA",
    "PlayerB",
    "Game",
    "Tournament",
    "TournamentLog",
    # Baselines
    "SimplePlayers",
    "MajorityPlayers",
    # Classical assisted
    "PRAssisted",
    "PRAssistedLayer",
    "PRAssistedPlayers",
    "PRAssistedPlayerA",
    "PRAssistedPlayerB",
    # Reference / analytic utilities
    "binary_entropy",
    "binary_entropy_reverse",
    "expected_win_rate_simple",
    "expected_win_rate_majority",
    "expected_win_rate_assisted",
    "limit_from_mutual_information",
    # Logit helpers + DRU
    "logit_to_prob",
    "logit_to_logprob",
    "dru_train",
    "dru_execute",
    # Lazy exports (optional layers)
    *sorted(_LAZY.keys()),
]