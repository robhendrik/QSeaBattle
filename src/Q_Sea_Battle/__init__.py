"""
QSeaBattle package initialisation (clean layered public API).

Design goals:
- Keep core gameplay API always importable.
- Avoid importing TensorFlow / heavy / experimental modules at import-time.
- Provide lazy access to optional components via __getattr__.
- Avoid name collisions (e.g., Lin vs Pyr imitation utilities).

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

from .simple_players import SimplePlayers
from .majority_players import MajorityPlayers

from .pr_assisted import PRAssisted
from .pr_assisted_layer import PRAssistedLayer
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
_LAZY: Dict[str, Tuple[str, str]] = {
    # Neural net players (TF)
    "NeuralNetPlayers": (".neural_net_players", "NeuralNetPlayers"),
    "NeuralNetPlayerA": (".neural_net_player_a", "NeuralNetPlayerA"),
    "NeuralNetPlayerB": (".neural_net_player_b", "NeuralNetPlayerB"),

    # Trainable assisted (TF)
    "TrainableAssistedPlayers": (".trainable_assisted_players", "TrainableAssistedPlayers"),
    "TrainableAssistedPlayerA": (".trainable_assisted_player_a", "TrainableAssistedPlayerA"),
    "TrainableAssistedPlayerB": (".trainable_assisted_player_b", "TrainableAssistedPlayerB"),

    # Lin trainable assisted models + layers (TF)
    "LinTrainableAssistedModelA": (".lin_trainable_assisted_model_a", "LinTrainableAssistedModelA"),
    "LinTrainableAssistedModelB": (".lin_trainable_assisted_model_b", "LinTrainableAssistedModelB"),
    "LinMeasurementLayerA": (".lin_measurement_layer_a", "LinMeasurementLayerA"),
    "LinCombineLayerA": (".lin_combine_layer_a", "LinCombineLayerA"),
    "LinMeasurementLayerB": (".lin_measurement_layer_b", "LinMeasurementLayerB"),
    "LinCombineLayerB": (".lin_combine_layer_b", "LinCombineLayerB"),

    # Pyr trainable assisted models + layers (TF)
    "PyrTrainableAssistedModelA": (".pyr_trainable_assisted_model_a", "PyrTrainableAssistedModelA"),
    "PyrTrainableAssistedModelB": (".pyr_trainable_assisted_model_b", "PyrTrainableAssistedModelB"),
    "PyrMeasurementLayerA": (".pyr_measurement_layer_a", "PyrMeasurementLayerA"),
    "PyrCombineLayerA": (".pyr_combine_layer_a", "PyrCombineLayerA"),
    "PyrMeasurementLayerB": (".pyr_measurement_layer_b", "PyrMeasurementLayerB"),
    "PyrCombineLayerB": (".pyr_combine_layer_b", "PyrCombineLayerB"),

    # Neural-net imitation utilities (TF)
    "make_segments": (".neural_net_imitation_utilities", "make_segments"),
    "compute_majority_comm": (".neural_net_imitation_utilities", "compute_majority_comm"),
    "generate_majority_dataset_model_a": (".neural_net_imitation_utilities", "generate_majority_dataset_model_a"),
    "generate_majority_dataset_model_b": (".neural_net_imitation_utilities", "generate_majority_dataset_model_b"),

    # Lin imitation utilities (avoid collisions by prefixing)
    "lin_generate_measurement_dataset_a": (".lin_trainable_assisted_imitation_utilities", "generate_measurement_dataset_a"),
    "lin_generate_measurement_dataset_b": (".lin_trainable_assisted_imitation_utilities", "generate_measurement_dataset_b"),
    "lin_generate_combine_dataset_a": (".lin_trainable_assisted_imitation_utilities", "generate_combine_dataset_a"),
    "lin_generate_combine_dataset_b": (".lin_trainable_assisted_imitation_utilities", "generate_combine_dataset_b"),
    "lin_to_tf_dataset": (".lin_trainable_assisted_imitation_utilities", "to_tf_dataset"),
    "transfer_layer_weights": (".lin_trainable_assisted_imitation_utilities", "transfer_layer_weights"),

    # Pyr imitation utilities (avoid collisions by prefixing)
    "pyr_generate_measurement_dataset_a": (".pyr_trainable_assisted_imitation_utilities", "generate_measurement_dataset_a"),
    "pyr_generate_measurement_dataset_b": (".pyr_trainable_assisted_imitation_utilities", "generate_measurement_dataset_b"),
    "pyr_generate_combine_dataset_a": (".pyr_trainable_assisted_imitation_utilities", "generate_combine_dataset_a"),
    "pyr_generate_combine_dataset_b": (".pyr_trainable_assisted_imitation_utilities", "generate_combine_dataset_b"),
    "pyr_to_tf_dataset": (".pyr_trainable_assisted_imitation_utilities", "to_tf_dataset"),
    "transfer_pyr_model_b_layer_weights": (".pyr_trainable_assisted_imitation_utilities", "transfer_pyr_model_b_layer_weights"),
    "transfer_pyr_model_a_layer_weights": (".pyr_trainable_assisted_imitation_utilities", "transfer_pyr_model_a_layer_weights"),
}


def __getattr__(name: str) -> Any:
    """
    Lazy attribute resolver.

    This keeps `import Q_Sea_Battle` stable even if optional ML modules are missing
    or temporarily broken. Import errors will occur only when accessing that symbol.
    """
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod_path, attr = spec
    mod = import_module(mod_path, package=__name__)
    value = getattr(mod, attr)
    globals()[name] = value  # cache for future access
    return value


def __dir__() -> list[str]:
    # Expose a friendly dir() including lazy exports
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
