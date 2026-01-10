
"""QSeaBattle package initialisation.

Author: Rob Hendriks
Version: 0.1
"""

"""Top-level package for the QSeaBattle framework.

This module exposes the main public classes and reference utilities
for convenient import.
"""

# Core game infrastructure
from .game_layout import GameLayout
from .game_env import GameEnv
from .players_base import Players, PlayerA, PlayerB
from .game import Game
from .tournament import Tournament
from .tournament_log import TournamentLog

# Deterministic baselines
from .simple_players import SimplePlayers
from .majority_players import MajorityPlayers

# Classical assisted (non-trainable)
from .pr_assisted import PRAssisted
from .pr_assisted_players import PRAssistedPlayers 
from .pr_assisted_player_a import PRAssistedPlayerA
from .pr_assisted_player_b import PRAssistedPlayerB
from .assisted_players import AssistedPlayers # deprecated
from .shared_randomness import SharedRandomness # deprecated

# Neural net players (no shared randomness)
from .neural_net_players import NeuralNetPlayers
from .neural_net_player_a import NeuralNetPlayerA
from .neural_net_player_b import NeuralNetPlayerB

# Shared randomness (Keras layer)
from .shared_randomness_layer import SharedRandomnessLayer

# Trainable assisted container + players
from .trainable_assisted_players import TrainableAssistedPlayers
from .trainable_assisted_player_a import TrainableAssistedPlayerA
from .trainable_assisted_player_b import TrainableAssistedPlayerB

# Linear (Lin) trainable assisted models + layers
from .lin_trainable_assisted_model_a import LinTrainableAssistedModelA
from .lin_trainable_assisted_model_b import LinTrainableAssistedModelB
from .lin_measurement_layer_a import LinMeasurementLayerA
from .lin_combine_layer_a import LinCombineLayerA
from .lin_measurement_layer_b import LinMeasurementLayerB
from .lin_combine_layer_b import LinCombineLayerB

# Pyramid (Pyr) trainable assisted models + layers (placeholders now, real later)
from .pyr_trainable_assisted_model_a import PyrTrainableAssistedModelA
from .pyr_trainable_assisted_model_b import PyrTrainableAssistedModelB
from .pyr_measurement_layer_a import PyrMeasurementLayerA
from .pyr_combine_layer_a import PyrCombineLayerA
from .pyr_measurement_layer_b import PyrMeasurementLayerB
from .pyr_combine_layer_b import PyrCombineLayerB

# Reference / analytic utilities
from .reference_performance_utilities import (
    binary_entropy,
    binary_entropy_reverse,
    expected_win_rate_simple,
    expected_win_rate_majority,
    expected_win_rate_assisted,
    limit_from_mutual_information,
)

# Logit helpers + DRU
from .logit_utils import logit_to_prob, logit_to_logprob
from .dru_utils import dru_train, dru_execute

# Imitation utilities
from .neural_net_imitation_utils import (
    make_segments,
    compute_majority_comm,
    generate_majority_dataset_model_a,
    generate_majority_dataset_model_b,
)

# Trainable-assisted imitation utilities (Lin/Pyr)
from .lin_trainable_assisted_imitation_utils import (
    generate_measurement_dataset_a,
    generate_measurement_dataset_b,
    generate_combine_dataset_a,
    generate_combine_dataset_b,
    to_tf_dataset,
    transfer_layer_weights,
)




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
    "SharedRandomness",
    "AssistedPlayers",
    # Neural net players
    "NeuralNetPlayers",
    "NeuralNetPlayerA",
    "NeuralNetPlayerB",
    # Shared randomness layer
    "SharedRandomnessLayer",
    # Trainable assisted container + players
    "TrainableAssistedPlayers",
    "TrainableAssistedPlayerA",
    "TrainableAssistedPlayerB",
    # Lin trainable assisted
    "LinTrainableAssistedModelA",
    "LinTrainableAssistedModelB",
    "LinMeasurementLayerA",
    "LinCombineLayerA",
    "LinMeasurementLayerB",
    "LinCombineLayerB",
    # Pyr trainable assisted
    "PyrTrainableAssistedModelA",
    "PyrTrainableAssistedModelB",
    "PyrMeasurementLayerA",
    "PyrCombineLayerA",
    "PyrMeasurementLayerB",
    "PyrCombineLayerB",
    # Reference performance
    "binary_entropy",
    "binary_entropy_reverse",
    "expected_win_rate_simple",
    "expected_win_rate_majority",
    "expected_win_rate_assisted",
    "limit_from_mutual_information",
    # Logit/DRU utils
    "logit_to_prob",
    "logit_to_logprob",
    "dru_train",
    "dru_execute",
    # Majority imitation utils
    "make_segments",
    "compute_majority_comm",
    "generate_majority_dataset_model_a",
    "generate_majority_dataset_model_b",
    # Lin imitation utils
    "generate_measurement_dataset_a",
    "generate_measurement_dataset_b",
    "generate_combine_dataset_a",
    "generate_combine_dataset_b",
    "to_tf_dataset",
    "transfer_layer_weights",
]

